"""Real-data check of the perception step: does the sensor model change the grasp planner's output
the way real depth does?

ROBI (Yang et al., IROS 2021) gives real RealSense D415 depth of a bin of metal parts plus ground-truth
object and camera poses. Per view we render the CAD parts at their GT poses (ideal depth, true instance
masks) and run the same grasp planner as in the grasp experiment on
    real      real D415 depth,
    ideal     rendered depth,
    sim       our sensor model on the rendered depth, D415 geometry, metal/edge tables and blob size
              from scene 4 only (scene 5 is the holdout),
    sim minus one effect, and the simpler noise models (iid, z^2, Dex-Net).
There is no robot, so outcomes are perception-level: valid depth on the parts, how many parts the
depth-only clustering separates (instance recall at IoU 0.5), which part the planner picks, and the
grasp pose error against the planner on ideal depth with the true mask of the same part.
A noise model is useful if its numbers are close to the real ones, not to the ideal ones.

Setup choices (identical for every condition):
  - Only pixels of annotated parts are used (real depth is masked with the rendered instance mask):
    bin walls and the bin floor are not in the GT, and leaving them in would favour or penalise one
    side. The table plane is the known world plane z = 0 (ROBI world frame), no RANSAC fit.
  - Sensor models run at the D415 resolution (1280x720, where the tables were measured); the planner
    then sees every 2nd pixel (640x360, f = 453 px), close to the D435 setting of the grasp experiment
    (f = 424 px), so its pixel thresholds keep their meaning.
  - Real depth contains GT alignment error (~1 mm, see configs/camera_d435.yaml); simulated depth does not.

Usage: python scripts/robi_grasp_check.py --scene 5          (needs scripts/robi_fetch.py --scene 5)
       python scripts/robi_grasp_check.py --scene 5 --report-only
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib
import mujoco
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robi_d415_metal import build_model, load_view_json, set_camera  # noqa: E402

from synthrgbd.perception.grasp_planner import PARAMS, clusters_from_depth, plan_grasp  # noqa: E402
from synthrgbd.render.mujoco_renderer import render_gbuffer  # noqa: E402
from synthrgbd.sensor.baselines import dexnet_noise, iid_gauss, z2_gauss  # noqa: E402
from synthrgbd.sensor.d435 import _v, build_d435, load_config  # noqa: E402
from synthrgbd.sensor.pipeline import SensorModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["real", "ideal", "sim", "sim_edge_const", "sim_no_metal_dropout", "sim_no_edge_artifacts",
              "iid_gauss", "z2_gauss", "dexnet"]
F_D435 = 424.0
MIN_PX = 300        # parts with fewer visible pixels (planner resolution) are not scored
MIN_VIS = 0.5       # ROBI visibility; lower ones may be hidden by the (not modelled) bin wall
PLANE = (np.array([0.0, 0.0, 1.0]), 0.0)


def _d415(B, f, corr_px, edge_by_step):
    """D435 model with the D415 geometry; metal/edge tables are D415 measurements already."""
    eff = []
    for name, fn, p in build_d435([-1], metal_corr_px=corr_px, edge_by_step=edge_by_step).effects:
        if name == "range":
            continue    # ROBI depth uses a disparity shift, every part depth (0.3-0.45 m) is in range
        p = dict(p)
        if "baseline_m" in p:
            p["baseline_m"] = B
        if name == "disparity_noise":
            p["corr_px"] = p["corr_px"] * f / F_D435     # unverified correlation length, scaled with f
        eff.append((name, fn, p))
    return SensorModel(eff)


def build_sensors(B, f, corr_px, z_ref):
    sim = _d415(B, f, corr_px, edge_by_step=True)
    sigma_px = _v(load_config()["stereo"]["subpixel_rms_px"])
    dn = yaml.safe_load((ROOT / "configs" / "noise_baselines.yaml").read_text())["dexnet"]
    return {
        "sim": sim, "sim_edge_const": _d415(B, f, corr_px, edge_by_step=False),
        "sim_no_metal_dropout": sim.without("metal_dropout"),
        "sim_no_edge_artifacts": sim.without("edge_artifacts"),
        "iid_gauss": SensorModel([("iid_gauss", iid_gauss, {"sigma_m": z_ref ** 2 * sigma_px / (f * B)})]),
        "z2_gauss": SensorModel([("z2_gauss", z2_gauss, {"sigma_px": sigma_px, "baseline_m": B})]),
        "dexnet": SensorModel([("dexnet", dexnet_noise, {k: dn[k] for k in
                                                         ("gamma_shape", "gamma_scale", "gp_sigma_m",
                                                          "gp_bandwidth_px")})]),
    }


_W = {}


def _worker_init(cfg):
    root, scene = Path(cfg["root"]), Path(cfg["scene_dir"])
    cam = cfg["cam"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    obj_poses = {k.lower(): np.array(v).reshape(4, 4)
                 for k, v in json.loads((scene / "GT_world2obj.json").read_text()).items()}
    model = build_model(root / "_CAD_models" / f"{cfg['object']}.stl", obj_poses, K, cam["width"], cam["height"])
    _W.update(model=model, data=mujoco.MjData(model), cfg=cfg,
              geom_to_obj={mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n): int(n.split("_")[1])
                           for n in obj_poses},
              sensors=build_sensors(cam["baseline"] / 1000.0, cam["fx"], cfg["corr_px"], cfg["z_ref"]))


def _err(g, ref):
    d = (g.yaw - ref.yaw) % np.pi
    return {"xy_mm": float(np.hypot(g.x - ref.x, g.y - ref.y) * 1000), "yaw_deg": float(np.degrees(min(d, np.pi - d))),
            "z_top_mm": float((g.z_top - ref.z_top) * 1000), "width_mm": float((g.width - ref.width) * 1000)}


def run_view(job):
    idx, view = job
    model, data, cfg = _W["model"], _W["data"], _W["cfg"]
    scene = Path(cfg["scene_dir"])
    T, vis = load_view_json(scene / f"RealSense/Depth/DEPTH_{view}.json")
    set_camera(model, data, T)
    g = render_gbuffer(model, data, "rs", cfg["cam"]["width"], cfg["cam"]["height"])
    real = cv2.imread(str(scene / f"RealSense/Depth/DEPTH_{view}.png"), cv2.IMREAD_ANYDEPTH)
    parts = g.instance >= 0
    depths = {"real": np.where(parts, real.astype(np.float32) * cfg["cam"]["depth_unit"] / 1000.0, 0.0),
              "ideal": g.depth.copy()}
    for name, s in _W["sensors"].items():
        depths[name] = s.apply(g, np.random.default_rng([idx, 1]))

    # Planner resolution: every 2nd pixel (pixel centre u_full = 2 u_sub -> f and c halve).
    sl = (slice(None, None, 2), slice(None, None, 2))
    K = g.K.copy()
    K[:2] /= 2
    gs = replace(g, depth=g.depth[sl], instance=g.instance[sl], material=g.material[sl],
                 normals=g.normals[sl], rgb=g.rgb[sl], K=K)
    inst = gs.instance
    objs = []
    for gid, oid in _W["geom_to_obj"].items():
        m = inst == gid
        if m.sum() >= MIN_PX and vis.get(oid, 0.0) >= MIN_VIS:
            objs.append((oid, m))
    part_of = np.full(inst.shape, -1, np.int32)
    for gid, oid in _W["geom_to_obj"].items():
        part_of[inst == gid] = oid

    def rng():
        return np.random.default_rng([idx, 2])

    ref = {oid: plan_grasp(gs.depth, gs, rng(), instance_masks=[m], plane=PLANE)[0] for oid, m in objs}
    rows = []
    for cond in CONDITIONS:
        d = depths[cond][sl]
        row = {"view": view, "condition": cond, "n_objs": len(objs),
               "valid_frac": float((d[inst >= 0] > 0).mean())}
        # Segmentation: depth-only clusters vs true masks.
        P_h = np.full(d.shape, np.nan)
        valid = d > 0
        # height above the known plane = world z of the back-projected point
        u, v = np.meshgrid(np.arange(d.shape[1]), np.arange(d.shape[0]))
        pc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
        P_h[valid] = (pc[valid] @ g.T_world_cam[:3, :3].T + g.T_world_cam[:3, 3])[:, 2]
        clusters = clusters_from_depth(P_h, d, PARAMS)
        found = 0
        for _, m in objs:
            best = max((float((m & c).sum() / (m | c).sum()) for c in clusters), default=0.0)
            found += best >= 0.5
        row.update(n_clusters=len(clusters), recall50=found / max(len(objs), 1))
        # Depth-only planner (as in the grasp experiment).
        grasp, info = plan_grasp(d, gs, rng(), plane=PLANE)
        row["plan"] = info["reason"]
        if grasp is not None:
            ids = part_of[info["region"]]
            ids = ids[ids >= 0]
            tgt = int(np.bincount(ids).argmax()) if len(ids) else -1
            row.update(target=tgt, purity=float((ids == tgt).mean()) if len(ids) else 0.0)
            if ref.get(tgt) is not None:
                row["err"] = _err(grasp, ref[tgt])
        # Perfect segmenter: grasp every scored part from its true mask.
        per_obj = {}
        for oid, m in objs:
            go, _ = plan_grasp(d, gs, rng(), instance_masks=[m], plane=PLANE)
            per_obj[oid] = None if (go is None or ref[oid] is None) else _err(go, ref[oid])
        row["per_obj"] = per_obj
        rows.append(row)
    return rows


# ---------------------------------------------------------------- summary

METRICS = [  # key, label, pipeline
    ("valid_frac", "valid depth on parts", "both"),
    ("recall50", "parts separated by depth clusters (IoU >= 0.5)", "depth"),
    ("xy_med", "grasp centre error, median [mm]", "depth"),
    ("obj_xy_med", "grasp centre error per part, median [mm]", "masks"),
    ("obj_xy_p90", "grasp centre error per part, p90 [mm]", "masks"),
    ("obj_ztop_med", "top height error per part, median [mm]", "masks"),
]


def _view_metrics(rows_c, rows_real):
    """Per-view metric vectors for one condition (arrays aligned over views)."""
    out = {"valid_frac": [], "recall50": [], "same_target_as_real": [], "xy": [], "obj_xy": [],
           "obj_ztop": [], "obj_fail": []}
    for r, rr in zip(rows_c, rows_real):
        out["valid_frac"].append(r["valid_frac"])
        out["recall50"].append(r["recall50"])
        out["same_target_as_real"].append(float(r.get("target", -2) == rr.get("target", -3)))
        out["xy"].append(r["err"]["xy_mm"] if "err" in r else np.nan)
        e = [v for v in r["per_obj"].values() if v is not None]
        out["obj_xy"].append([v["xy_mm"] for v in e])
        out["obj_ztop"].append([v["z_top_mm"] for v in e])
        out["obj_fail"].append(np.mean([v is None for v in r["per_obj"].values()]) if r["per_obj"] else np.nan)
    return out


def _stat(key, vm, idx):
    if key in ("valid_frac", "recall50", "same_target_as_real", "obj_fail"):
        return float(np.nanmean(np.asarray(vm[key])[idx]))
    if key == "xy_med":
        return float(np.nanmedian(np.asarray(vm["xy"])[idx]))
    lists = [vm["obj_xy" if "xy" in key else "obj_ztop"][i] for i in idx]
    flat = np.concatenate([np.asarray(x, float) for x in lists]) if lists else np.array([])
    if flat.size == 0:
        return np.nan
    return float(np.median(flat) if key.endswith("med") else np.percentile(np.abs(flat), 90))


def summarise(rows, rng, n_boot=2000):
    by = {}
    for r in rows:
        by.setdefault(r["condition"], {})[r["view"]] = r
    views = sorted(by["real"], key=lambda v: int(v.split("_")[1]))
    vm = {c: _view_metrics([by[c][v] for v in views], [by["real"][v] for v in views]) for c in CONDITIONS}
    n = len(views)
    boots = [rng.integers(0, n, n) for _ in range(n_boot)]
    table = {}
    for c in CONDITIONS:
        table[c] = {}
        for key, _, _ in METRICS:
            val = _stat(key, vm[c], np.arange(n))
            # Gap to real, paired bootstrap over views (views of one scene are correlated -> optimistic CI).
            gaps = [_stat(key, vm[c], b) - _stat(key, vm["real"], b) for b in boots]
            table[c][key] = {"value": val, "gap_to_real": val - _stat(key, vm["real"], np.arange(n)),
                             "gap_ci": np.nanpercentile(gaps, [2.5, 97.5]).tolist()}
    return table, n


def plot(table, path, title):
    cols = [m for m in METRICS]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
    BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
    conds = CONDITIONS[1:]
    ys = np.arange(len(conds))
    for ax, (key, label, _) in zip(axes.flat, cols):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, length=0, labelsize=8)
        real = table["real"][key]["value"]
        ax.axvline(real, color=ORANGE, lw=2)
        for y, c in zip(ys, conds):
            t = table[c][key]
            lo, hi = real + t["gap_ci"][0], real + t["gap_ci"][1]
            ax.plot([lo, hi], [y, y], color=BLUE, lw=2, alpha=0.4, solid_capstyle="round")
            ax.plot(t["value"], y, "o", ms=6, color=BLUE, mec="#fcfcfb", mew=1.5)
        ax.set_title(label, fontsize=8.5, loc="left")
    axes[0, 0].set_yticks(ys, conds, fontsize=8)
    axes[1, 0].set_yticks(ys, conds, fontsize=8)
    axes[0, 0].invert_yaxis()
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    axes[0, 0].plot([], [], color=ORANGE, lw=2, label="real D415")
    axes[0, 0].plot([], [], "o", color=BLUE, label="model (bar: 95 % CI of gap to real)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, facecolor="#fcfcfb")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/robi")
    ap.add_argument("--object", default="Zigzag")
    ap.add_argument("--scene", type=int, default=5)
    ap.add_argument("--calib-scene", type=int, default=4, help="scene whose blob-size fit is used")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--n", type=int, help="only the first n views (smoke test)")
    a = ap.parse_args()
    root = Path(a.root)
    tag = f"{a.object.lower()}_s{a.scene}"
    out = ROOT / "results" / "robi" / f"{tag}_grasp_check.jsonl"

    if not a.report_only:
        cam = yaml.safe_load((root / "toolkit/cam/RealSense_DEPTH.yml").read_text())
        blobs = json.loads((ROOT / "results" / "robi" / f"{a.object.lower()}_s{a.calib_scene}_blobs.json").read_text())
        scene_dir = root / a.object / f"Scene_{a.scene}"
        views = (scene_dir / "RealSense/list.txt").read_text().split()[:a.n]
        # Reference depth for the amplitude-matched iid model: median camera-to-parts distance.
        obj = np.array([np.array(v).reshape(4, 4)[:3, 3] for v in
                        json.loads((scene_dir / "GT_world2obj.json").read_text()).values()]).mean(0)
        z_ref = float(np.median([np.linalg.norm(load_view_json(scene_dir / f"RealSense/Depth/DEPTH_{v}.json")[0][:3, 3]
                                                - obj) for v in views]) / 1000.0)
        cfg = {"root": str(root), "scene_dir": str(scene_dir), "object": a.object, "cam": cam,
               "corr_px": blobs["corr_px_fit_d415"], "z_ref": z_ref}
        print(f"corr_px {cfg['corr_px']:.2f} (D415 px, from scene {a.calib_scene}), z_ref {z_ref:.3f} m")
        t0 = time.time()
        with open(out, "w") as f, ProcessPoolExecutor(a.workers, initializer=_worker_init, initargs=(cfg,)) as ex:
            for i, rows in enumerate(ex.map(run_view, list(enumerate(views)))):
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                print(f"[{i + 1}/{len(views)}] {rows[0]['view']}  elapsed {time.time() - t0:.0f}s", flush=True)

    rows = [json.loads(line) for line in open(out)]
    table, n = summarise(rows, np.random.default_rng(0))
    (ROOT / "results" / "robi" / f"{tag}_grasp_check_summary.json").write_text(json.dumps(table, indent=1))
    print(f"{n} views")
    print(f"{'metric':45s}" + "".join(f"{c[:12]:>13s}" for c in CONDITIONS))
    for key, label, _ in METRICS:
        print(f"{label[:45]:45s}" + "".join(f"{table[c][key]['value']:13.3f}" for c in CONDITIONS))
    plot(table, ROOT / "figures" / f"robi_{tag}_grasp_check.png",
         f"Perception step on real vs simulated depth, ROBI {a.object} scene {a.scene} ({n} views)")


if __name__ == "__main__":
    main()

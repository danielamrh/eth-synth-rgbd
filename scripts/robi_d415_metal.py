"""Measure real RealSense D415 depth errors on metal parts (ROBI) against CAD-rendered ground truth.

ROBI provides D415 depth maps, D415 camera poses and object poses, but ground-truth depth only
for the Ensenso views. We therefore render the CAD models at their ground-truth poses into each
D415 depth view with our own MuJoCo renderer and compare per pixel.

Only objects with ROBI visibility >= --min-vis are evaluated, because the bin is not modelled
and could otherwise occlude rendered parts.

Usage: python scripts/robi_fetch.py --object Zigzag --scene 4
       python scripts/robi_d415_metal.py --object Zigzag --scene 4
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

from synthrgbd.render.mujoco_renderer import camera_xml_attrs, render_gbuffer
from synthrgbd.sensor import SensorModel
from synthrgbd.sensor.d435 import build_d435
from synthrgbd.sensor.effects.stereo import stereo_shadow
from synthrgbd.sensor.fields import edge_band, incidence_deg

_CV_TO_MJ = np.diag([1.0, -1.0, -1.0])
ANGLE_BINS = np.arange(0, 91, 10)


def load_view_json(path: Path):
    """Camera pose (mm, OpenCV, camera-in-world) and per-object visibility from a ROBI view file."""
    text = path.read_text()
    pose = np.array(json.loads(re.search(r'\{"cam_pose".*?\}', text).group(0))["cam_pose"]).reshape(4, 4)
    vis = {int(k): float(v) for k, v in re.findall(r'"object_(\d+)_visibility":\s*([0-9.eE+-]+)', text)}
    return pose, vis


def build_model(stl: Path, obj_poses: dict, K: np.ndarray, W: int, H: int):
    bodies = []
    for name, T in obj_poses.items():
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
        p = T[:3, 3] / 1000.0
        bodies.append(f'<body name="{name}" pos="{p[0]} {p[1]} {p[2]}" quat="{" ".join(map(str, quat))}">'
                      f'<geom name="{name}" type="mesh" mesh="part"/></body>')
    xml = f"""<mujoco>
      <visual><global offwidth="{W}" offheight="{H}"/><map znear="0.001" zfar="100"/></visual>
      <asset><mesh name="part" file="{stl.resolve().as_posix()}" scale=".001 .001 .001"/></asset>
      <worldbody><camera name="rs" {camera_xml_attrs(K, W, H)}/>{"".join(bodies)}</worldbody>
    </mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def set_camera(model, data, T_world_cam_mm: np.ndarray):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (T_world_cam_mm[:3, :3] @ _CV_TO_MJ).flatten())
    model.cam_pos[0] = T_world_cam_mm[:3, 3] / 1000.0
    model.cam_quat[0] = quat
    mujoco.mj_forward(model, data)


def bin_stats(err_mm, invalid, angle, bins):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (angle >= lo) & (angle < hi)
        n = int(m.sum())
        if n == 0:
            continue
        valid = m & ~invalid
        e = err_mm[valid]
        rows.append(dict(angle_lo=int(lo), angle_hi=int(hi), n_px=n,
                         invalid_rate=float(invalid[m].mean()),
                         median_abs_err_mm=float(np.median(np.abs(e))) if e.size else np.nan,
                         mean_signed_err_mm=float(e.mean()) if e.size else np.nan,
                         median_signed_err_mm=float(np.median(e)) if e.size else np.nan,
                         mad_sigma_mm=float(1.4826 * np.median(np.abs(e - np.median(e)))) if e.size else np.nan,
                         frac_err_gt_2mm=float((np.abs(e) > 2).mean()) if e.size else np.nan,
                         frac_err_gt_5mm=float((np.abs(e) > 5).mean()) if e.size else np.nan))
    return rows


def simulated_d415(baseline_m: float) -> SensorModel:
    """Geometry + metal + edge effects from configs/camera_d435.yaml with the D415 baseline.

    The ROBI meshes carry no material, so every rendered object pixel (material id -1) is metal.
    Disparity noise is left out: the real errors are dominated by the GT alignment floor anyway.
    """
    full = build_d435([-1])
    sim = SensorModel([e for e in full.effects if e[0] in ("stereo_shadow", "metal_dropout",
                                                          "edge_artifacts", "metal_depth_error")])
    sim.effects[0][2]["baseline_m"] = baseline_m
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/robi")
    ap.add_argument("--object", default="Zigzag")
    ap.add_argument("--scene", type=int, default=4)
    ap.add_argument("--min-vis", type=float, default=0.95)
    ap.add_argument("--edge-px", type=int, default=3)
    ap.add_argument("--out", default="results/robi")
    ap.add_argument("--simulate", action="store_true",
                    help="replace the real D415 depth by our sensor model applied to the rendered GT "
                         "(holdout check of the calibrated tables)")
    args = ap.parse_args()

    root = Path(args.root)
    scene = root / args.object / f"Scene_{args.scene}"
    cam = yaml.safe_load((root / "toolkit/cam/RealSense_DEPTH.yml").read_text())
    W, H = int(cam["width"]), int(cam["height"])
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    obj_poses = {k.lower(): np.array(v).reshape(4, 4)
                 for k, v in json.loads((scene / "GT_world2obj.json").read_text()).items()}
    model = build_model(root / "_CAD_models" / f"{args.object}.stl", obj_poses, K, W, H)
    data = mujoco.MjData(model)
    geom_to_obj = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n): int(n.split("_")[1]) for n in obj_poses}

    sensor = simulated_d415(cam["baseline"] / 1000.0) if args.simulate else None
    views = (scene / "RealSense/list.txt").read_text().split()
    acc = {k: [] for k in ("err", "invalid", "angle", "edge", "z")}
    per_view, example = [], None
    n_shadow = n_good = 0
    for view in views:
        T, vis = load_view_json(scene / f"RealSense/Depth/DEPTH_{view}.json")
        set_camera(model, data, T)
        g = render_gbuffer(model, data, "rs", W, H)
        real = cv2.imread(str(scene / f"RealSense/Depth/DEPTH_{view}.png"), cv2.IMREAD_ANYDEPTH)
        real_m = real.astype(np.float32) * cam["depth_unit"] / 1000.0
        if sensor is not None:
            real_m = sensor.apply(g, np.random.default_rng(int(view.split("_")[1])))

        good = np.zeros_like(g.instance, bool)
        for gid, oid in geom_to_obj.items():
            if vis.get(oid, 0.0) >= args.min_vis:
                good |= g.instance == gid
        # Measure net of geometry: pixels our stereo_shadow predicts as hidden from the right imager
        # are excluded, so the tables only describe dropouts beyond what the shadow effect produces.
        shadow = (stereo_shadow(g.depth, g, {"baseline_m": cam["baseline"] / 1000.0}, None) == 0) & good
        n_shadow += int(shadow.sum())
        n_good += int(good.sum())
        good &= ~shadow
        if not good.any():
            continue
        invalid = real_m == 0
        err_mm = (real_m - g.depth) * 1000.0
        edge = edge_band(g.instance, args.edge_px)
        angle = incidence_deg(g)
        for k, v in (("err", err_mm), ("invalid", invalid), ("angle", angle), ("edge", edge), ("z", g.depth)):
            acc[k].append(v[good])
        interior = good & ~edge
        e = err_mm[interior & ~invalid]
        per_view.append(dict(view=view, n_px=int(interior.sum()), invalid_rate=float(invalid[interior].mean()),
                             median_signed_err_mm=float(np.median(e)) if e.size else np.nan))
        if example is None or interior.sum() > example[0]:
            example = (interior.sum(), view, real_m, g.depth, good)

    a = {k: np.concatenate(v) for k, v in acc.items()}
    interior = ~a["edge"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_int = bin_stats(a["err"][interior], a["invalid"][interior], a["angle"][interior], ANGLE_BINS)
    rows_edge = bin_stats(a["err"][a["edge"]], a["invalid"][a["edge"]], a["angle"][a["edge"]], np.array([0, 91]))
    valid_int = interior & ~a["invalid"]
    summary = dict(
        object=args.object, scene=args.scene, n_views=len(per_view), min_visibility=args.min_vis,
        excluded_predicted_shadow_frac=n_shadow / max(n_good, 1),
        z_range_m=[float(np.percentile(a["z"], 1)), float(np.percentile(a["z"], 99))],
        interior=dict(n_px=int(interior.sum()), invalid_rate=float(a["invalid"][interior].mean()),
                      median_abs_err_mm=float(np.median(np.abs(a["err"][valid_int]))),
                      median_signed_err_mm=float(np.median(a["err"][valid_int])),
                      frac_err_gt_5mm=float((np.abs(a["err"][valid_int]) > 5).mean())),
        edge=rows_edge[0], by_angle_interior=rows_int,
        per_view_median_signed_err_mm=dict(
            p10=float(np.nanpercentile([r["median_signed_err_mm"] for r in per_view], 10)),
            p50=float(np.nanpercentile([r["median_signed_err_mm"] for r in per_view], 50)),
            p90=float(np.nanpercentile([r["median_signed_err_mm"] for r in per_view], 90))),
    )
    tag = f"{args.object.lower()}_s{args.scene}" + ("_sim" if args.simulate else "")
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    (out / f"{tag}_per_view.json").write_text(json.dumps(per_view, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_angle_interior"}, indent=2))
    print("angle  n_px    invalid  med|e|mm  mean_signed  med_signed  mad_sigma  >5mm")
    for r in rows_int:
        print(f"{r['angle_lo']:2d}-{r['angle_hi']:2d} {r['n_px']:7d}  {r['invalid_rate']:.3f}   "
              f"{r['median_abs_err_mm']:6.2f}   {r['mean_signed_err_mm']:6.2f}  {r['median_signed_err_mm']:6.2f}  "
              f"{r['mad_sigma_mm']:6.2f}  {r['frac_err_gt_5mm']:.3f}")

    # Figure: rates vs angle + example view
    _, view, real_m, gt, good = example
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    mids = [(r["angle_lo"] + r["angle_hi"]) / 2 for r in rows_int]
    ax[0].plot(mids, [r["invalid_rate"] for r in rows_int], "o-", label="invalid (depth = 0)")
    ax[0].plot(mids, [r["frac_err_gt_5mm"] for r in rows_int], "s-", label="valid but |err| > 5 mm")
    ax[0].set_xlabel("incidence angle [deg]"); ax[0].set_ylabel("fraction of object pixels")
    ax[0].set_title(f"D415 on {args.object} (interior px)"); ax[0].legend()
    roi = np.ix_(np.any(good, 1), np.any(good, 0))
    vmin, vmax = np.percentile(gt[good], [1, 99])
    ax[1].imshow(np.where(real_m > 0, real_m, np.nan)[roi], vmin=vmin, vmax=vmax); ax[1].set_title(f"real D415 depth ({view})")
    ax[2].imshow(np.where(gt > 0, gt, np.nan)[roi], vmin=vmin, vmax=vmax); ax[2].set_title("CAD-rendered GT depth")
    err = np.where(good & (real_m > 0), (real_m - gt) * 1000, np.nan)
    im = ax[3].imshow(err[roi], cmap="coolwarm", vmin=-10, vmax=10); ax[3].set_title("error [mm] (evaluated objects)")
    fig.colorbar(im, ax=ax[3])
    for a_ in ax[1:]:
        a_.axis("off")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/robi_{tag}.png", dpi=110)
    print(f"saved figures/robi_{tag}.png and {out}/{tag}_summary.json")


if __name__ == "__main__":
    main()

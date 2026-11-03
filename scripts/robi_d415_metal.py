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


def incidence_deg(depth: np.ndarray, normals: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(depth)], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(normals * ray, -1)), 0, 1)))


def edge_band(instance: np.ndarray, width: int) -> np.ndarray:
    """Pixels within `width` px of a change in instance id (incl. object/background)."""
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    inst = (instance + 1).astype(np.float32)
    return cv2.dilate(inst, k) != cv2.erode(inst, k)


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
                         frac_err_gt_2mm=float((np.abs(e) > 2).mean()) if e.size else np.nan,
                         frac_err_gt_5mm=float((np.abs(e) > 5).mean()) if e.size else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/robi")
    ap.add_argument("--object", default="Zigzag")
    ap.add_argument("--scene", type=int, default=4)
    ap.add_argument("--min-vis", type=float, default=0.95)
    ap.add_argument("--edge-px", type=int, default=3)
    ap.add_argument("--out", default="results/robi")
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

    views = (scene / "RealSense/list.txt").read_text().split()
    acc = {k: [] for k in ("err", "invalid", "angle", "edge", "z")}
    per_view, example = [], None
    for view in views:
        T, vis = load_view_json(scene / f"RealSense/Depth/DEPTH_{view}.json")
        set_camera(model, data, T)
        g = render_gbuffer(model, data, "rs", W, H)
        real = cv2.imread(str(scene / f"RealSense/Depth/DEPTH_{view}.png"), cv2.IMREAD_ANYDEPTH)
        real_m = real.astype(np.float32) * cam["depth_unit"] / 1000.0

        good = np.zeros_like(g.instance, bool)
        for gid, oid in geom_to_obj.items():
            if vis.get(oid, 0.0) >= args.min_vis:
                good |= g.instance == gid
        if not good.any():
            continue
        invalid = real_m == 0
        err_mm = (real_m - g.depth) * 1000.0
        edge = edge_band(g.instance, args.edge_px)
        angle = incidence_deg(g.depth, g.normals, g.K)
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
    tag = f"{args.object.lower()}_s{args.scene}"
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    (out / f"{tag}_per_view.json").write_text(json.dumps(per_view, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_angle_interior"}, indent=2))
    print("angle  n_px    invalid  med|e|mm  signed  >5mm")
    for r in rows_int:
        print(f"{r['angle_lo']:2d}-{r['angle_hi']:2d} {r['n_px']:7d}  {r['invalid_rate']:.3f}   "
              f"{r['median_abs_err_mm']:6.2f}   {r['mean_signed_err_mm']:6.2f}  {r['frac_err_gt_5mm']:.3f}")

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

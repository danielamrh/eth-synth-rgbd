"""Measure how real D415 dropouts at part boundaries depend on the depth step across the boundary.

Found with scripts/robi_grasp_check.py: a constant edge dropout rate (p_drop, measured over all
silhouette bands) drops too much where two parts touch, so depth-only clustering separates simulated
parts that stay merged in real depth. Here the silhouette band (3 px, as in the rate tables) of the
rendered parts is split by the local depth step (max - min of rendered part depth in a 7x7 window),
and band pixels next to unmodelled background (bin, floor) form their own class. Rates are net of the
predicted stereo shadow, like the other tables.

Usage: python scripts/robi_edge_steps.py --scene 4 [--step 2]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robi_d415_metal import build_model, load_view_json, set_camera  # noqa: E402

from synthrgbd.render.mujoco_renderer import render_gbuffer  # noqa: E402
from synthrgbd.sensor.effects.stereo import stereo_shadow  # noqa: E402
from synthrgbd.sensor.fields import edge_band, local_depth_step  # noqa: E402

STEP_EDGES_MM = [0, 2.5, 5, 10, 15, 25, 40, np.inf]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/robi")
    ap.add_argument("--object", default="Zigzag")
    ap.add_argument("--scene", type=int, default=4)
    ap.add_argument("--edge-px", type=int, default=3)
    ap.add_argument("--step", type=int, default=1, help="use every n-th view")
    ap.add_argument("--out", default="results/robi")
    a = ap.parse_args()
    root = Path(a.root)
    scene = root / a.object / f"Scene_{a.scene}"
    cam = yaml.safe_load((root / "toolkit/cam/RealSense_DEPTH.yml").read_text())
    W, H = int(cam["width"]), int(cam["height"])
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    obj_poses = {k.lower(): np.array(v).reshape(4, 4)
                 for k, v in json.loads((scene / "GT_world2obj.json").read_text()).items()}
    model = build_model(root / "_CAD_models" / f"{a.object}.stl", obj_poses, K, W, H)
    data = mujoco.MjData(model)
    B = cam["baseline"] / 1000.0

    n_bins = len(STEP_EDGES_MM) - 1
    n, inv = np.zeros(n_bins + 1), np.zeros(n_bins + 1)      # last entry: next to background
    views = (scene / "RealSense/list.txt").read_text().split()[::a.step]
    for i, view in enumerate(views):
        T, _ = load_view_json(scene / f"RealSense/Depth/DEPTH_{view}.json")
        set_camera(model, data, T)
        g = render_gbuffer(model, data, "rs", W, H)
        real = cv2.imread(str(scene / f"RealSense/Depth/DEPTH_{view}.png"), cv2.IMREAD_ANYDEPTH)
        parts = g.instance >= 0
        band = edge_band(g.instance, a.edge_px) & parts
        band &= stereo_shadow(g.depth, g, {"baseline_m": B}, None) > 0
        step_mm, next_to_bg = local_depth_step(g.depth, a.edge_px)
        step_mm = step_mm * 1000
        cls = np.digitize(step_mm, STEP_EDGES_MM[1:-1])
        cls[next_to_bg] = n_bins
        for k in range(n_bins + 1):
            m = band & (cls == k)
            n[k] += m.sum()
            inv[k] += (m & (real == 0)).sum()
        print(f"[{i + 1}/{len(views)}] {view}", flush=True)

    rows = [{"step_lo_mm": float(lo), "step_hi_mm": float(hi), "n_px": int(n[k]), "invalid_rate": float(inv[k] / n[k])}
            for k, (lo, hi) in enumerate(zip(STEP_EDGES_MM[:-1], STEP_EDGES_MM[1:]))]
    summary = {"object": a.object, "scene": a.scene, "n_views": len(views), "edge_px": a.edge_px,
               "by_step": rows, "next_to_background": {"n_px": int(n[-1]), "invalid_rate": float(inv[-1] / n[-1])},
               "all_band": {"n_px": int(n.sum()), "invalid_rate": float(inv.sum() / n.sum())}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.object.lower()}_s{a.scene}_edge_steps.json"
    path.write_text(json.dumps(summary, indent=1))
    for r in rows:
        print(f"step {r['step_lo_mm']:5.1f}-{r['step_hi_mm']:5.1f} mm  n={r['n_px']:8d}  invalid {r['invalid_rate']:.3f}")
    print(f"next to background      n={int(n[-1]):8d}  invalid {inv[-1] / n[-1]:.3f}")
    print(f"all band                n={int(n.sum()):8d}  invalid {inv.sum() / n.sum():.3f}")
    print("wrote", path)


if __name__ == "__main__":
    main()

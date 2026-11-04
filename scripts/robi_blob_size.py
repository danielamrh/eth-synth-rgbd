"""Measure how clustered real D415 dropouts on metal are, and fit the blob size corr_px of metal_dropout.

Statistic: conditional dropout probability P(r) = P(pixel at offset r invalid | pixel invalid),
over pairs of interior object pixels (silhouette band and predicted stereo shadow excluded, as for
the rate tables), averaged over horizontal and vertical offsets. Salt-and-pepper dropouts give
P(r) = base rate for all r >= 1; blobs give P(r) > base rate that decays with r.

The simulation applies only metal_dropout to the same rendered views, so the dropout rate per
incidence angle is the measured one and only corr_px differs. corr_px is fitted by least squares on
P(r), r = 1..R. Result is in D415 pixels at 1280x720 (f = 906 px).

Usage: python scripts/robi_blob_size.py --scene 4 [--step 2]
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import mujoco
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robi_d415_metal import build_model, load_view_json, set_camera  # noqa: E402

from synthrgbd.render.mujoco_renderer import render_gbuffer  # noqa: E402
from synthrgbd.sensor.d435 import build_d435  # noqa: E402
from synthrgbd.sensor.effects.stereo import stereo_shadow  # noqa: E402
from synthrgbd.sensor.fields import edge_band  # noqa: E402

R_MAX = 15
CORR = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
SEEDS = 2


def pair_counts(invalid, interior, r_max=R_MAX):
    """Counts n_inv[r] (invalid pixels whose partner at offset r is interior) and n_both[r]."""
    n_inv, n_both = np.zeros(r_max + 1), np.zeros(r_max + 1)
    for r in range(1, r_max + 1):
        for a_sl, b_sl in (((slice(None), slice(0, -r)), (slice(None), slice(r, None))),
                           ((slice(0, -r), slice(None)), (slice(r, None), slice(None)))):
            both = interior[a_sl] & interior[b_sl]
            ia, ib = invalid[a_sl] & both, invalid[b_sl] & both
            n_inv[r] += ia.sum() + ib.sum()           # count both directions
            n_both[r] += 2 * (ia & ib).sum()
    return n_inv, n_both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/robi")
    ap.add_argument("--object", default="Zigzag")
    ap.add_argument("--scene", type=int, default=4)
    ap.add_argument("--min-vis", type=float, default=0.95)
    ap.add_argument("--edge-px", type=int, default=3)
    ap.add_argument("--step", type=int, default=1, help="use every n-th view")
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
    B = cam["baseline"] / 1000.0
    dropout = {c: [e for e in build_d435([-1], metal_corr_px=c).effects if e[0] == "metal_dropout"][0]
               for c in CORR}

    real_acc = np.zeros((2, R_MAX + 1))
    sim_acc = {c: np.zeros((2, R_MAX + 1)) for c in CORR}
    hole_area = {"real": [], **{c: [] for c in CORR}}
    views = (scene / "RealSense/list.txt").read_text().split()[::args.step]
    for i, view in enumerate(views):
        T, vis = load_view_json(scene / f"RealSense/Depth/DEPTH_{view}.json")
        set_camera(model, data, T)
        g = render_gbuffer(model, data, "rs", W, H)
        real = cv2.imread(str(scene / f"RealSense/Depth/DEPTH_{view}.png"), cv2.IMREAD_ANYDEPTH)
        good = np.zeros_like(g.instance, bool)
        for gid, oid in geom_to_obj.items():
            if vis.get(oid, 0.0) >= args.min_vis:
                good |= g.instance == gid
        good &= stereo_shadow(g.depth, g, {"baseline_m": B}, None) > 0
        interior = good & ~edge_band(g.instance, args.edge_px)
        if interior.sum() < 500:
            continue
        inv = (real == 0) & interior
        real_acc += pair_counts(inv, interior)
        n, lab, st, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), connectivity=8)
        hole_area["real"] += st[1:, cv2.CC_STAT_AREA].tolist()
        for c in CORR:
            name, fn, params = dropout[c]
            for s in range(SEEDS):
                sim = fn(g.depth.copy(), g, params, np.random.default_rng([int(view.split("_")[1]), s]))
                sinv = (sim == 0) & interior
                sim_acc[c] += pair_counts(sinv, interior)
                n, lab, st, _ = cv2.connectedComponentsWithStats(sinv.astype(np.uint8), connectivity=8)
                hole_area[c] += st[1:, cv2.CC_STAT_AREA].tolist()
        print(f"[{i + 1}/{len(views)}] {view}", flush=True)

    r = np.arange(1, R_MAX + 1)
    P_real = real_acc[1, 1:] / real_acc[0, 1:]
    P_sim = {c: sim_acc[c][1, 1:] / sim_acc[c][0, 1:] for c in CORR}
    sse = {c: float(((P_sim[c] - P_real) ** 2).sum()) for c in CORR}
    # Refine between grid points with a parabola through the best three.
    cs = np.array(CORR)
    k = int(np.argmin([sse[c] for c in CORR]))
    k = min(max(k, 1), len(CORR) - 2)
    coef = np.polyfit(cs[k - 1:k + 2], [sse[c] for c in cs[k - 1:k + 2]], 2)
    c_fit = float(-coef[1] / (2 * coef[0])) if coef[0] > 0 else float(cs[k])

    def area_stats(v):
        v = np.asarray(v)
        return {"n_holes": int(v.size), "median_px": float(np.median(v)) if v.size else 0.0,
                "area_weighted_mean_px": float((v ** 2).sum() / v.sum()) if v.size else 0.0}

    summary = {
        "object": args.object, "scene": args.scene, "n_views": len(views), "resolution": [W, H],
        "f_px": float(cam["fx"]), "r_px": r.tolist(), "P_real": P_real.tolist(),
        "P_sim": {str(c): P_sim[c].tolist() for c in CORR}, "sse": {str(c): sse[c] for c in CORR},
        "corr_px_fit_d415": c_fit,
        "holes": {"real": area_stats(hole_area["real"]), **{str(c): area_stats(hole_area[c]) for c in CORR}},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.object.lower()}_s{args.scene}_blobs"
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    print("SSE per corr_px:", {c: round(v, 5) for c, v in sse.items()})
    print(f"fitted corr_px = {c_fit:.2f} px at {W}x{H}")
    print("hole areas:", json.dumps(summary["holes"], indent=0))

    fig, ax = plt.subplots(figsize=(6, 4))
    for c in CORR:
        ax.plot(r, P_sim[c], color="#2a78d6", alpha=0.25 + 0.75 * (c / max(CORR)), lw=1.5)
        ax.text(r[-1] + 0.2, P_sim[c][-1], f"{c:g}", fontsize=7, color="#52514e", va="center")
    ax.plot(r, P_real, "o-", color="#eb6834", lw=2, ms=4, label="real D415")
    ax.plot([], [], color="#2a78d6", label="simulated, corr_px (label)")
    ax.set_xlabel("pixel offset r (D415, 1280x720)")
    ax.set_ylabel("P(invalid at r | invalid)")
    ax.set_title(f"Clustering of metal dropouts, ROBI {args.object} S{args.scene}")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"figures/robi_{tag}.png", dpi=150)
    print(f"saved figures/robi_{tag}.png")


if __name__ == "__main__":
    main()

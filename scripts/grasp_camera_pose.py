"""Where does the camera model matter? Gap ideal - D435 in grasp success vs camera mount.

The camera pose of every scene is drawn from its seed (scene.cell.sample_scene), so it is recovered
here without rerunning anything. Scenes are split into terciles of camera distance and tilt; per bin
the paired success difference ideal - d435 with a 95 % paired bootstrap interval over scenes.

    python scripts/grasp_camera_pose.py --tag main
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from synthrgbd.scene import cell  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def camera_params(seed):
    s = cell.sample_scene(np.random.default_rng(seed))
    pos, tgt = np.array(s["camera"]["pos"]), np.array(s["camera"]["target"])
    v = pos - tgt
    return float(np.linalg.norm(v)), float(np.degrees(np.arccos(v[2] / np.linalg.norm(v))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--a", default="ideal")
    ap.add_argument("--b", default="d435")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(ROOT / "results" / "grasp" / a.tag / "trials.jsonl")]
    ok = {}
    for r in rows:
        if r["pipeline"] == "depth" and r["condition"] in (a.a, a.b):
            ok.setdefault(r["seed"], {})[r["condition"]] = float(r["success"])
    seeds = sorted(s for s in ok if len(ok[s]) == 2)
    x = np.array([ok[s][a.a] for s in seeds])
    y = np.array([ok[s][a.b] for s in seeds])
    cam = np.array([camera_params(s) for s in seeds])
    rng = np.random.default_rng(0)
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, (k, name, unit) in zip(axes, [(0, "camera distance", "m"), (1, "camera tilt from vertical", "deg")]):
        edges = np.percentile(cam[:, k], [0, 100 / 3, 200 / 3, 100])
        b = np.clip(np.digitize(cam[:, k], edges[1:-1]), 0, 2)
        res = []
        for i in range(3):
            m = b == i
            d = x[m] - y[m]
            boot = [d[rng.integers(0, m.sum(), m.sum())].mean() for _ in range(5000)]
            res.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(m.sum()),
                        f"success_{a.a}": float(x[m].mean()), f"success_{a.b}": float(y[m].mean()),
                        "gap": float(d.mean()), "gap_ci": np.percentile(boot, [2.5, 97.5]).tolist()})
        out[name] = res
        labels = [f"{r['lo']:.2f}-{r['hi']:.2f}" if unit == "m" else f"{r['lo']:.0f}-{r['hi']:.0f}" for r in res]
        xs = np.arange(3)
        ax.axhline(0, color=MUTED, lw=1)
        for xi, r in zip(xs, res):
            ax.plot([xi, xi], np.array(r["gap_ci"]) * 100, color=BLUE, lw=2, alpha=0.4, solid_capstyle="round")
            ax.plot(xi, r["gap"] * 100, "o", ms=7, color=BLUE, mec="#fcfcfb", mew=1.5)
            ax.text(xi + 0.08, r["gap"] * 100, f"{r['gap'] * 100:+.1f}", fontsize=8, color=INK, va="center")
        ax.set_xticks(xs, labels, fontsize=8, color=INK)
        ax.set_xlabel(f"{name} [{unit}], terciles of {len(seeds)} scenes", fontsize=8.5, color=INK)
        ax.set_xlim(-0.5, 2.6)
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, length=0, labelsize=8)
    axes[0].set_ylabel(f"success {a.a} - {a.b} [pp]", fontsize=8.5, color=INK)
    fig.tight_layout()
    path = ROOT / "figures" / f"grasp_camera_pose_{a.tag}.png"
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    (ROOT / "results" / "grasp" / a.tag / "camera_pose.json").write_text(json.dumps(out, indent=1))
    for name, res in out.items():
        print(name)
        for r in res:
            print(f"  {r['lo']:.2f}-{r['hi']:.2f}  n={r['n']}  {a.a} {r[f'success_{a.a}']:.3f}  {a.b} {r[f'success_{a.b}']:.3f}"
                  f"  gap {r['gap'] * 100:+.1f} pp [{r['gap_ci'][0] * 100:+.1f}, {r['gap_ci'][1] * 100:+.1f}]")
    print("wrote", path)


if __name__ == "__main__":
    main()

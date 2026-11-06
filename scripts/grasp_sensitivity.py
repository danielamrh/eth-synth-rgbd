"""Which part of the camera model is worth measuring? Tornado plot of grasp success vs the full model.

Each row moves one part of the D435 model and shows the paired success difference to the full
model (95 % paired bootstrap CI over scenes), from results/grasp/<tag>/summary.json:
  - measured ranges (blob size 3.3-6.1 px), model choices (constant edge dropout),
  - stress values without a measured range (metal effects x0.5 / x2),
  - removing one effect entirely.
Rows with a wide bar decide the result and deserve real measurements; rows near 0 do not.

    python scripts/grasp_report.py --tag main && python scripts/grasp_sensitivity.py --tag main
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
ROWS = [  # label, low condition, high condition (None = one-sided)
    ("metal dropout + error x0.5 / x2 (stress)", "metal_x0.5", "metal_x2"),
    ("dropout blob size 3.3 / 6.1 px (measured range)", "metal_corr3.3", "metal_corr6.1"),
    ("edge dropout constant (first model)", "d435_edge_const", None),
    ("remove edge artifacts", "no_edge_artifacts", None),
    ("remove metal dropout", "no_metal_dropout", None),
    ("remove metal depth error", "no_metal_depth_error", None),
    ("remove stereo shadow", "no_stereo_shadow", None),
    ("remove disparity noise", "no_disparity_noise", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    table = json.loads((ROOT / "results" / "grasp" / a.tag / "summary.json").read_text())
    t = {r["condition"]: r for r in table if r["pipeline"] == "depth"}
    rows = [r for r in ROWS if r[1] in t and (r[2] is None or r[2] in t)]
    rows.sort(key=lambda r: -max(abs(t[c]["vs_d435"]["diff"]) for c in r[1:] if c))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.1))
    ax.set_facecolor("#fcfcfb")
    ax.axvline(0, color=MUTED, lw=1)
    gap = t["ideal"]["vs_d435"]["diff"] * 100
    ax.axvline(gap, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(gap, -0.75, " ideal depth", fontsize=7.5, color=MUTED, va="bottom")
    for y, (label, lo, hi) in enumerate(rows):
        for c, col in ((lo, BLUE), (hi, ORANGE)):
            if c is None:
                continue
            v = t[c]["vs_d435"]
            yy = y if hi is None else y + (-0.16 if c == lo else 0.16)
            ax.barh(yy, v["diff"] * 100, height=0.55 if hi is None else 0.3, color=col, alpha=0.85)
            ax.plot(np.array(v["ci"]) * 100, [yy, yy], color=INK, lw=1)
            end = v["ci"][1] * 100 if v["diff"] >= 0 else v["ci"][0] * 100
            ax.text(end + (0.3 if v["diff"] >= 0 else -0.3), yy,
                    f"{v['diff'] * 100:+.1f}", fontsize=7.5, color=INK, va="center",
                    ha="left" if v["diff"] >= 0 else "right")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8, color=INK)
    ax.set_xlabel(f"grasp success minus full D435 model [pp]  (full model: {t['d435']['success'] * 100:.1f} %, "
                  f"ideal: {t['ideal']['success'] * 100:.1f} %)", fontsize=8.5, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax.legend(handles=[Patch(color=BLUE, label="lower value / change"), Patch(color=ORANGE, label="higher value"),
                       plt.Line2D([], [], color=INK, lw=1, label="95 % paired CI")],
              loc="lower left", frameon=False, fontsize=8)
    ax.set_ylim(len(rows) - 0.4, -1.0)
    lo_x = min(min(t[c]["vs_d435"]["ci"][0] for c in r[1:] if c) for r in rows) * 100
    ax.set_xlim(lo_x - 3, None)
    fig.tight_layout()
    path = ROOT / "figures" / f"grasp_sensitivity_{a.tag}.png"
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    print("wrote", path)


if __name__ == "__main__":
    main()

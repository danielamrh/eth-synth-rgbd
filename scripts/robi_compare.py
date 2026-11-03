"""Holdout check: do the metal tables calibrated on ROBI scene 4 describe scene 5?

Compares (a) real D415 on scene 4 (calibration), (b) real D415 on scene 5 (holdout) and
(c) our sensor model applied to the CAD-rendered GT of scene 5.
Run after: robi_d415_metal.py --scene 4, --scene 5, --scene 5 --simulate.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = {"S4 real (calibration)": "zigzag_s4", "S5 real (holdout)": "zigzag_s5", "S5 model": "zigzag_s5_sim"}
METRICS = [("invalid_rate", "dropout rate"), ("median_signed_err_mm", "median signed error [mm]"),
           ("frac_err_gt_5mm", "valid with |err| > 5 mm")]


def weighted_abs_diff(a, b, key):
    w = np.array([r["n_px"] for r in b], float)
    return float(np.sum(w * np.abs(np.array([r[key] for r in a]) - np.array([r[key] for r in b]))) / w.sum())


def main(out=Path("results/robi")):
    s = {k: json.loads((out / f"{v}_summary.json").read_text()) for k, v in RUNS.items()}
    rows = {k: v["by_angle_interior"] for k, v in s.items()}
    real4, real5, sim5 = rows.values()

    print(f"{'':24s} {'interior dropout':>17s} {'median |err| mm':>16s} {'edge dropout':>13s}")
    for k, v in s.items():
        print(f"{k:24s} {v['interior']['invalid_rate']:17.3f} {v['interior']['median_abs_err_mm']:16.2f} "
              f"{v['edge']['invalid_rate']:13.3f}")
    print("\npixel-weighted mean |difference| per angle bin, against S5 real:")
    for key, label in METRICS:
        print(f"  {label:28s} S4 real: {weighted_abs_diff(real4, real5, key):.3f}   "
              f"S5 model: {weighted_abs_diff(sim5, real5, key):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, label) in zip(axes, METRICS):
        for (name, r), style in zip(rows.items(), ["o-", "s-", "^--"]):
            ax.plot([(x["angle_lo"] + x["angle_hi"]) / 2 for x in r], [x[key] for x in r], style, label=name)
        ax.set_xlabel("incidence angle [deg]")
        ax.set_title(label)
    axes[0].legend()
    fig.suptitle("ROBI Zigzag, D415, object interior pixels")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/robi_holdout.png", dpi=110)
    print("saved figures/robi_holdout.png")


if __name__ == "__main__":
    main()

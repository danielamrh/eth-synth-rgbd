"""Summarise the grasp experiment: success rates with CIs, paired tests, geometric grasp errors.

    python scripts/grasp_report.py --tag main

Statistics (all paired over scenes, because every condition sees the same scenes):
  - success rate with a 95 % Wilson interval,
  - difference to a reference condition with a 95 % paired bootstrap interval over scenes,
  - exact McNemar test on the discordant scenes (one condition succeeds, the other fails),
  - grasp error relative to the planner's grasp on ideal depth with the true mask of the same part.
"""
import argparse
import json
from math import comb
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [
    ("References", ["ideal", "d435", "d435_edge_const"]),
    ("D435 minus one effect", ["no_stereo_shadow", "no_metal_dropout", "no_edge_artifacts",
                               "no_metal_depth_error", "no_disparity_noise"]),
    ("Simpler noise models", ["iid_gauss", "z2_gauss", "dexnet"]),
    ("Metal sensitivity", ["metal_x0.5", "metal_x2", "metal_corr0", "metal_corr3.3", "metal_corr6.1"]),
]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on discordant pairs b (A ok, B fail) and c (A fail, B ok)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_boot(a, b, rng, n_boot=5000):
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return np.percentile(d, [2.5, 97.5])


def load(tag):
    rows = [json.loads(line) for line in open(ROOT / "results" / "grasp" / tag / "trials.jsonl")]
    by = {}
    for r in rows:
        by.setdefault((r["pipeline"], r["condition"]), {})[r["seed"]] = r
    return by


def summarise(by, rng):
    seeds = sorted(by[("depth", "ideal")])
    succ = {k: np.array([v[s]["success"] for s in seeds], float) for k, v in by.items()}
    table = []
    for (pipe, cond), v in by.items():
        x = succ[(pipe, cond)]
        k, n = int(x.sum()), len(x)
        row = {"pipeline": pipe, "condition": cond, "n": n, "success": k / n, "ci": wilson(k, n)}
        for ref in ("ideal", "d435"):
            y = succ[(pipe, ref)] if (pipe, ref) in succ else succ[("depth", ref)]
            b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
            row[f"vs_{ref}"] = {"diff": float(x.mean() - y.mean()), "ci": paired_boot(x, y, rng).tolist(),
                                "b": b, "c": c, "p_mcnemar": mcnemar_exact(b, c)}
        errs = [v[s]["err"] for s in seeds if "err" in v[s]]
        row["err_median"] = {key: float(np.median([e[key] for e in errs])) for key in errs[0]} if errs else {}
        row["err_p90"] = {key: float(np.percentile(np.abs([e[key] for e in errs]), 90)) for key in errs[0]} if errs else {}
        row["valid_frac_parts"] = float(np.mean([v[s]["valid_frac_parts"] for s in seeds]))
        row["plan_fail"] = int(sum(v[s]["plan"] != "ok" for s in seeds))
        row["double_pick"] = int(sum(v[s].get("n_lifted", 0) > 1 for s in seeds))
        row["purity"] = float(np.mean([v[s]["cluster_purity"] for s in seeds if "cluster_purity" in v[s]]))
        table.append(row)
    return table


def print_table(table):
    hdr = (f"{'pipeline':7s} {'condition':22s} {'success':>8s} {'95% CI':>13s} {'vs d435':>8s} {'95% CI':>15s}"
           f" {'p':>6s} {'valid':>6s} {'xy med':>7s} {'xy p90':>7s} {'ztop med':>8s} {'purity':>6s}")
    print(hdr)
    for r in table:
        v = r["vs_d435"]
        print(f"{r['pipeline']:7s} {r['condition']:22s} {r['success']:8.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]"
              f" {v['diff']:+8.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] {v['p_mcnemar']:6.3f}"
              f" {r['valid_frac_parts']:6.3f} {r['err_median'].get('xy_mm', np.nan):7.1f}"
              f" {r['err_p90'].get('xy_mm', np.nan):7.1f} {r['err_median'].get('z_top_mm', np.nan):8.2f}"
              f" {r['purity']:6.3f}")


def plot(table, path):
    t = {(r["pipeline"], r["condition"]): r for r in table}
    labels, ys, y = [], [], 0
    group_y = []
    for g, conds in GROUPS:
        group_y.append((g, y))
        y += 0.8
        for c in conds:
            if ("depth", c) in t:
                labels.append(c)
                ys.append(y)
                y += 1
        y += 0.4
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.28 * y + 1.2), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 1]})
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, length=0)
    ax = axes[0]
    ref = t[("depth", "d435")]["success"]
    ax.axvline(ref, color=MUTED, lw=1, ls=(0, (3, 3)))
    for c, yy in zip(labels, ys):
        r = t[("depth", c)]
        ax.plot(r["ci"], [yy, yy], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(r["success"], yy, "o", ms=6, color=BLUE, mec="#fcfcfb", mew=2)
        if ("masks", c) in t:
            m = t[("masks", c)]
            ax.plot(m["ci"], [yy + 0.3] * 2, color=ORANGE, lw=2, solid_capstyle="round")
            ax.plot(m["success"], yy + 0.3, "D", ms=5, color=ORANGE, mec="#fcfcfb", mew=1.5)
    ax.set_xlabel("grasp success rate (95 % Wilson CI)", color=INK)
    ax.plot([], [], "o-", color=BLUE, label="depth-only clusters")
    ax.plot([], [], "D-", color=ORANGE, label="true instance masks + depth")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8, labelcolor=INK)
    ax = axes[1]
    for c, yy in zip(labels, ys):
        r = t[("depth", c)]
        ax.plot([r["err_median"]["xy_mm"], r["err_p90"]["xy_mm"]], [yy, yy], color=BLUE, lw=2, alpha=0.35,
                solid_capstyle="round")
        ax.plot(r["err_median"]["xy_mm"], yy, "o", ms=6, color=BLUE, mec="#fcfcfb", mew=2)
    ax.set_xlabel("grasp centre error [mm], median to p90", color=INK)
    ax.set_xlim(left=0)
    axes[0].set_yticks(ys, labels, color=INK, fontsize=8)
    for g, gy in group_y:
        axes[0].text(-0.02, gy, g, transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                     fontsize=8, color=MUTED, fontweight="bold")
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    table = summarise(load(a.tag), np.random.default_rng(0))
    print_table(table)
    out = ROOT / "results" / "grasp" / a.tag / "summary.json"
    out.write_text(json.dumps(table, indent=1, default=float))
    (ROOT / "figures").mkdir(exist_ok=True)
    plot(table, ROOT / "figures" / f"grasp_{a.tag}.png")


if __name__ == "__main__":
    main()

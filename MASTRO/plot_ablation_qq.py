"""
Q-Q plot of the ablation calibration run (ablation_calibration.py): the
p-values of the real-data family scored on every per-patient null dataset
(pvals_fixed/perm_val), pooled per variant, against the uniform quantiles, on
-log10 scales. A valid test stays on or below the diagonal; above it the test
is anti-conservative. --family mined plots the family mined on each null
dataset instead, which is selected for support and drifts above the diagonal
even under a valid test.

    python3 plot_ablation_qq.py --dir results/ablation/breast_sigma5
"""

import argparse
import csv
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "maestri": ("Multi-MASTRO", "C3"),
    "perm_tree": ("perm. per tree", "C0"),
    "count_trees": ("count trees", "C2"),
    "count_perm_tree": ("count trees + perm. per tree", "C1"),
    "trees_as_patients": ("trees as patients", "C4"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pool", default="perm_val")
    ap.add_argument("--family", choices=["fixed", "mined"], default="fixed")
    args = ap.parse_args()
    d = Path(args.dir)
    sub = "pvals_fixed" if args.family == "fixed" else "pvals"

    pv = {}
    for path in sorted((d / sub / args.pool).glob("*.csv.gz")):
        with gzip.open(path, "rt") as f:
            for row in csv.DictReader(f):
                for v, p in row.items():
                    if v != "pattern":
                        pv.setdefault(v, []).append(max(float(p), 1e-300))
    if not pv:
        raise SystemExit(f"no p-values under {d / sub / args.pool}")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    top = 0.0
    for v, ps in pv.items():
        ps = np.sort(np.array(ps))
        n = len(ps)
        exp = -np.log10(np.arange(1, n + 1) / (n + 1))
        obs = -np.log10(ps)
        name, c = STYLE.get(v, (v, None))
        ax.plot(exp, obs, ".", ms=2, color=c, label=f"{name} ({n} p-values)")
        top = max(top, exp.max())
    ax.plot([0, top], [0, top], color="gray", lw=0.8, ls="--")
    ax.set_xlabel("expected -log10 p (uniform)", fontsize=8)
    ax.set_ylabel("observed -log10 p", fontsize=8)
    ax.set_title(f"Null datasets ({args.pool}), {args.family} family", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, markerscale=4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(d / f"qq_{args.pool}_{args.family}.{ext}", dpi=150)
    print(f"[plot] {d / f'qq_{args.pool}_{args.family}.pdf'}")


if __name__ == "__main__":
    main()

"""
Plot empirical FDR curve from run_wy_correction_ensemble.py output

Reads fdr_exp.csv and/or fdr_theta.csv, plots FDR(delta) vs rank k,
and reports the top-k where FDR <= 0.2

Usage:
    python3 plot_fdr_v2.py \\
        --fdr_exp   results/tracerx_wy_sigma2_theta1.0/fdr_exp.csv \\
        --fdr_theta results/tracerx_wy_sigma2_theta1.0/fdr_theta.csv \\
        --out       results/fdr_plot_sigma2_theta1.pdf \\
        --title     "TRACERx sigma=2 theta=1.0" \\
        --fdr_cutoff 0.2
"""

import argparse
import csv
from pathlib import Path


def load_fdr_csv(path, fdr_col):
    """Return (ranks, pvalues, fdr_values) from a fdr_*.csv file"""
    ranks, pvals, fdrs = [], [], []
    for row in csv.DictReader(open(path)):
        ranks.append(int(row["rank"]))
        pvals.append(float(row["pvalue"]))
        fdrs.append(float(row[fdr_col]))
    return ranks, pvals, fdrs


def main():
    ap = argparse.ArgumentParser(
        description="Plot empirical FDR curve for MASTRO V2 (V2 column names)"
    )
    ap.add_argument("--fdr_exp", default=None,
                    help="fdr_exp.csv from run_wy_correction_ensemble.py")
    ap.add_argument("--fdr_theta", default=None,
                    help="fdr_theta.csv from run_wy_correction_ensemble.py")
    ap.add_argument("--out", required=True, help="Output PDF/PNG path")
    ap.add_argument("--title", default="Empirical FDR", help="Plot title")
    ap.add_argument("--fdr_cutoff", type=float, default=0.2,
                    help="FDR level to highlight in the plot (default: 0.2)")
    args = ap.parse_args()

    if not args.fdr_exp and not args.fdr_theta:
        ap.error("Provide at least one of --fdr_exp, --fdr_theta")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel(r"$k$ most significant results (rank)")
    ax.set_ylabel("Empirical FDR")
    ax.set_title(args.title)
    ax.axhline(args.fdr_cutoff, color="gray", linestyle="--", linewidth=0.8,
               label=f"FDR = {args.fdr_cutoff}")
    ax.grid(which="both", color="0.93", zorder=0)

    for csv_path, col, label, color in [
        (args.fdr_exp,   "fdr_exp",   "exp-support",      "tab:blue"),
        (args.fdr_theta, "fdr_theta", "theta-consensus",  "tab:orange"),
    ]:
        if not csv_path:
            continue
        p = Path(csv_path)
        if not p.exists():
            print(f"[WARN] {p} not found, skipping")
            continue
        ranks, pvals, fdrs = load_fdr_csv(p, col)
        ax.plot(ranks, fdrs, label=label, color=color, marker=".", markersize=3)

        # Report top-k where FDR is below the cutoff. The estimated FDR is not
        # monotone in k, so "largest k below the level" is ambiguous: we report
        # the PREFIX value, the largest k before the curve FIRST crosses the
        # cutoff, so that every prefix of the reported set also satisfies the
        # bound. This is the convention used in the thesis tables. The global
        # maximum is printed alongside when the two differ, since a curve that
        # dips back below the level later would otherwise look like a mismatch.
        prefix_k = 0
        for r, fdr in zip(ranks, fdrs):
            if fdr <= args.fdr_cutoff:
                prefix_k = r
            else:
                break
        global_k = max((r for r, fdr in zip(ranks, fdrs)
                        if fdr <= args.fdr_cutoff), default=0)
        if prefix_k:
            extra = (f"   (global max, curve is non-monotone: {global_k})"
                     if global_k != prefix_k else "")
            print(f"[{label}] max k with FDR <= {args.fdr_cutoff}: {prefix_k}{extra}")
        else:
            print(f"[{label}] no k with FDR <= {args.fdr_cutoff}")

    ax.legend()
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"[OK] Plot saved to {out}")


if __name__ == "__main__":
    main()

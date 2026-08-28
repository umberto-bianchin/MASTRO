"""
Figure for the POTTR vs Multi-MASTRO comparison.

Reads the output of pottr_significance.py and produces a two-panel figure:

  (a) POTTR trajectory size vs support, showing the number of candidate trees
      and the number of DISTINCT patients that carry each trajectory. For the
      large trajectories the two diverge sharply (many trees, one patient)
      which is the signature of counting a single patient's candidate trees as
      independent recurrences.

  (b) POTTR trajectory size vs multi-tree expected support s_exp, with points
      marked by whether the theta-consensus test finds them significant. The
      large trajectories sit at s_exp <= 1 (one patient's worth of weight) and
      are not theta-significant.

Usage:
    python3 plot_pottr_comparison.py \\
        --csv results/pottr_cmp/pottr_significance.csv \\
        --out results/pottr_cmp/pottr_comparison.pdf [--alpha 0.05]
"""

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="pottr_significance.py output CSV")
    ap.add_argument("--out", required=True, help="Output PDF/PNG")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level for the theta-consensus test")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    n = [int(r["n_nodes"]) for r in rows]
    trees = [int(r["support_trees"]) for r in rows]
    pats = [int(r["support_patients"]) for r in rows]
    s_exp = [float(r["s_exp"]) for r in rows]
    th_sig = [bool(r["mastro_pval_theta"]) and float(r["mastro_pval_theta"]) <= args.alpha
              for r in rows]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9, 3.6))

    # ---- Panel (a): support in trees vs in distinct patients, over size ----
    axA.scatter(n, trees, s=28, color="tab:blue", label="candidate trees",
                zorder=3)
    axA.scatter(n, pats, s=28, color="tab:red", marker="s",
                label="distinct patients", zorder=3)
    for xi, yt, yp in zip(n, trees, pats):
        axA.plot([xi, xi], [yp, yt], color="0.75", lw=0.8, zorder=1)
    axA.set_xlabel("POTTR trajectory size (nodes)")
    axA.set_ylabel("support")
    axA.set_title("(a) trees vs distinct patients")
    axA.legend(frameon=False, fontsize=8, loc="upper left")
    axA.grid(color="0.93", zorder=0)

    # ---- Panel (b): size vs multi-tree expected support ----
    for sig, col, lab in [(False, "tab:gray", rf"not $\theta$-sig."),
                          (True, "tab:green", rf"$\theta$-sig. ($p\leq{args.alpha}$)")]:
        xs = [ni for ni, s in zip(n, th_sig) if s == sig]
        ys = [se for se, s in zip(s_exp, th_sig) if s == sig]
        axB.scatter(xs, ys, s=28, color=col, label=lab, zorder=3)
    axB.axhline(1.0, color="0.4", ls="--", lw=0.9,
                label="one patient ($s^{\\exp}=1$)")
    axB.set_xlabel("POTTR trajectory size (nodes)")
    axB.set_ylabel(r"multi-tree expected support $s^{\exp}$")
    axB.set_title("(b) multi-tree support vs size")
    axB.legend(frameon=False, fontsize=8, loc="upper right")
    axB.grid(color="0.93", zorder=0)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[OK] figure -> {out}")


if __name__ == "__main__":
    main()

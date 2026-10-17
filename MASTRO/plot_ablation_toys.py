"""
Plot the ablation toys written by ablation_toys.py: one panel per toy, the
p-value (log scale) against the toy's sweep variable.

    python3 plot_ablation_toys.py --dir results/ablation_toys
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLOOR = 1e-300
STYLE = {
    "maestri": ("MAESTRI", "C3", "o", "-"),
    "perm_tree": ("per-tree null", "C0", "s", "-"),
    "count_trees": ("tree counts", "C2", "^", "-"),
    "count_perm_tree": ("tree counts + per-tree null", "C1", "v", "--"),
    "trees_as_patients": ("pooled trees", "C4", "x", ":"),
    "strict": ("no resolution (MAESTRI)", "C3", "o", "-"),
    "lenient": ("same node = both orders", "C0", "s", "-"),
    "majority": ("same node = majority order", "C1", "v", "-"),
}


def load(path):
    return list(csv.DictReader(path.open()))


def series(rows, variant, pattern, col, toy=None):
    pts = sorted((float(r["x"]), max(float(r[col]), FLOOR)) for r in rows
                 if r["variant"] == variant and r["pattern"] == pattern
                 and (toy is None or r["toy"] == toy) and r[col] != "")
    return [x for x, _ in pts], [p for _, p in pts]


def draw(ax, rows, variants, pattern, col, toy=None, suffix="", ls=None, label=True):
    for v in variants:
        xs, ps = series(rows, v, pattern, col, toy)
        if not xs:
            continue
        name, c, m, l = STYLE[v]
        ax.plot(xs, ps, color=c, marker=m, ls=ls or l, ms=4,
                label=(name + suffix) if label else None)


def finish(ax, title, xlabel, logx=True):
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.axhline(0.05, color="gray", lw=0.8, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("p-value", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/ablation_toys")
    args = ap.parse_args()
    d = Path(args.dir)
    # count_perm_tree has the expected-support law of trees_as_patients: not drawn
    allv = ["maestri", "perm_tree", "count_trees", "trees_as_patients"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    rows = load(d / "shape.csv")
    draw(axes[0, 0], rows, allv, "A->-B", "pval_exp")
    finish(axes[0, 0], "Shape: 100 A->B trees over a patients", "carriers a (100/a trees each)")

    rows = load(d / "competition.csv")
    draw(axes[0, 1], rows, allv, "B->-A", "pval_exp", suffix=": B->A")
    draw(axes[0, 1], rows, ["maestri", "count_trees", "trees_as_patients"], "A->-B",
         "pval_exp", suffix=": A->B", ls=":")
    finish(axes[0, 1], "Competition: 1 patient M x A->B, 100 patients B->A",
           "trees M of the A->B patient")

    rows = load(d / "duplication.csv")
    ax = axes[0, 2]
    for v in ("maestri", "perm_tree"):
        name, c, m, _ = STYLE[v]
        for col, toy, ls, tag in (("pval_exp", "duplication_theta1", "-", "s_exp"),
                                  ("pval_theta", "duplication_theta1", "--", "s_theta, theta=1"),
                                  ("pval_theta", "duplication_any", ":", "s_theta, any tree")):
            xs, ps = series(rows, v, "A->-B", col, toy)
            ax.plot(xs, ps, color=c, marker=m, ls=ls, ms=4, label=f"{name}: {tag}")
    finish(ax, "Duplication: every tree copied M times", "copies M")

    rows = load(d / "ambiguity.csv")
    ax = axes[1, 0]
    for pat, toy, ls, tag, col in (("A->-B", "ambiguity_AB", "-", "{A->B, B->A}: A->B", "pval_theta"),
                                   ("A->-B A->-C B->-C", "ambiguity_ABC", "--", "6 chains: A->B->C", "pval_theta")):
        for v in ("maestri", "perm_tree"):
            name, c, m, _ = STYLE[v]
            xs, ps = series(rows, v, pat, col, toy)
            ax.plot(xs, ps, color=c, marker=m, ls=ls, ms=4, label=f"{name}, {tag}")
    finish(ax, "Ambiguity: every order present (theta = observed pi)", "patients n")

    rows = load(d / "resolution.csv")
    ax = axes[1, 1]
    for pat, ls, tag in (("A->-B", "-", "A->B"), ("B->-A", ":", "B->A")):
        for v in ("strict", "lenient", "majority"):
            name, c, m, _ = STYLE[v]
            xs, ps = series(rows, v, pat, "pval_exp")
            ax.plot(xs, ps, color=c, marker=m, ls=ls, ms=4, label=f"{name}: {tag}")
    finish(ax, "Resolution: 2 A->B, 1 B->A, m with A,B in one node",
           "same-node patients m", logx=False)

    axes[1, 2].axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(d / f"ablation_toys.{ext}", dpi=150)
    print(f"[plot] {d / 'ablation_toys.pdf'}")


if __name__ == "__main__":
    main()

"""
Q-Q plot of the ablation calibration run (ablation_calibration.py): the
p-values of the real-data family scored on every per-patient null dataset
(pvals_fixed/perm_val), pooled per variant, against the uniform quantiles, on
-log10 scales. A valid test stays on or below the diagonal; above it the test
is anti-conservative. --family mined plots the family mined on each null
dataset instead, which is selected for support and drifts above the diagonal
even under a valid test.

The y axis is cut at --ymax (default: 1.3 x the largest expected value), so
one anti-conservative variant does not flatten the others; points beyond it
are drawn as triangles on the edge and counted in the legend. The grey band is the pointwise 95% band of the order
statistics of independent uniform p-values.

The plotted points are also written to qq_<pool>_<family>.csv (thinned in the
bulk, complete in the tail), and the plot is redrawn from that file when the
raw p-values are not available (e.g. on a machine that only has the git copy).

    python3 plot_ablation_qq.py --dir results/ablation/breastCancer_sigma5
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
DEFAULT_VARIANTS = "maestri,count_trees,perm_tree,trees_as_patients"
FLOOR = 1e-300


def load_raw(d, sub, pool):
    pv = {}
    for path in sorted((d / sub / pool).glob("*.csv.gz")):
        with gzip.open(path, "rt") as f:
            for row in csv.DictReader(f):
                for v, p in row.items():
                    if v != "pattern":
                        pv.setdefault(v, []).append(max(float(p), FLOOR))
    return pv


def qq_points(ps, keep_bulk=3000):
    """(expected, observed) in -log10, all points with expected >= 1, the
    bulk thinned to about keep_bulk points."""
    ps = np.sort(np.asarray(ps))
    n = len(ps)
    exp = -np.log10(np.arange(1, n + 1) / (n + 1))
    obs = -np.log10(ps)
    tail = exp >= 1.0
    bulk = np.nonzero(~tail)[0]
    if len(bulk) > keep_bulk:
        bulk = bulk[np.linspace(0, len(bulk) - 1, keep_bulk).astype(int)]
    idx = np.concatenate([np.nonzero(tail)[0], bulk])
    return exp[idx], obs[idx], n


def uniform_band(n, n_points=200, n_sim=400, seed=0):
    """Pointwise 95% band of -log10 order statistics of n uniforms."""
    rng = np.random.default_rng(seed)
    ranks = np.unique(np.geomspace(1, n, n_points).astype(int)) - 1
    sims = np.empty((n_sim, len(ranks)))
    for s in range(n_sim):
        sims[s] = np.sort(rng.random(n))[ranks]
    lo, hi = np.quantile(sims, [0.025, 0.975], axis=0)
    exp = -np.log10((ranks + 1) / (n + 1))
    return exp, -np.log10(hi), -np.log10(lo)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pool", default="perm_val")
    ap.add_argument("--family", choices=["fixed", "mined"], default="fixed")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS,
                    help="count_perm_tree is left out by default: its "
                         "expected-support law equals trees_as_patients")
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    d = Path(args.dir)
    sub = "pvals_fixed" if args.family == "fixed" else "pvals"
    tag = f"qq_{args.pool}_{args.family}"
    variants = [v for v in args.variants.split(",") if v]

    # points: from the raw p-values when present, else from the saved CSV
    pts = {}
    raw = load_raw(d, sub, args.pool)
    if raw:
        rows = []
        for v, ps in raw.items():
            e, o, n = qq_points(ps)
            pts[v] = (e, o, n)
            rows += [(v, n, f"{a:.5f}", f"{b:.5f}") for a, b in zip(e, o)]
        with (d / f"{tag}.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["variant", "n", "expected", "observed"])
            wr.writerows(rows)
    elif (d / f"{tag}.csv").exists():
        acc = {}
        for row in csv.DictReader((d / f"{tag}.csv").open()):
            a = acc.setdefault(row["variant"], ([], [], int(row["n"])))
            a[0].append(float(row["expected"]))
            a[1].append(float(row["observed"]))
        pts = {v: (np.array(e), np.array(o), n) for v, (e, o, n) in acc.items()}
    else:
        raise SystemExit(f"no p-values under {d / sub / args.pool} and no {tag}.csv")

    variants = [v for v in variants if v in pts]
    n = max(pts[v][2] for v in variants)
    top = -np.log10(1 / (n + 1))
    ymax = args.ymax or 1.3 * top

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    be, blo, bhi = uniform_band(n)
    ax.fill_between(be, blo, bhi, color="0.85", lw=0, label="95% band, uniform")
    ax.plot([0, top], [0, top], color="0.4", lw=0.8, ls="--")

    for v in variants:
        e, o, nv = pts[v]
        name, c = STYLE.get(v, (v, None))
        inside = o <= ymax
        label = name
        if (~inside).any():
            label += f"  ({(~inside).sum()} beyond, min p = 1e-{o.max():.0f})"
        ax.plot(e[inside], o[inside], ".", ms=2.5, color=c, label=label)
        if (~inside).any():
            ax.plot(e[~inside], np.full((~inside).sum(), ymax), "^", ms=4,
                    color=c, clip_on=False)

    ax.set_xlim(0, top * 1.03)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"expected $-\log_{10} p$", fontsize=8)
    ax.set_ylabel(r"observed $-\log_{10} p$", fontsize=8)
    ax.set_title(args.title or f"{d.name}: {n} null p-values per variant", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, markerscale=3, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{tag}.{ext}", dpi=200)
    print(f"[plot] {d / (tag + '.pdf')}")


if __name__ == "__main__":
    main()

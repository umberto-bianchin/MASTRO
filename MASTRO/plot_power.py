"""
Plot the power / implanted-trajectory recovery experiment.

Reads the recall.csv files written by implant_experiment_ensemble.py (one per
theta, under <power_dir>/implant_theta<theta>/recall.csv) and draws one panel
for the expected-support test and one per theta for the theta-consensus test:
recall vs per-patient consistency f, one curve per cohort size N. The
expected-support trials are the same in every theta run, so that panel is read
from the first one.

The point the figure makes: the expected-support test recovers the implant
once the expected count N*f is large enough, while the theta-consensus test
recovers it only once f >= theta. A vertical dotted line marks f = theta.

Usage:
    python3 plot_power.py --power_dir results/power --out results/power/power_recall.pdf

Several --power_dir values merge the (N, f) cells of runs over disjoint f grids,
e.g. a later run of the f = theta points only.
"""

import argparse
import csv
from pathlib import Path


def load_recall(csv_paths):
    """recall.csv files -> dict[test][N] = list of (f, recall) sorted by f."""
    by, seen = {}, set()
    for csv_path in csv_paths:
        for r in csv.DictReader(open(csv_path)):
            key = (r["test"], int(r["N"]), float(r["f"]))
            if key in seen:
                raise SystemExit(f"cell {key} appears twice ({csv_path})")
            seen.add(key)
            by.setdefault(r["test"], {}).setdefault(int(r["N"]), []).append(
                (float(r["f"]), float(r["recall"])))
    for test in by:
        for N in by[test]:
            by[test][N].sort()
    return by


def _plot_curves(ax, by_N, plt):
    """One curve per cohort size N, coloured by N."""
    Ns = sorted(by_N)
    cmap = plt.cm.viridis
    for i, N in enumerate(Ns):
        color = cmap(0.08 + 0.84 * i / max(1, len(Ns) - 1))
        fs = [f for f, _ in by_N[N]]
        rc = [r for _, r in by_N[N]]
        ax.plot(fs, rc, marker="o", ms=4, color=color, lw=1.6,
                label=f"N={N}", zorder=3)
    ax.set_xlabel("per-patient consistency $f$")
    ax.set_ylim(-0.03, 1.08)
    ax.grid(True, alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power_dir", nargs="+", default=["results/power"],
                    help="Dir(s) holding implant_theta<theta>/recall.csv")
    ap.add_argument("--out", required=True, help="Output PDF/PNG")
    args = ap.parse_args()

    by_theta = {}
    for pdir in map(Path, args.power_dir):
        for p in pdir.glob("implant_theta*/recall.csv"):
            by_theta.setdefault(float(p.parent.name.split("theta")[1]), []).append(p)
    if not by_theta:
        raise SystemExit(f"no implant_theta*/recall.csv under {args.power_dir}")
    thetas = sorted(by_theta)
    runs = [by_theta[t] for t in thetas]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One panel for the expected-support test (its trials are the same in
    # every theta run, so we read it from the first) and one per theta
    fig, axes = plt.subplots(1, len(runs) + 1,
                             figsize=(4.0 * (len(runs) + 1), 3.6),
                             sharey=True, squeeze=False)
    axes = axes[0]

    exp_by = load_recall(runs[0]).get("exp", {})
    _plot_curves(axes[0], exp_by, plt)
    axes[0].set_title("expected support")

    for ax, theta, csv_paths in zip(axes[1:], thetas, runs):
        by = load_recall(csv_paths)
        _plot_curves(ax, by.get("theta", {}), plt)
        ax.axvline(theta, color="tab:red", ls=":", lw=1.2, zorder=1)
        ax.text(theta, 0.03, r" $f=\theta$", color="tab:red", fontsize=8,
                ha="left", va="bottom")
        ax.set_title(rf"$\theta$-consensus, $\theta = {theta:g}$")

    axes[0].set_ylabel("recall of implanted trajectory")
    # single shared legend below the panels (series are identical across panels)
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = len(labels)
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=ncol,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()

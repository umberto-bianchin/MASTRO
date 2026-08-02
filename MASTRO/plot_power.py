"""
Plot the power / implanted-trajectory recovery experiment.

Reads the recall.csv files written by implant_experiment_ensemble.py (one per
theta, under <power_dir>/implant_theta<theta>/recall.csv) and draws one panel
per theta: recall vs per-patient consistency f, one curve per cohort size N for
the theta-consensus test, plus the expected-support test as a reference.

The point the figure makes: the expected-support test recovers the implant
whenever the expected count N*f is large (recall ~ 1 across the grid), while the
theta-consensus test recovers it only once f >= theta. A vertical dashed line
marks f = theta.

Usage:
    python3 plot_power.py --power_dir results/power --out results/power/power_recall.pdf
"""

import argparse
import csv
from pathlib import Path


def load_recall(csv_path):
    """recall.csv -> dict[test][N] = list of (f, recall) sorted by f."""
    by = {}
    for r in csv.DictReader(open(csv_path)):
        by.setdefault(r["test"], {}).setdefault(int(r["N"]), []).append(
            (float(r["f"]), float(r["recall"])))
    for test in by:
        for N in by[test]:
            by[test][N].sort()
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power_dir", default="results/power",
                    help="Dir holding implant_theta<theta>/recall.csv")
    ap.add_argument("--out", required=True, help="Output PDF/PNG")
    args = ap.parse_args()

    pdir = Path(args.power_dir)
    runs = sorted(pdir.glob("implant_theta*/recall.csv"),
                  key=lambda p: float(p.parent.name.split("theta")[1]))
    if not runs:
        raise SystemExit(f"no implant_theta*/recall.csv under {pdir}")
    thetas = [float(p.parent.name.split("theta")[1]) for p in runs]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3.8),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, theta, csv_path in zip(axes, thetas, runs):
        by = load_recall(csv_path)
        Ns = sorted(by.get("theta", {}))
        cmap = plt.cm.viridis
        colors = {N: cmap(0.08 + 0.84 * i / max(1, len(Ns) - 1))
                  for i, N in enumerate(Ns)}

        # theta-consensus test: one curve per cohort size N, distinct colours
        for N in Ns:
            fs = [f for f, _ in by["theta"][N]]
            rc = [r for _, r in by["theta"][N]]
            ax.plot(fs, rc, marker="o", ms=4, color=colors[N], lw=1.6,
                    label=f"$s^{{(\\theta)}}$, N={N}", zorder=3)

        # expected-support test: reference (identical across N here -> plot once)
        if "exp" in by:
            N0 = sorted(by["exp"])[0]
            fs = [f for f, _ in by["exp"][N0]]
            rc = [r for _, r in by["exp"][N0]]
            ax.plot(fs, rc, marker="s", ms=4, color="tab:blue", lw=1.6,
                    ls="--", label="$s^{\\mathrm{exp}}$", zorder=2)

        ax.axvline(theta, color="tab:red", ls=":", lw=1.2, zorder=1)
        ax.text(theta, 0.03, r" $f=\theta$", color="tab:red", fontsize=8,
                ha="left", va="bottom")
        ax.set_xlabel("per-patient consistency $f$")
        ax.set_title(rf"$\theta = {theta:g}$")
        ax.set_ylim(-0.03, 1.08)
        ax.grid(True, alpha=0.25)

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

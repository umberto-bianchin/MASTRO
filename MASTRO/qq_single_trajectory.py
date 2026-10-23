"""
Q-Q plot of ONE trajectory over many per-patient null datasets, for the four
ablation variants (maestri, tree counts, per-tree null, pooled trees).

A null dataset relabels every patient by a uniform bijection of its own
alterations. For a two-node trajectory X->Y only the images of X and Y matter,
so a patient's relabelling reduces to a uniform ordered pair (u, v) of its
alterations, and the patient matches in every tree that holds u->-v. Hence:

  - the observed statistic of a null dataset is a sum, over the patients that
    carry X and Y, of a table lookup at one random pair per patient;
  - the null law each variant tests against is the same on every null dataset
    (a uniform relabelling composed with a fixed one is again uniform), so it is
    computed once, here exactly (no Monte-Carlo fallback).

The variants, as in ablation_toys.VARIANTS:
  maestri      statistic sum_i round(L c_i / M_i), null: one pair per patient
  count_trees  statistic sum_i c_i,                null: one pair per patient
  perm_tree    statistic of maestri, null: an independent pair per tree. Each
               tree weighs round(L / M_i) on the grid, as compute_significance_
               ensemble.py does (a tree of a patient with M_i > 2L weighs 0)
  trees_as_patients  statistic of count_trees, null: every tree is a patient
               relabelled over its own alterations
c_i is the number of trees of patient i that match, L = 1000 the grid of the
weighted support (--L_cap of compute_significance_ensemble.py).

A fifth series, maestri_rand, breaks the ties of the MAESTRI statistic at
random, P(S > s) + U P(S = s): exactly uniform under the null, it shows how
far below the diagonal the discreteness alone puts the test.

The y axis is linear up to 1.3 x the largest expected value (dotted line) and
logarithmic above it.

Outputs, in --out: <tag>.npz (p-values of every null dataset), <tag>.csv
(rejection rates), <tag>_qq.{pdf,png}.

    python3 qq_single_trajectory.py --graphs_all <graphs_all.txt> \\
        --owner <owner.txt> --pattern "TP53->-PIK3CA" --n_datasets 1000000
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_ablation_qq import STYLE
from utils import read_ints

SEP = re.compile(r"->-|-\?-|-/-")
VARIANTS = ["maestri", "count_trees", "perm_tree", "trees_as_patients"]
FLOOR = 1e-300


def load_patients(graphs_all, owner_path):
    """{patient: list of item lists}, owner ids made 0-based."""
    owner = read_ints(owner_path)
    shift = 1 if min(owner) == 1 else 0
    patients = defaultdict(list)
    with open(graphs_all) as f:
        for t, line in enumerate(f):
            patients[owner[t] - shift].append(line.split())
    return patients


def tree_labels(items):
    return {x for it in items for x in SEP.split(it)}


def patient_tables(trees, x, y):
    """Per-pair match counts and per-tree match probabilities of one patient.

    Returns (counts, q_patient, q_own): counts[k] is the number of trees that
    match when (x, y) is sent to the k-th ordered pair of the patient's
    alterations; q_patient[j] the probability that tree j matches under a
    uniform pair of the patient's alterations (per-tree null); q_own[j] the
    same over tree j's own alterations (pooled trees).
    """
    alts = sorted(set().union(*(tree_labels(t) for t in trees)))
    idx = {a: i for i, a in enumerate(alts)}
    n = len(alts)
    count = np.zeros((n, n), dtype=np.int64)
    q_patient, q_own = [], []
    for items in trees:
        anc = [it.split("->-") for it in items if "->-" in it]
        for u, v in anc:
            count[idx[u], idx[v]] += 1
        q_patient.append(len(anc) / (n * (n - 1)))
        own = tree_labels(items)
        q_own.append(len(anc) / (len(own) * (len(own) - 1))
                     if {x, y} <= own else 0.0)
    off = ~np.eye(n, dtype=bool)
    return count[off], count[idx[x], idx[y]], np.array(q_patient), np.array(q_own)


def law_of(values):
    """Law of a uniform draw from *values* (non-negative ints) as an array."""
    return np.bincount(values) / len(values)


def bernoulli_sum(q, step):
    """Law of step * (sum of independent Bernoulli(q_j)) as an array."""
    if step == 0:
        return np.array([1.0])
    pb = np.array([1.0])
    for qj in q:
        pb = np.concatenate([pb * (1 - qj), [0.0]]) + np.concatenate([[0.0], pb * qj])
    if step == 1:
        return pb
    out = np.zeros(step * (len(pb) - 1) + 1)
    out[::step] = pb
    return out


def convolve_all(laws):
    """Law of the sum of independent variables, each given as a sparse array."""
    total = np.array([1.0])
    for law in laws:
        nz = np.nonzero(law)[0]
        new = np.zeros(len(total) + len(law) - 1)
        for k in nz:
            new[k:k + len(total)] += law[k] * total
        total = new
    return total


def tail(F):
    """tail[s] = P(S >= s), with tail[len(F)] = 0."""
    return np.concatenate([np.cumsum(F[::-1])[::-1], [0.0]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs_all", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--pattern", required=True, help='Two-node trajectory, e.g. "TP53->-PIK3CA"')
    ap.add_argument("--n_datasets", type=int, default=1_000_000)
    ap.add_argument("--L", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ablation/single_trajectory")
    args = ap.parse_args()
    x, y = args.pattern.split("->-")
    L = args.L
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"qq_{x}_{y}"

    patients = load_patients(args.graphs_all, args.owner)
    carriers = [i for i, trees in patients.items()
                if {x, y} <= set().union(*(tree_labels(t) for t in trees))]
    print(f"[data] {len(patients)} patients, {len(carriers)} carry {x} and {y}, "
          f"{sum(len(patients[i]) for i in carriers)} trees")

    # per-patient tables and null laws
    w_vals, c_vals, real_w, real_c = [], [], 0, 0
    laws = {v: [] for v in VARIANTS}
    for i in carriers:
        trees = patients[i]
        M = len(trees)
        counts, c_real, q_pat, q_own = patient_tables(trees, x, y)
        w = np.rint(L * counts / M).astype(np.int64)
        w_vals.append(w)
        c_vals.append(counts)
        real_w += int(round(L * c_real / M))
        real_c += int(c_real)
        laws["maestri"].append(law_of(w))
        laws["count_trees"].append(law_of(counts))
        laws["perm_tree"].append(bernoulli_sum(q_pat, int(round(L / M))))
        laws["trees_as_patients"].append(bernoulli_sum(q_own, 1))
    tails = {v: tail(convolve_all(laws[v])) for v in VARIANTS}
    stat_of = {"maestri": "w", "perm_tree": "w", "count_trees": "c", "trees_as_patients": "c"}

    def pvalue(v, s):
        t = tails[v]
        return t[np.minimum(s, len(t) - 1)]

    real = {"w": real_w, "c": real_c}
    print("[real] " + "  ".join(f"{v}={pvalue(v, real[stat_of[v]]):.3e}" for v in VARIANTS))

    # null datasets: one uniform pair per carrier
    rng = np.random.default_rng(args.seed)
    n = args.n_datasets
    S = {"w": np.zeros(n, dtype=np.int64), "c": np.zeros(n, dtype=np.int64)}
    for w, c in zip(w_vals, c_vals):
        k = rng.integers(0, len(w), size=n)
        S["w"] += w[k]
        S["c"] += c[k]
    P = {v: pvalue(v, S[stat_of[v]]) for v in VARIANTS}
    # randomized MAESTRI p-value P(S > s) + U P(S = s): exactly uniform under
    # the null, it shows where the discrete one sits below the diagonal
    t, s = tails["maestri"], S["w"]
    lo = t[np.minimum(s + 1, len(t) - 1)]
    P["maestri_rand"] = lo + rng.random(n) * (pvalue("maestri", s) - lo)
    np.savez_compressed(out / f"{tag}.npz", **P)

    alphas = (0.05, 0.01, 1e-3, 1e-4, 1e-5)
    with (out / f"{tag}.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["variant", "n_datasets", "p_real"] + [f"P(p<={a})" for a in alphas]
                    + ["frac_p_le_1e-300"])
        for v in VARIANTS + ["maestri_rand"]:
            row = [v, n, f"{pvalue(v, real[stat_of[v]]):.3e}" if v in VARIANTS else ""]
            row += [f"{np.mean(P[v] <= a):.4g}" for a in alphas]
            row += [f"{np.mean(P[v] <= FLOOR):.4g}"]
            wr.writerow(row)
            print(f"[null] {v:18s} " + "  ".join(f"P(p<={a:g})={r}" for a, r in zip(alphas, row[3:])))

    # Q-Q plot: y linear up to the largest expected value, logarithmic above
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))
    top = expected[0]
    ymax = max(-np.log10(max(P[v].min(), FLOOR)) for v in VARIANTS)
    lin = top * 1.3
    ax.plot([0, top], [0, top], color="0.4", lw=0.8, ls="--")
    ax.set_yscale("symlog", linthresh=lin, linscale=3)
    ax.set_xlim(0, top * 1.03)
    ax.set_ylim(0, max(ymax, top) * 1.05)
    ticks = list(range(0, int(lin) + 1)) + [t for t in (10, 30, 100, 300) if lin < t <= ymax * 1.05]
    ax.set_yticks(ticks)
    labels = [str(t) for t in ticks]
    if ymax >= -np.log10(FLOOR) - 1e-6:
        labels[-1] = r"$\geq$" + labels[-1]   # p-values below FLOOR, exact zeros included
    ax.set_yticklabels(labels)
    ax.minorticks_off()
    if ymax > lin:
        ax.axhline(lin, color="0.6", lw=0.6, ls=":")
    for v in ["maestri_rand"] + VARIANTS:
        obs = -np.log10(np.maximum(np.sort(P[v]), FLOOR))
        # one marker per quarter pixel: the PDF stays light, nothing visible is lost
        disp = ax.transData.transform(np.column_stack([expected, obs]))
        _, keep = np.unique(np.floor(disp * 4), axis=0, return_index=True)
        name, col = STYLE.get(v, ("MAESTRI, ties randomized", "0.55"))
        ax.plot(expected[keep], obs[keep], ".", ms=2.5, color=col, label=name)
    ax.set_xlabel(r"expected $-\log_{10} p$", fontsize=8)
    ax.set_ylabel(r"observed $-\log_{10} p$", fontsize=8)
    ax.set_title(f"{x} $\\prec$ {y}, {n:,} null datasets", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, markerscale=3, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{tag}_qq.{ext}", dpi=200)
    print(f"[plot] {out / (tag + '_qq.pdf')}")


if __name__ == "__main__":
    main()

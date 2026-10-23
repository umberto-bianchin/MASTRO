"""
Ablation toys: break one design choice of Multi-MASTRO at a time.

Every toy is a cohort small enough for exact p-values. The variants differ
only in their inputs to compute_significance_ensemble.py, so each toy runs the
code used in the experiments:

  maestri            weights 1/M_i, owner = patient, null 'perm'
  perm_tree          weights 1/M_i, owner = patient, null 'perm_tree'
                     (one permutation per tree instead of one per patient)
  count_trees        weights 1,     owner = patient, null 'perm'
                     (support counts trees, the null still moves a patient's
                     trees together)
  count_perm_tree    weights 1,     owner = patient, null 'perm_tree'
  trees_as_patients  weights 1,     owner = tree,    null 'perm'
                     (Multi-MASTRO run on the pooled trees, as if every tree
                     were a patient; same expected-support law as
                     count_perm_tree)

The theta test is reported for the weighted variants only: with weight 1 per
tree pi_i counts trees and theta loses its meaning.

Toys (--toy), every tree on A and B only unless stated:
  shape       100 A->B trees spread over a carriers with 100/a trees each,
              100 patients in total, the others with A and B unordered. Does it matter whether 1 patient or 100
              patients hold the evidence?
  competition 1 patient with M trees A->B, 100 patients with one tree B->A.
  duplication every tree of a patient copied M times: identical evidence,
              the p-value must not move with M.
  ambiguity   every patient holds every order of its alterations (A->B and
              B->A, or the 6 chains of A, B, C): no information on the order.
  resolution  one tree per patient; 2 patients A->B, 1 patient B->A (trees
              on A and B only), m patients with A and B in the same node
              (AB->C or C->AB). Strict
              matching (Multi-MASTRO) against two rules that resolve the
              same-node pair: 'lenient' (same node supports both orders),
              'majority' (resolved towards the cohort's majority order), both
              recomputed on every null dataset, and 'observed' (the observed
              same-node trees are resolved towards the tested order once, and
              the null relabels the resolved trees). Exact computation, does
              not call the significance script.

Run from the MASTRO/ code directory:
    python3 ablation_toys.py --toy all --out results/ablation_toys
"""

import argparse
import csv
import itertools
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

VARIANTS = {
    # name: (weights, owner, null)
    "maestri": ("patient", "patient", "perm"),
    "perm_tree": ("patient", "patient", "perm_tree"),
    "count_trees": ("tree", "patient", "perm"),
    "count_perm_tree": ("tree", "patient", "perm_tree"),
    "trees_as_patients": ("tree", "tree", "perm"),
}


# =====================================================================
# Tree encoding
# =====================================================================
def tree_items(clones, parent):
    """Transaction items of a clone tree.

    clones[c] is the set of alterations of clone c, parent[c] its parent clone
    (-1 under the germline). Pairs in the same clone are '-?-', ancestor pairs
    '->-', every other pair '-/-'.
    """
    anc = []
    for c in range(len(clones)):
        a, p = set(), parent[c]
        while p != -1:
            a.add(p)
            p = parent[p]
        anc.append(a)
    where = {x: c for c, cl in enumerate(clones) for x in cl}
    items = []
    for x, y in itertools.combinations(sorted(where), 2):
        cx, cy = where[x], where[y]
        if cx == cy:
            items.append(f"{x}-?-{y}")
        elif cx in anc[cy]:
            items.append(f"{x}->-{y}")
        elif cy in anc[cx]:
            items.append(f"{y}->-{x}")
        else:
            items.append(f"{x}-/-{y}")
    return " ".join(items)


def pair_tree(top, bottom, others):
    """One ancestor pair top->bottom, every other alteration on its own branch."""
    clones = [{top}, {bottom}] + [{o} for o in others]
    return tree_items(clones, [-1, 0] + [-1] * len(others))


def chain_tree(order):
    return tree_items([{x} for x in order], [-1] + list(range(len(order) - 1)))


# Two alterations per tree. A random relabelling turns A->B into A->B or B->A
# with probability 1/2 each; NOISE holds A and B on different branches, which
# no relabelling orders, so it never supports either order
AB = pair_tree("A", "B", [])
BA = pair_tree("B", "A", [])
NOISE = tree_items([{"A"}, {"B"}], [-1, -1])


# =====================================================================
# Significance through compute_significance_ensemble.py
# =====================================================================
def run_variant(patients, patterns, variant, theta, workdir):
    """p-values of *patterns* on a cohort under one variant.

    patients: list of patients, each a list of transaction strings.
    Returns {pattern: row} with s_exp, s_theta, pval_exp, pval_theta.
    """
    wmode, omode, null = VARIANTS[variant]
    workdir.mkdir(parents=True, exist_ok=True)
    g, w, o = (workdir / n for n in ("graphs_all.txt", "weights.txt", "owner.txt"))
    lines = []
    with g.open("w") as fg, w.open("w") as fw, o.open("w") as fo:
        for i, trees in enumerate(patients):
            for t in trees:
                tid = len(lines)
                lines.append(set(t.split()))
                fg.write(t + "\n")
                fw.write(f"{1.0 / len(trees) if wmode == 'patient' else 1.0}\n")
                fo.write(f"{i if omode == 'patient' else tid}\n")
    pat = workdir / "patterns.txt"
    with pat.open("w") as f:
        for p in patterns:
            items = set(p.split())
            occ = [t for t, s in enumerate(lines) if items <= s]
            f.write(f"{p} ({len(occ)})\n{' '.join(map(str, occ))}\n")
    out = workdir / "pvalues.csv"
    test = "both" if wmode == "patient" else "exp"
    subprocess.run([sys.executable, str(SCRIPT_DIR / "compute_significance_ensemble.py"),
                    "-i", str(pat), "-o", str(out), "--graphs_all", str(g),
                    "-w", str(w), "--owner", str(o), "--null", null,
                    "--test", test, "--theta", repr(theta),
                    "--mc_cutoff", "1000000", "--L_cap", "100000"],
                   check=True, stdout=subprocess.DEVNULL)
    rows = {r["pattern"]: r for r in csv.DictReader(out.open())}
    return {p: rows[" ".join(sorted(p.split()))] for p in patterns}


def sweep(name, configs, patterns, theta_of, variants, tmp):
    """configs: list of (x, patients). One output row per x, variant, pattern."""
    out = []
    for x, patients in configs:
        for v in variants:
            theta = theta_of(x)
            res = run_variant(patients, patterns, v, theta, tmp / f"{name}_{x}_{v}")
            for p, r in res.items():
                out.append({
                    "toy": name, "x": x, "variant": v, "pattern": p,
                    "theta": theta if VARIANTS[v][0] == "patient" else "",
                    "s_exp": float(r["s_exp"]),
                    "s_theta": r["s_theta"] if VARIANTS[v][0] == "patient" else "",
                    "pval_exp": float(r["pval_exp"]),
                    "pval_theta": float(r["pval_theta"]) if r["pval_theta"] else "",
                })
                print(f"[{name}] x={x:<4} {v:18s} {p:22s} "
                      f"s_exp={float(r['s_exp']):8.3f} p_exp={float(r['pval_exp']):.3e}"
                      + (f"  p_theta={float(r['pval_theta']):.3e}" if r["pval_theta"] else ""),
                      flush=True)
    return out


# =====================================================================
# Toys through the significance script
# =====================================================================
def toy_shape(tmp, n_patients=100, n_trees=100):
    configs = []
    for a in [d for d in range(1, n_trees + 1) if n_trees % d == 0]:
        patients = [[AB] * (n_trees // a) for _ in range(a)]
        patients += [[NOISE] for _ in range(n_patients - a)]
        configs.append((a, patients))
    return sweep("shape", configs, ["A->-B"], lambda x: 1.0, list(VARIANTS), tmp)


def toy_competition(tmp, n_ba=100, M_list=(1, 10, 50, 99, 100, 200)):
    configs = [(M, [[AB] * M] + [[BA] for _ in range(n_ba)]) for M in M_list]
    return sweep("competition", configs, ["A->-B", "B->-A"], lambda x: 1.0,
                 list(VARIANTS), tmp)


def toy_duplication(tmp, n_carriers=10, n_patients=100, M_list=(1, 2, 3, 5, 10, 20, 50)):
    rows = []
    for theta, tag in ((1.0, "duplication_theta1"), (1e-6, "duplication_any")):
        configs = [(M, [[AB] * M for _ in range(n_carriers)]
                    + [[NOISE] * M for _ in range(n_patients - n_carriers)])
                   for M in M_list]
        variants = list(VARIANTS) if tag == "duplication_theta1" else ["maestri", "perm_tree"]
        rows += sweep(tag, configs, ["A->-B"], lambda x, t=theta: t, variants, tmp)
    return rows


def toy_ambiguity(tmp, n_list=(1, 2, 5, 10, 20, 50, 100)):
    two = [chain_tree("AB"), chain_tree("BA")]
    six = [chain_tree(p) for p in itertools.permutations("ABC")]
    rows = sweep("ambiguity_AB", [(n, [two] * n) for n in n_list],
                 ["A->-B"], lambda x: 0.5, list(VARIANTS), tmp)
    rows += sweep("ambiguity_ABC", [(n, [six] * n) for n in n_list],
                  ["A->-B A->-C B->-C"], lambda x: 1.0 / 6, list(VARIANTS), tmp)
    return rows


# =====================================================================
# Resolution toy (exact, one tree per patient)
# =====================================================================
# The ordered patients carry only A and B; the same-node patients carry
# {A, B, C}, C making the node pair part of a tree. Under the per-patient
# permutation (of the patient's own alterations) each tree puts the pair (A, B)
# in one of three states: 'f' A->B, 'r' B->A, 's' same node, or 'n' none of
# them.
T_AB = tree_items([{"A"}, {"B"}], [-1, 0])
T_BA = tree_items([{"B"}, {"A"}], [-1, 0])
T_ABC = tree_items([{"A", "B"}, {"C"}], [-1, 0])
T_CAB = tree_items([{"C"}, {"A", "B"}], [-1, 0])


def pair_state(items):
    s = set(items.split())
    if "A->-B" in s:
        return "f"
    if "B->-A" in s:
        return "r"
    if "A-?-B" in s:
        return "s"
    return "n"


def relabel(items, sigma):
    out = []
    for it in items.split():
        for sep in ("->-", "-?-", "-/-"):
            if sep in it:
                x, y = it.split(sep)
                x, y = sigma[x], sigma[y]
                if sep != "->-" and x > y:
                    x, y = y, x
                out.append(f"{x}{sep}{y}")
                break
    return " ".join(out)


def labels(items):
    """The alterations of a tree, sorted."""
    return sorted({x for it in items.split()
                   for x in re.split(r"->-|-\?-|-/-", it)})


def state_law(items):
    """Null law of the (A, B) state of a tree under a uniform relabelling of
    its own alterations."""
    labs = labels(items)
    perms = list(itertools.permutations(labs))
    c = Counter(pair_state(relabel(items, dict(zip(labs, perm)))) for perm in perms)
    return {k: v / len(perms) for k, v in c.items()}


def statistic(nf, nr, ns, rule, direction):
    """Support of A->B (direction 'f') or B->A ('r') under a resolution rule."""
    own, other = (nf, nr) if direction == "f" else (nr, nf)
    if rule == "strict":
        return own
    if rule == "lenient":
        return own + ns
    # majority: same-node pairs go to the strictly more frequent order
    return own + (ns if own > other else 0)


def resolve(items, direction):
    """The tree with its same-node pair A, B turned into the tested order."""
    edge = "A->-B" if direction == "f" else "B->-A"
    return " ".join(edge if it == "A-?-B" else it for it in items.split())


def exact_null(laws):
    """Joint law of (#f, #r, #s) over independent patients, as a dict."""
    dist = {(0, 0, 0): 1.0}
    for law in laws:
        new = Counter()
        for (f, r, s), p in dist.items():
            for st, q in law.items():
                key = (f + (st == "f"), r + (st == "r"), s + (st == "s"))
                new[key] += p * q
        dist = new
    return dist


def toy_resolution(m_list=(0, 2, 5, 10, 20, 30, 40, 60)):
    rows = []
    for m in m_list:
        trees = [T_AB] * 2 + [T_BA] + [T_ABC if j % 2 == 0 else T_CAB for j in range(m)]
        obs = Counter(pair_state(t) for t in trees)
        null = exact_null([state_law(t) for t in trees])
        # 'observed': the same-node trees count for the tested order, then the
        # null relabels those trees with the order already written in, so a
        # null tree supports the pattern only through a real edge
        null_res = {d: exact_null([state_law(resolve(t, d)) for t in trees])
                    for d in ("f", "r")}
        for rule in ("strict", "lenient", "majority", "observed"):
            for d, name in (("f", "A->-B"), ("r", "B->-A")):
                if rule == "observed":
                    s_obs = statistic(obs["f"], obs["r"], obs["s"], "lenient", d)
                    dist, null_rule = null_res[d], "strict"
                else:
                    s_obs = statistic(obs["f"], obs["r"], obs["s"], rule, d)
                    dist, null_rule = null, rule
                p = sum(pr for (f, r, s), pr in dist.items()
                        if statistic(f, r, s, null_rule, d) >= s_obs)
                rows.append({"toy": "resolution", "x": m, "variant": rule,
                             "pattern": name, "theta": "", "s_exp": s_obs,
                             "s_theta": "", "pval_exp": p, "pval_theta": ""})
                print(f"[resolution] m={m:<3} {rule:8s} {name}  support={s_obs:3d}  p={p:.3e}",
                      flush=True)
    return rows


def resolution_null(m_list=(0, 2, 5, 10, 20, 30, 40, 60), n_rep=10000,
                    alphas=(0.01, 0.05), seed=0):
    """Type I error of each rule for A->B on cohorts that are null by
    construction: every tree of the resolution toy relabelled at random.

    'strict', 'lenient' and 'majority' score the relabelled cohort against the
    null of their own statistic. 'observed' resolves the trees that the
    relabelled cohort holds in one node towards A->B and computes the null on
    those resolved trees, so its null depends on the data."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in m_list:
        trees = [T_AB] * 2 + [T_BA] + [T_ABC if j % 2 == 0 else T_CAB for j in range(m)]
        laws = [state_law(t) for t in trees]
        null = exact_null(laws)
        tail, q_res = {}, {}
        rejected = Counter()
        for _ in range(n_rep):
            relab = [relabel(t, dict(zip(labels(t), rng.permutation(labels(t)))))
                     for t in trees]
            st = Counter(pair_state(t) for t in relab)
            key = (st["f"], st["r"], st["s"])
            for rule in ("strict", "lenient", "majority"):
                if (rule, key) not in tail:
                    s_obs = statistic(*key, rule, "f")
                    tail[rule, key] = sum(pr for k, pr in null.items()
                                          if statistic(*k, rule, "f") >= s_obs)
                for a in alphas:
                    rejected[rule, a] += tail[rule, key] <= a
            # 'observed': strict count of A->B over the resolved trees
            dist = np.array([1.0])
            for t in relab:
                if t not in q_res:
                    q_res[t] = state_law(resolve(t, "f")).get("f", 0.0)
                dist = np.convolve(dist, [1 - q_res[t], q_res[t]])
            p = dist[statistic(*key, "lenient", "f"):].sum()
            for a in alphas:
                rejected["observed", a] += p <= a
        for rule in ("strict", "lenient", "majority", "observed"):
            for a in alphas:
                rate = rejected[rule, a] / n_rep
                rows.append({"m": m, "variant": rule, "alpha": a,
                             "rejection_rate": rate, "n_rep": n_rep})
            print(f"[resolution_null] m={m:<3} {rule:8s} "
                  + "  ".join(f"P(p<={a})={rejected[rule, a] / n_rep:.3f}"
                              for a in alphas), flush=True)
    return rows


TOYS = {
    "shape": toy_shape,
    "competition": toy_competition,
    "duplication": toy_duplication,
    "ambiguity": toy_ambiguity,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toy", default="all",
                    help="Comma-separated subset of: shape, competition, "
                         "duplication, ambiguity, resolution, or 'all'")
    ap.add_argument("--out", default="results/ablation_toys")
    args = ap.parse_args()

    names = (list(TOYS) + ["resolution"]) if args.toy == "all" else args.toy.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["toy", "x", "variant", "pattern", "theta", "s_exp", "s_theta",
              "pval_exp", "pval_theta"]
    with tempfile.TemporaryDirectory() as td:
        for name in names:
            rows = toy_resolution() if name == "resolution" else TOYS[name](Path(td))
            with (out / f"{name}.csv").open("w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=fields)
                wr.writeheader()
                wr.writerows(rows)
            print(f"[out] {out / (name + '.csv')}")
            if name == "resolution":
                rows = resolution_null()
                with (out / "resolution_null.csv").open("w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                    wr.writeheader()
                    wr.writerows(rows)
                print(f"[out] {out / 'resolution_null.csv'}")


if __name__ == "__main__":
    main()

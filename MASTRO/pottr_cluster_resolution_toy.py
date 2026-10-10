"""
Toy experiment: POTTR resolves unresolved pairs on the word of a few trees.

POTTR reads A-?-B as a cluster, i.e. A and B with unknown order, and its
conflict graph does not treat a cluster against a directed edge as a conflict
(compute_conflict_graph.py, conflict_row[0] & conflict_row[3]). One tree with
A->B is therefore enough to impose that order on every selected tree that holds
A and B in a cluster, and the significance test then scores those trees as if
they were ordered (MASTRO_significance_test/compute_significance.py adds the
trajectory edges to every supporting tree that lacks them).

Cohort, one tree per patient:
  * n_clusters patients with A and B in one cluster, alternating
        AB-C  (A-?-B A->-C B->-C)   and   C-AB  (A-?-B C->-A C->-B),
    so the cluster trees agree on {A,B} and disagree on where C goes;
  * n_ab patients with A->B and n_ba with B->A, over A and B only.

The cohort holds n_ab trees that order A before B, n_ba that order B before A,
and n_clusters that say nothing on the order. A trajectory A->B is supported by
n_ab patients. The script has two parts, and every POTTR run is followed by
the Multi-MASTRO test of the same trajectory under both nulls:

  sweep  the n_ab + n_ba ordered trees with m = 0, 2, ... cluster patients
         added, POTTR default at k = m + n_ab. m = 0 is the control: on the
         ordered trees alone A->B is not significant, and every added tree
         that says nothing on the order pushes POTTR's p-value down.
  grid   at n_clusters cluster patients, every k in --k_list with POTTR run
         three ways (default, -rf, -rt n_ab+1), recording the returned
         trajectory, whether POTTR reported "Trajectory introduces order!",
         its tree support and its own p-values.

Usage (MASTRO env; POTTR runs in its own env through --pottr_python):
    python3 pottr_cluster_resolution_toy.py \\
        --pottr_python ~/miniforge3/envs/pottr_env/bin/python \\
        --out results/pottr_toy
"""

import argparse
import csv
import subprocess
from pathlib import Path

from pottr_toy_experiment import (
    build_cohort,
    default_pottr_code,
    read_pottr,
    run_ensemble,
    run_pottr,
)

TREE_AB_C = "A-?-B A->-C B->-C"
TREE_C_AB = "A-?-B C->-A C->-B"
TREE_AB = "A->-B"
TREE_BA = "B->-A"


def cohort_trees(n_clusters, n_ab, n_ba):
    """Cluster patients first (alternating AB-C / C-AB), then A->B, then B->A."""
    clusters = [TREE_AB_C if i % 2 == 0 else TREE_C_AB
                for i in range(n_clusters)]
    return clusters + [TREE_AB] * n_ab + [TREE_BA] * n_ba


def run_one(args, root, dags, ens, k, extra, row, label):
    """POTTR at one (cohort, k, flags), then Multi-MASTRO on its trajectory."""
    out_k = root / "pottr" / f"k{k}"
    r = dict(row, k=k, pottr_flags=" ".join(extra) or "default")
    try:
        run_pottr(Path(args.pottr_python).expanduser(),
                  Path(args.pottr_code), dags, out_k, k, extra)
        r.update(read_pottr(out_k))
    except (subprocess.CalledProcessError, FileNotFoundError,
            StopIteration, AssertionError, KeyError, ValueError) as e:
        # An infeasible ILP or a trajectory POTTR does not score
        r["error"] = f"pottr: {type(e).__name__}"
        print(f"[{label}] POTTR failed ({r['error']})", flush=True)
        return r
    log = (out_k / "pottr.log").read_text()
    r["pottr_resolved_order"] = "Trajectory introduces order!" in log
    for null in ("perm", "indep"):
        try:
            e = run_ensemble(root, ens, null)
            r.update({f"mm_{key}_{null}": v for key, v in e.items()})
        except (subprocess.CalledProcessError, StopIteration) as err:
            # No resolved relation left once -?- is dropped
            r[f"mm_error_{null}"] = type(err).__name__
    print(f"[{label}] POTTR {r['traj']:<28} "
          f"resolved={r['pottr_resolved_order']!s:<5} "
          f"supp={r['supp_trees']} trees ({r['supp_patients']} pts) "
          f"p_perm={r['pottr_pval_perm']:.3e} || "
          f"MM s_exp={r.get('mm_s_exp_perm', float('nan')):g} "
          f"p_exp_perm={r.get('mm_pval_exp_perm', float('nan')):.3e}",
          flush=True)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_clusters", type=int, default=30,
                    help="Patients with A and B in one cluster (half AB-C, half C-AB)")
    ap.add_argument("--n_ab", type=int, default=2, help="Patients with A->B")
    ap.add_argument("--n_ba", type=int, default=1, help="Patients with B->A")
    ap.add_argument("--sweep", default="0,2,6,10,20,30",
                    help="Numbers of cluster patients to add to the n_ab + n_ba "
                         "ordered ones, POTTR default at k = n + n_ab. 0 is the "
                         "control: the ordered trees alone")
    ap.add_argument("--k_list", default=None,
                    help="POTTR k values for the grid at --n_clusters (default: "
                         "n_clusters/2, n_clusters, n_clusters+1, n_clusters+n_ab)")
    ap.add_argument("--pottr_python", required=True,
                    help="Python of the env with gurobipy/networkx/pandas/filelock")
    ap.add_argument("--pottr_code", default=str(default_pottr_code()))
    ap.add_argument("--out", default="results/pottr_toy")
    args = ap.parse_args()

    n = args.n_clusters
    out = Path(args.out) / f"clusters_{n}_{args.n_ab}_{args.n_ba}"
    rows = []

    # Part 1: the same A->B / B->A evidence, diluted in a growing number of trees
    # that say nothing on the order. k = m + n_ab forces POTTR to use all of
    # them, so at m = 0 it tests the ordered trees alone
    for m in [int(x) for x in args.sweep.split(",")]:
        trees = cohort_trees(m, args.n_ab, args.n_ba)
        root = out / "sweep" / f"n{m}"
        dags, ens = build_cohort(root / "cohort", trees, [1] * len(trees))
        row = dict(part="sweep", n_clusters=m, n_ab=args.n_ab, n_ba=args.n_ba)
        rows.append(run_one(args, root, dags, ens, m + args.n_ab, [], row,
                            f"sweep n={m:>3}"))

    # Part 2: at n cluster patients, every k with every resolution option
    if args.k_list:
        k_list = [int(x) for x in args.k_list.split(",")]
    else:
        k_list = [n // 2, n, n + 1, n + args.n_ab]
    # -rt: a cluster is resolved only by at least this many directed trees, one
    # more than the A->B trees, so it can never be met here
    configs = {"default": [], "rf": ["-rf"],
               f"rt{args.n_ab + 1}": ["-rt", str(args.n_ab + 1)]}
    trees = cohort_trees(n, args.n_ab, args.n_ba)
    dags, ens = build_cohort(out / "cohort", trees, [1] * len(trees))
    row = dict(part="grid", n_clusters=n, n_ab=args.n_ab, n_ba=args.n_ba)
    for k in k_list:
        for tag, extra in configs.items():
            rows.append(run_one(args, out / f"k{k}_{tag}", dags, ens, k, extra,
                                row, f"k={k:>3} {tag:>7}"))

    fields = []
    for r in rows:
        fields += [key for key in r if key not in fields]
    summary = out / "summary.csv"
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[toy] written -> {summary}")


if __name__ == "__main__":
    main()

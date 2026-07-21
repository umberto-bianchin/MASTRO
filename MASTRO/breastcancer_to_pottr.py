"""
Prepare a MATCHED POTTR + ensemble-MASTRO cohort from a multi-tree-per-patient
.npy dataset (e.g. breastCancer.npy), so the two methods can be compared on
exactly the same trees.

Why a dedicated prep (and why a naive breastcancer_dags run is wrong)
--------------------------------------------------------------------
POTTR reads a directory in which each file holds one transitively-closed DAG
(one tree). It groups files into evolutionary processes (patients) by the file
name ``<patient>-<tree>...`` and its ILP then selects at most one tree per
patient. Two things break a naive run on breastCancer:

1. **Scale.** POTTR builds a pairwise conflict graph for *every* pair of trees
   across patients, O(T^2) in the total number of trees T. breastCancer has
   ~2700 candidate trees, i.e. millions of pairs: the ILP never finishes.

2. **Support double-counts trees as patients.** ``compute_support.py`` tallies
   support by *graph name* (= per tree, ``P<i>-<j>``) and rescans *all*
   candidate trees, not just the one the ILP picked. A patient with 8 near-
   identical trees can therefore add up to 8 to a trajectory's "support". On a
   single-tree cohort (AML/NSCLC) name == patient so this is harmless; on a
   multi-tree cohort it inflates recurrence. This is exactly the artifact the
   ensemble test is meant to expose.

To make the comparison meaningful and tractable, this script:
  * deduplicates identical trees within a patient (POTTR would count them
    separately (see point 2),
  * caps the number of candidate trees per patient (``--max_trees``),
  * optionally restricts to multi-tree patients (``--multitree_only``), the
    only regime where POTTR and the ensemble test can differ, and/or to a
    random subset of ``--n_patients`` patients, to keep the ILP tractable,
  * writes the POTTR dags (``P<i>-<j>_bc.txt``) AND the matched ensemble inputs
    (``graphs_all.txt``, ``weights_uniform.txt``, ``owner.txt``,
    ``graphs_sampled.txt``) built from the *identical* selected trees, plus a
    ``manifest.csv`` recording the selection.

The ensemble inputs feed ``pottr_significance.py`` / the ensemble MASTRO tests;
because both sides describe the same trees, occurrence matching is faithful.

Usage:
    python3 breastcancer_to_pottr.py \\
        --npy ../data/breastCancer.npy \\
        --out ../../POTTR/data/breastcancer_dags \\
        --ensemble_out results/pottr_cmp/bc_inputs \\
        --max_trees 5 --multitree_only --n_patients 80 --seed 0
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from utils import (
    build_inputs,
    extract_edges_from_tree,
    tree_edges_to_transaction_items,
)


def _tree_key(tr, drop_gl):
    """Canonical hashable key for a tree = its set of directed edges."""
    edges = extract_edges_from_tree(tr)
    if drop_gl:
        edges = [(u, v) for u, v in edges if u != "GL" and v != "GL"]
    return frozenset(edges)


def _dedup_trees(trees, drop_gl):
    """Drop trees with an identical edge set, keeping first occurrence order."""
    seen = set()
    out = []
    for tr in trees:
        k = _tree_key(tr, drop_gl)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(tr)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--npy", required=True, help="Multi-tree .npy dataset")
    ap.add_argument("--out", required=True, help="Output dir of POTTR DAG files")
    ap.add_argument("--ensemble_out", default="",
                    help="If set, also write matched ensemble inputs "
                         "(graphs_all/weights/owner/sampled) here")
    ap.add_argument("--max_trees", type=int, default=5,
                    help="Cap on DISTINCT candidate trees per patient after "
                         "dedup (default 5). Total trees drive POTTR's O(T^2) cost")
    ap.add_argument("--min_trees", type=int, default=1,
                    help="Keep only patients with >= this many DISTINCT trees")
    ap.add_argument("--multitree_only", action="store_true",
                    help="Keep only patients with >=2 distinct trees (the only "
                         "regime where POTTR and the ensemble test can differ)")
    ap.add_argument("--n_patients", type=int, default=0,
                    help="Random subset of eligible patients (0 = all eligible)")
    ap.add_argument("--keep_gl", action="store_true",
                    help="Do not drop the germline root GL")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    drop_gl = not args.keep_gl
    data = np.load(args.npy, allow_pickle=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # purge stale dags from a previous selection: same-named files get
    # overwritten, but files for trees no longer selected would linger and
    # POTTR would read them as extra trees. Clear them first.
    for f in out.glob("*_bc.txt"):
        f.unlink()

    min_trees = max(args.min_trees, 2 if args.multitree_only else 1)

    # ---- select cohort ONCE: dedup -> cap -> eligibility -> subset ----------
    selected = []  # list of (orig_patient_idx, [trees])
    for i, tlist in enumerate(data):
        trees = _dedup_trees(list(tlist), drop_gl)
        if len(trees) < min_trees:
            continue
        if len(trees) > args.max_trees:
            trees = trees[:args.max_trees]  # deterministic: first distinct trees
        selected.append((i, trees))

    if args.n_patients and args.n_patients < len(selected):
        idx = sorted(rng.choice(len(selected), size=args.n_patients,
                                replace=False).tolist())
        selected = [selected[j] for j in idx]

    # ---- write POTTR dags: one file per tree, named P<i>-<j>_bc.txt ---------
    n_files = 0
    tree_counts = []
    for pi, trees in selected:
        tree_counts.append(len(trees))
        for j, tr in enumerate(trees):
            edges = extract_edges_from_tree(tr)
            items = tree_edges_to_transaction_items(edges, drop_gl=drop_gl)
            (out / f"P{pi}-{j}_bc.txt").write_text(" ".join(items) + "\n")
            n_files += 1

    # ---- write MATCHED ensemble inputs from the identical selected trees ----
    if args.ensemble_out:
        ens = Path(args.ensemble_out)
        ens.mkdir(parents=True, exist_ok=True)
        patients_trees = [trees for _, trees in selected]
        build_inputs(patients_trees, ens, seed=args.seed, drop_gl=drop_gl)
        with (ens / "manifest.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ensemble_owner_id", "orig_patient_idx", "n_trees"])
            for pos, (pi, trees) in enumerate(selected):
                w.writerow([pos, pi, len(trees)])

    tc = np.array(tree_counts) if tree_counts else np.array([0])
    print(f"[bc->POTTR] patients selected = {len(selected)}, tree files = {n_files}")
    print(f"[bc->POTTR] distinct trees/patient: min={tc.min()} "
          f"median={int(np.median(tc))} max={tc.max()} "
          f"(multi-tree patients = {(tc > 1).sum()})")
    print(f"[bc->POTTR] total trees T = {int(tc.sum())} "
          f"-> POTTR pairwise cost ~ {int(tc.sum())**2 // 2} conflict graphs")
    print(f"[bc->POTTR] POTTR dags: {out}")
    if args.ensemble_out:
        print(f"[bc->POTTR] matched ensemble inputs: {args.ensemble_out}")


if __name__ == "__main__":
    main()

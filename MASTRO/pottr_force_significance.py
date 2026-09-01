"""
Force POTTR's *own* permutation test on every POTTR trajectory it can score.

``run_POTTR.py`` only runs the MASTRO significance test when the maximum
trajectory of that k has fewer than 13 nodes::

    # run_POTTR.py
    if len(trajectories[0].nodes) < 13:
        compute_significance.run_stat_significancce_test(...)
    else:
        print('Trajectory size is too large to execute the significance test.')

The gate exists because ``compute_num_automorph`` enumerates *all* n!
permutations of the trajectory nodes, which is hopeless past ~12 nodes.  But
the gate is applied per *k*: it looks only at the largest trajectory of that
run, so a whole k directory is skipped, small trajectories included.  That is
why most POTTR trajectories in the k-sweep end up with no p-value of their own.

This script re-applies the gate where it belongs, per trajectory: it collects
the trajectories of every ``k<k>/converted_graphs.txt``, deduplicates them,
keeps those with at most ``--max_nodes`` non-root nodes, and runs POTTR's
significance test once on the union (loading the DAGs a single time).  The
result is a normal POTTR ``significance_output.txt`` that
``pottr_significance.py`` can read via ``--pottr_sig_global``.

Directories that already contain POTTR's own ``significance_output.txt`` are
still used: their trajectories are skipped here, so nothing is recomputed and
POTTR's original files are never overwritten.

Usage:
    python3 pottr_force_significance.py \\
        --pottr_dir results/pottr_cmp/pottr_bc --k_range 2,50 \\
        --dags ../POTTR/data/breastcancer_dags \\
        --pottr_repo ../POTTR --max_nodes 6 --cores 20
"""

import argparse
import os
import sys
from pathlib import Path

from utils import ensure_dir, read_result_pairs, parse_items_from_pattern_line

ROOT_LABEL = "0"
SEPS = ("->-", "-/-", "-?-")


def endpoints(item):
    for sep in SEPS:
        if sep in item:
            a, b = item.split(sep, 1)
            return a, sep, b
    return None


def trajectory_nodes(items):
    """Non-root nodes of the trajectory, exactly as POTTR's load_graph builds
    them (every endpoint of every item, '-?-' cluster pairs included)."""
    nodes = set()
    for it in items:
        ep = endpoints(it)
        if ep:
            nodes.add(ep[0])
            nodes.add(ep[2])
    nodes.discard(ROOT_LABEL)
    return nodes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pottr_dir", required=True, help="dir with k<k>/ subdirs")
    ap.add_argument("--k_range", default="2,50", help="min,max k (inclusive)")
    ap.add_argument("--dags", required=True, help="POTTR dags dir (same one POTTR ran on)")
    ap.add_argument("--pottr_repo", default="../POTTR", help="POTTR repo root (holds code/)")
    ap.add_argument("--max_nodes", type=int, default=6,
                    help="skip trajectories with more non-root nodes; cost blows "
                         "up fast in n (n! automorphisms, and C(tree_nodes, n) "
                         "candidate subsets per tree). 6 covers everything the "
                         "thesis table reports; raise it at your own risk")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--pottr_sig_name", default="significance_output.txt")
    ap.add_argument("--out", default="",
                    help="output file (default: <pottr_dir>/significance_forced.txt)")
    args = ap.parse_args()

    kmin, kmax = (int(x) for x in args.k_range.split(","))
    pottr_dir = Path(args.pottr_dir)
    out_path = Path(args.out) if args.out else pottr_dir / "significance_forced.txt"
    ensure_dir(out_path.parent)

    # --- collect the trajectories that still need a POTTR p-value ------------
    pairs = {}          # key -> (pattern_line, occ_line)
    n_seen = n_covered = n_too_big = 0
    for k in range(kmin, kmax + 1):
        kdir = pottr_dir / f"k{k}"
        cg = kdir / "converted_graphs.txt"
        if not cg.exists():
            continue
        already = (kdir / args.pottr_sig_name).exists()
        for pl, ol in read_result_pairs(cg):
            items = parse_items_from_pattern_line(pl)
            if not items:
                continue
            n_seen += 1
            key = " ".join(sorted(items))
            if key in pairs:
                continue
            if already:                      # POTTR scored this k itself
                n_covered += 1
                continue
            if len(trajectory_nodes(items)) > args.max_nodes:
                n_too_big += 1
                continue
            pairs[key] = (pl, ol)

    sizes = {}
    for pl, _ol in pairs.values():
        n = len(trajectory_nodes(parse_items_from_pattern_line(pl)))
        sizes[n] = sizes.get(n, 0) + 1
    print(f"[force] {n_seen} trajectory instances over k in [{kmin},{kmax}]: "
          f"{len(pairs)} to score, {n_covered} already scored by POTTR, "
          f"{n_too_big} above --max_nodes {args.max_nodes}", flush=True)
    if sizes:
        hist = ", ".join(f"{n} nodes: {c}" for n, c in sorted(sizes.items()))
        print(f"[force] to score, by size -- {hist}", flush=True)
    if not pairs:
        print("[force] nothing to do")
        return

    support_file = out_path.with_suffix(".input.txt")
    with open(support_file, "w") as f:
        for pl, ol in pairs.values():
            f.write(pl if pl.endswith("\n") else pl + "\n")
            f.write(ol if ol.endswith("\n") else ol + "\n")

    # --- run POTTR's own test once on the union ------------------------------
    code_dir = (Path(args.pottr_repo).resolve() / "code")
    if not (code_dir / "MASTRO_significance_test").is_dir():
        sys.exit(f"[err] no MASTRO_significance_test under {code_dir}")
    dags_abs = str(Path(args.dags).resolve())
    support_abs = str(support_file.resolve())
    out_abs = str(out_path.resolve())

    sys.path.insert(0, str(code_dir))
    cwd = os.getcwd()
    os.chdir(code_dir)                       # POTTR imports are CWD-relative
    try:
        from MASTRO_significance_test import compute_significance
        compute_significance.run_stat_significancce_test(
            support_file=support_abs, graph_file=dags_abs,
            output_file=out_abs, cores=args.cores)
    finally:
        os.chdir(cwd)

    print(f"[force] wrote {out_path}")


if __name__ == "__main__":
    main()

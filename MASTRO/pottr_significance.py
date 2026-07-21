"""
Re-evaluate POTTR trajectories under the ensemble-MASTRO significance framework.

POTTR reports large *incomplete* posets and assesses their significance with
MASTRO's original single-tree test, after having selected, via ILP, the maximum
recurrent trajectory (resolving mutation clusters and choosing favourable trees).
This script asks a sharper question: how much of each POTTR trajectory is still
significant once (i) all candidate trees of every patient are properly weighted
(the ensemble expected-support and theta-consensus tests) and (ii) the trajectory
is evaluated on its own recomputed occurrences?

For each POTTR trajectory we keep only its *resolved* order relations
(``a->-b`` and, if present, ``a-/-b``), dropping the germline root ``0`` and the
unresolved ``-?-`` pairs, which make no ordering claim. We then recompute the
trajectory's occurrences directly in the ensemble transaction database and run
``compute_significance_ensemble.py --allow-incomplete`` on the resulting
(incomplete) pattern, so only the relations POTTR actually commits to constrain
the null.

Output: one row per POTTR trajectory with its size, recomputed ensemble support,
expected-support and theta-consensus p-values, and, when a POTTR
``significance_output.txt`` is available, POTTR's own permutation p-value for a
side-by-side comparison.

Usage:
    python3 pottr_significance.py \\
        --pottr_dir results/pottr_tracerx --k_range 2,50 \\
        --graphs_all INPUTS/graphs_all.txt \\
        -w INPUTS/weights_uniform.txt --owner INPUTS/owner.txt \\
        --theta 1.0 --null perm --mc_samples 2000 \\
        --out results/pottr_cmp/pottr_significance.csv
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

from utils import (
    SCRIPT_DIR,
    ensure_dir,
    read_result_pairs,
    parse_items_from_pattern_line,
    load_weights_and_owner,
)

ROOT_LABEL = "0"          # POTTR germline / root node (dropped, like MASTRO's GL)
RESOLVED_SEPS = ("->-", "-/-")  # ordered / incomparable; '-?-' is unresolved -> dropped


def endpoints(item):
    """Return (a, sep, b) for an edge item, or None."""
    for sep in ("->-", "-/-", "-?-"):
        if sep in item:
            a, b = item.split(sep, 1)
            return a, sep, b
    return None


def resolved_relations(items):
    """Keep only the resolved (ordered/incomparable) relations among non-root
    nodes; drop '-?-' (unresolved) and anything touching the root '0'."""
    keep = []
    for it in items:
        ep = endpoints(it)
        if ep is None:
            continue
        a, sep, b = ep
        if sep == "-?-":
            continue
        if a == ROOT_LABEL or b == ROOT_LABEL:
            continue
        keep.append(it)
    return keep


def load_transaction_sets(graphs_all):
    """Each transaction (tree) -> frozenset of its edge-items."""
    sets = []
    for line in open(graphs_all):
        sets.append(frozenset(line.split()))
    return sets


def match_occurrences(pattern_items, tx_sets):
    """Indices of transactions that contain all of the pattern's items."""
    pat = frozenset(pattern_items)
    return [t for t, s in enumerate(tx_sets) if pat <= s]


def load_pottr_pvals(sig_path):
    """Map resolved-``->-`` edge-set -> POTTR permutation p-value, from a POTTR
    significance_output.txt (';'-separated; edges in the first column)."""
    out = {}
    if not sig_path.exists():
        return out
    with open(sig_path) as f:
        header = f.readline().rstrip("\n").split(";")
        try:
            i_edges = header.index("edges_traj")
            i_pperm = header.index("pval_perm")
        except ValueError:
            return out
        for line in f:
            cols = line.rstrip("\n").split(";")
            if len(cols) <= max(i_edges, i_pperm):
                continue
            edges = re.findall(r"[^\[\],\s]+->-[^\[\],\s]+", cols[i_edges])
            key = frozenset(e for e in edges if ROOT_LABEL not in endpoints(e)[:1] + endpoints(e)[2:])
            out[key] = cols[i_pperm]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pottr_dir", required=True,
                    help="Directory with per-k subdirs (kN/converted_graphs.txt)")
    ap.add_argument("--k_range", default="2,50", help="min,max k (inclusive)")
    ap.add_argument("--graphs_all", required=True)
    ap.add_argument("-w", "--weights", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--null", choices=["perm", "indep"], default="perm")
    ap.add_argument("--mc_cutoff", type=int, default=8)
    ap.add_argument("--mc_samples", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pottr_sig_name", default="significance_output.txt",
                    help="POTTR significance file inside each kN dir (optional)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kmin, kmax = (int(x) for x in args.k_range.split(","))
    pottr_dir = Path(args.pottr_dir)
    tx_sets = load_transaction_sets(args.graphs_all)
    _, owner0, _, _ = load_weights_and_owner(args.weights, args.owner)

    # --- gather POTTR trajectories across k, dedup by resolved-relation set ---
    # meta[pattern_key] = dict(n_nodes_full, n_edges_full, n_rel, support, occ, ks, pottr_pval)
    meta = {}
    for k in range(kmin, kmax + 1):
        cg = pottr_dir / f"k{k}" / "converted_graphs.txt"
        if not cg.exists():
            continue
        pottr_pvals = load_pottr_pvals(pottr_dir / f"k{k}" / args.pottr_sig_name)
        for pl, _ol in read_result_pairs(cg):
            items = parse_items_from_pattern_line(pl)
            if not items:
                continue
            full_nodes = set()
            for it in items:
                ep = endpoints(it)
                if ep:
                    full_nodes.add(ep[0]); full_nodes.add(ep[2])
            rel = resolved_relations(items)
            if not rel:
                continue
            key = " ".join(sorted(rel))
            if key not in meta:
                occ = match_occurrences(rel, tx_sets)
                fwd = frozenset(e for e in rel if "->-" in e)
                meta[key] = dict(
                    n_nodes_full=len(full_nodes),
                    n_edges_full=len(items),
                    n_rel=len(rel),
                    n_nodes_rel=len({n for it in rel for n in (endpoints(it)[0], endpoints(it)[2])}),
                    occ=occ,
                    n_patients=len({owner0[t] for t in occ if 0 <= t < len(owner0)}),
                    ks=set(),
                    pottr_pval=pottr_pvals.get(fwd, ""),
                )
            meta[key]["ks"].add(k)

    print(f"[POTTR] {len(meta)} distinct POTTR trajectories over k in "
          f"[{kmin},{kmax}]", flush=True)

    # --- write a MASTRO-format filtered file and run the ensemble sig test ---
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    filt = out_path.with_suffix(".pottr_patterns.txt")
    with filt.open("w") as f:
        for key, m in meta.items():
            f.write(f"{key} ({len(m['occ'])})\n")
            f.write(" ".join(str(t) for t in m["occ"]) + "\n")

    sig_csv = out_path.with_suffix(".ensemble_sig.csv")
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "compute_significance_ensemble.py"),
        "-i", str(filt), "-o", str(sig_csv),
        "-w", str(args.weights), "--owner", str(args.owner),
        "--graphs_all", str(args.graphs_all),
        "--test", "both", "--theta", str(args.theta), "--null", args.null,
        "--mc_cutoff", str(args.mc_cutoff), "--mc_samples", str(args.mc_samples),
        "--n_jobs", str(args.n_jobs), "--seed", str(args.seed),
        "--allow-incomplete",
    ], check=True)

    sig = {row["pattern"]: row for row in csv.DictReader(open(sig_csv))}

    # --- merge and write the final comparison table ---
    with out_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["k_values", "n_nodes", "n_resolved_rel", "n_edges_full",
                     "support_trees", "support_patients", "s_exp", "s_theta",
                     "mastro_pval_exp", "mastro_pval_theta", "pottr_pval_perm"])
        for key, m in sorted(meta.items(), key=lambda kv: -kv[1]["n_nodes_rel"]):
            s = sig.get(key, {})
            wr.writerow([
                ",".join(str(x) for x in sorted(m["ks"])),
                m["n_nodes_rel"], m["n_rel"], m["n_edges_full"],
                len(m["occ"]), m["n_patients"], s.get("s_exp", ""), s.get("s_theta", ""),
                s.get("pval_exp", ""), s.get("pval_theta", ""), m["pottr_pval"],
            ])
    print(f"[POTTR] written -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

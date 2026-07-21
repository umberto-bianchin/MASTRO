# =============================================================================
# Post-filter for Algorithm 2 (theta-frequent) and Algorithm 3 (theta-maximal).
#
# Context:
#   Each patient i has M_i phylogenetic trees. All trees are pooled into a single
#   transaction database where each transaction t gets weight w_t = 1/M_{owner(t)}
#   (uniform weighting) and an owner label owner(t) = i. Stage 1 (LCM with
#   weighted support) has already produced the expected-support frequent
#   candidates. This script applies the Stage 2 post-filter.
#
# Definitions:
#   pi_i(P)    = sum of w_t over the occurrences t of P that are owned by patient
#                i = the weighted "presence" of pattern P for patient i. When
#                w_t = 1/M_i this is the fraction of patient i's trees containing P.
#
#   s_theta(P) = number of patients i for which pi_i(P) >= theta, i.e. how many
#                patients' weighted presence of P reaches the threshold theta.
#
# Output (Algorithm 2, default):
#   theta-frequent candidates: patterns P with s_theta(P) >= sigma_theta.
#
# Output (Algorithm 3, --maximal flag):
#   theta-maximal candidates: theta-frequent patterns P for which no proper
#   superset Q has s_theta(Q) = s_theta(P).
# =============================================================================

import argparse
from pathlib import Path
from utils import (
    load_weights_and_owner,
    parse_occ_ids,
    parse_items_from_pattern_line,
    compute_theta_support,
    read_result_pairs,
)

parser = argparse.ArgumentParser(
    description="Algorithm 2/3 post-filter: theta-frequent (default) or theta-maximal (--maximal)"
)
parser.add_argument("-i",       help="input file (filtered results from Stage 1)")
parser.add_argument("-o",       help="output file")
parser.add_argument("-w",       help="weights file (one weight per transaction line)")
parser.add_argument("-owner",   help="owner file (one patient-id per transaction line)")
parser.add_argument("-theta",   type=float, default=1.0, help="theta threshold in (0,1]")
parser.add_argument("-st",      type=int,   default=2,   help="sigma_theta: minimum theta-support")
parser.add_argument("--maximal", action="store_true",
                    help="if set, run Algorithm 3: keep only theta-maximal patterns (Algorithm 2 + maximality pruning)")

args = parser.parse_args()

# --- Load weights and owner vectors (one entry per transaction) ---
weights, owner0, n, K = load_weights_and_owner(args.w, args.owner)

# =============================================================================
# Step 1: collect all theta-frequent candidates.
# =============================================================================
candidates = []

for pattern_line, occ_line in read_result_pairs(Path(args.i)):
    items = parse_items_from_pattern_line(pattern_line)
    if not items:
        continue

    occ_ids = parse_occ_ids(occ_line)
    s_theta, _ = compute_theta_support(occ_ids, weights, owner0, n, K, args.theta)

    if s_theta >= args.st:
        candidates.append({
            "items":        items,
            "pattern_line": pattern_line,
            "occ_line":     occ_line,
            "s_theta":      s_theta,
        })

# =============================================================================
# Step 2 (Algorithm 3 only): enforce theta-maximality within each s_theta bucket.
#
# A pattern P is discarded if a proper superset Q with the same s_theta value
# exists. Two patterns can only make one non-maximal if they share the same
# s_theta, so we group candidates into buckets by s_theta and prune inside each.
# Processing each bucket largest-first guarantees any superset is already kept
# by the time its subsets are examined.
# =============================================================================
if args.maximal:
    buckets = {}
    for c in candidates:
        buckets.setdefault(c["s_theta"], []).append(c)

    kept_entries = []
    for bucket in buckets.values():
        bucket_sorted = sorted(bucket, key=lambda x: len(x["items"]), reverse=True)
        kept = []
        for c in bucket_sorted:
            # c["items"] < Q is a proper-subset test on the item sets: keep c only
            # if it is not strictly contained in an already-kept larger pattern.
            if not any(c["items"] < Q for Q in kept):
                kept.append(c["items"])
                kept_entries.append(c)

    candidates = sorted(kept_entries,
                        key=lambda x: (x["s_theta"], len(x["items"])),
                        reverse=True)

# =============================================================================
# Write output
# =============================================================================
with open(args.o, "w") as fout:
    for c in candidates:
        fout.write(c["pattern_line"])
        fout.write(c["occ_line"])

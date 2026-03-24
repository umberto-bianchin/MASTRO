# =============================================================================
# Post-filter for Algorithm 2 (θ-frequent) and Algorithm 3 (θ-maximal)
#
# Context:
#   Each patient i has M_i phylogenetic trees.  All trees are pooled into a
#   single transaction database where each transaction t gets weight
#   w_t = 1/M_{owner(t)}  (uniform weighting) and an owner label owner(t) = i.
#   Stage 1 (LCM with weighted support) has already produced expected-support
#   frequent candidates.  This script applies the Stage 2 post-filter:
#
# Definitions:
#   π_i(P) = Σ_{t ∈ occ(P), owner(t)=i}  w_t
#          = weighted "presence" of pattern P for patient i.
#          When w_t = 1/M_i this equals the fraction of patient i's trees
#          that contain P.
#
#   s^(θ)(P) = |{ i : π_i(P) ≥ θ }|
#            = number of patients for whom P's weighted presence reaches
#              the threshold θ.
#
# Output (Algorithm 2, default):
#   θ-frequent candidates — patterns P satisfying  s^(θ)(P) ≥ σ_θ.
#
# Output (Algorithm 3, --maximal flag):
#   θ-maximal candidates — θ-frequent patterns P for which no proper
#   superset Q exists with s^(θ)(Q) = s^(θ)(P).
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
# Step 1: collect all θ-frequent candidates.
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
# Step 2 (Algorithm 3 only): enforce θ-maximality within each s_theta bucket.
#
# A pattern P is discarded if a proper superset Q exists in the same bucket
# (same s^(θ) value).  Sorting largest-first ensures supersets are seen first.
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

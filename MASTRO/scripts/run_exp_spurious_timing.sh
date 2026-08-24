#!/bin/bash
# =============================================================================
# EXPERIMENT: Spurious-itemset overhead + per-stage wall-clock timing (Ch. 6).
#
# Re-runs analyze_spurious_itemsets.py on breastCancer (weighted mode, same
# configuration as the thesis Section 6.2 figures) with the new per-stage
# timers: LCM enumeration vs. the real downstream pipeline stages
# (convert_results.py + filter_results.py, i.e. Steps 3-4 of
# run_MASTRO_weighted.py).
#
# Outputs (in results/spurious_analysis):
#   spurious_summary.csv   -> per-sigma counts + t_convert_s, t_filter_real_s,
#                             t_postprocess_s (columns used for the thesis text)
#   spurious_details.json  -> full breakdown
#   spurious_analysis.png  -> updated 2x2 figure; panel (c) now stacks LCM
#                             time vs. post-processing time
#   spurious_by_size.png   -> valid/spurious breakdown by |V(A)|
#
# Independent of the other experiment scripts, safe to run in parallel.
# NOTE: the thesis currently states these numbers were measured on a MacBook
# M2 Pro; if this run replaces them, update the hardware sentence in Sec. 6.2.
# =============================================================================
set -euo pipefail
# Locate the directory that holds analyze_spurious_itemsets.py, robustly to
# both layouts: scripts/ inside the package (cd ..) or scripts/ as a sibling
# of MASTRO/ (cd ../MASTRO, e.g. the server 'code/' bundle).
SDIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SDIR/../analyze_spurious_itemsets.py" ]; then
  cd "$SDIR/.."
elif [ -f "$SDIR/../MASTRO/analyze_spurious_itemsets.py" ]; then
  cd "$SDIR/../MASTRO"
else
  echo "ERROR: cannot locate analyze_spurious_itemsets.py from $SDIR" >&2
  exit 1
fi

# Same configuration as the thesis run (Sec. 6.2): all sigmas of the figure,
# 5e6 itemset cap, 180 s timeout per LCM invocation, weighted mode, seed 0.
SIGMAS=${SIGMAS:-2,3,4,5,6,8,10,15,20}
MAXITEMS=${MAXITEMS:-5000000}
TIMEOUT=${TIMEOUT:-180}

python3 analyze_spurious_itemsets.py \
  --npy ../data/breastCancer.npy \
  --sigmas "$SIGMAS" \
  --seed 0 \
  --mode weighted \
  --max_itemsets "$MAXITEMS" \
  --timeout "$TIMEOUT" \
  --outdir results/spurious_analysis

echo "=== spurious timing complete -> results/spurious_analysis ==="

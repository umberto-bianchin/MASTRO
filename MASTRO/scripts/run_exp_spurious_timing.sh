#!/bin/bash
# =============================================================================
# EXPERIMENT: spurious-itemset overhead + per-stage wall-clock timing.
#
# Runs analyze_spurious_itemsets.py on breastCancer in weighted mode with
# per-stage timers, separating LCM enumeration from the downstream pipeline
# stages (convert_results.py + filter_results.py). Most itemsets LCM emits are
# not valid trajectories - they are discarded downstream for not being
# relation-complete - and this measures what that costs.
#
# Outputs (in results/spurious_analysis):
#   spurious_summary.csv   -> per-sigma counts + t_convert_s, t_filter_real_s,
#                             t_postprocess_s
#   spurious_details.json  -> full breakdown
#   spurious_analysis.png  -> 2x2 figure; panel (c) compares LCM time against
#                             post-processing time
#   spurious_by_size.png   -> valid/spurious breakdown by |V(A)|
#
# Independent of the other experiment scripts, safe to run in parallel.
# NOTE: these timings are hardware-dependent; record the machine alongside
# any number taken from this script.
# =============================================================================
set -euo pipefail
# Every path below is relative to the code directory (<repo>/MASTRO), which
# holds this scripts/ folder; the datasets are in <repo>/data (../data).
cd "$(dirname "$0")/.."

# All sigmas of the figure, 5e6 itemset cap, 180 s timeout per LCM
# invocation, weighted mode, seed 0.
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

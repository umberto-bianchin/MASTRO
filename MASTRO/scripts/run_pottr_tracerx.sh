#!/bin/bash
# =============================================================================
# EXPERIMENT: POTTR vs Multi-MASTRO on TRACERx, the counterpart of
# run_pottr_breastcancer.sh (Tab. pottr of the Supplementary Material).
#
# TRACERx needs no cap: POTTR runs on all 469 trees. Both methods see the same
# trees: the POTTR dags (one file <patient>-<j>_tracerx.txt per tree) and the
# ensemble inputs are both written from data/TRACERx/graphs_tracerx.txt and
# owner_tracerx.txt, the tracked merge of data_tracerx.
# '-?-' means "same clone" in this cohort, so POTTR's cluster pairs are kept
# in the re-scoring (KEEP_CLUSTER=1), and the three patients above the MC
# cutoff are exact whenever that is cheaper than sampling.
#
# POTTR follows the protocol of its paper (Sect. 4.1): a solution pool of 50000
# and every k from 2 on; the paper reports one-alteration trajectories only
# from k = 36 on in this cohort, so k = 2..35 covers every ordered trajectory.
#
# Needs Gurobi (turing). Run from the repo's MASTRO/ directory:
#   bash scripts/run_pottr_tracerx.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."                     # -> <repo>/MASTRO, the code directory

GRAPHS=${GRAPHS:-../data/TRACERx/graphs_tracerx.txt}
OWNER=${OWNER:-../data/TRACERx/owner_tracerx.txt}
CDIR=${CDIR:-results/pottr_cmp/tracerx_cohort}
SIGMA=${SIGMA:-2}

# ---- matched cohort ---------------------------------------------------------
mkdir -p "$CDIR/dags"
if [ -z "$(ls -A "$CDIR/dags")" ]; then
  awk -v d="$CDIR/dags" 'NR==FNR { own[FNR] = $0; next }
    { p = own[FNR]; f = d "/" p "-" (n[p]++) "_tracerx.txt"; print > f; close(f) }' \
    "$OWNER" "$GRAPHS"
fi
N_DAGS=$(ls "$CDIR/dags" | wc -l | tr -d ' ')
[ "$N_DAGS" = 469 ] || { echo "[err] $N_DAGS dags in $CDIR/dags, expected 469" >&2; exit 1; }

if [ ! -f "$CDIR/inputs/graphs_all.txt" ]; then
  python3 run_pipeline.py --graphs "$GRAPHS" --owner "$OWNER" --sigma "$SIGMA" --theta_list 1.0 \
    --seed 0 --outdir "$CDIR"
fi

# ---- POTTR k-sweep, POTTR p-values, Multi-MASTRO re-scoring ------------------
SKIP_PREP=1 KEEP_CLUSTER=1 EXACT_WHEN_CHEAPER=1 \
POOL=${POOL:-50000} K_LIST=${K_LIST:-"$(seq 2 35 | tr '\n' ' ')"} \
MC_SAMPLES=${MC_SAMPLES:-20000} \
DAGS_DIR="$CDIR/dags" ENS_DIR="$CDIR/inputs" \
POTTR_OUT=${POTTR_OUT:-results/pottr_cmp/pottr_tracerx} \
SIG_OUT=${SIG_OUT:-results/pottr_cmp/pottr_tracerx_significance.csv} \
PLOT_OUT=${PLOT_OUT:-results/pottr_cmp/pottr_tracerx_comparison.pdf} \
  bash scripts/run_pottr_breastcancer.sh

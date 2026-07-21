#!/bin/bash
# =============================================================================
# EXPERIMENT: Calibration: empirical FWER control on breastCancer.
#
# Validates that the Westfall-Young thresholds actually control the FWER at the
# nominal alpha for both ensemble tests.
#   1. run_pipeline.py (no --significance) just to materialise the input files.
#   2. empirical_fwer_ensemble.py: null-dataset pool + bootstrap of
#      calibration/validation splits -> mean +/- std empirical FWER per alpha.
#
# Independent of the other experiment scripts, safe to run in parallel.
# Heavier than discovery (N minings under the null), give it its own cores.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PAR=${PAR:-4}
SIGMA_LIST=${SIGMA_LIST:-"2"}
THETA_LIST=${THETA_LIST:-"0.5 1.0"}
NULL=${NULL:-perm}
N_DATASETS=${N_DATASETS:-500}
M_CAL=${M_CAL:-250}
N_TRIALS=${N_TRIALS:-500}
ALPHAS=${ALPHAS:-0.01,0.05,0.1}
MC_SAMPLES=${MC_SAMPLES:-3000}   # per-patient MC draws; dominant cost knob
MC_CUTOFF=${MC_CUTOFF:-8}        # M_i above which a patient uses MC
NPY=../data/breastCancer.npy
THETA_CSV=$(echo "$THETA_LIST" | tr ' ' ',')

[ -f "$NPY" ] || { echo "ERROR: $NPY not found" >&2; exit 1; }
[ -x lcm53/lcm ] || ( echo "building LCM"; cd lcm53 && make )

for SIGMA in $SIGMA_LIST; do
  INP=results/calibration/breastCancer_inputs_sigma${SIGMA}
  echo "=== [1] build inputs: sigma=$SIGMA ==="
  python3 run_pipeline.py --npy "$NPY" --sigma "$SIGMA" --seed 0 \
    --theta_list "$THETA_CSV" --outdir "$INP"

  for THETA in $THETA_LIST; do
    echo "=== [2] empirical FWER: sigma=$SIGMA theta=$THETA null=$NULL ==="
    python3 empirical_fwer_ensemble.py \
      --graphs_all "$INP/inputs/graphs_all.txt" \
      -w           "$INP/inputs/weights_uniform.txt" \
      --owner      "$INP/inputs/owner.txt" \
      --sigma "$SIGMA" --theta "$THETA" --null "$NULL" --test both \
      --n_datasets "$N_DATASETS" --m "$M_CAL" --n_trials "$N_TRIALS" \
      --alphas "$ALPHAS" \
      --mc_cutoff "$MC_CUTOFF" --mc_samples "$MC_SAMPLES" --par "$PAR" \
      --outdir results/calibration/breastCancer_sigma${SIGMA}_theta${THETA}
  done
done
echo "=== calibration (breastCancer) complete -> results/calibration ==="

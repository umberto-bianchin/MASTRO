#!/bin/bash
# =============================================================================
# EXPERIMENT: Calibration: empirical FWER control on TRACERx.
# Same as the breastCancer calibration script, using the TRACERx transaction
# file (--graphs / --owner mode) to build the input files.
#
# Independent of the other experiment scripts, safe to run in parallel.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PAR=${PAR:-4}
SIGMA_LIST=${SIGMA_LIST:-"2"}
THETA_LIST=${THETA_LIST:-"0.5 1.0"}
NULL=${NULL:-perm}
N_DATASETS=${N_DATASETS:-1000}
M_CAL=${M_CAL:-500}
N_TRIALS=${N_TRIALS:-500}
ALPHAS=${ALPHAS:-0.01,0.05,0.1}
MC_SAMPLES=${MC_SAMPLES:-3000}   # per-patient MC draws; dominant cost knob
MC_CUTOFF=${MC_CUTOFF:-8}        # M_i above which a patient uses MC
GRAPHS=../data/TRACERx/graphs_tracerx.txt
OWNER=../data/TRACERx/owner_tracerx.txt
THETA_CSV=$(echo "$THETA_LIST" | tr ' ' ',')

[ -f "$GRAPHS" ] || { echo "ERROR: $GRAPHS not found" >&2; exit 1; }
[ -x lcm53/lcm ] || ( echo "building LCM"; cd lcm53 && make )

for SIGMA in $SIGMA_LIST; do
  INP=results/calibration/tracerx_inputs_sigma${SIGMA}
  echo "=== [1] build inputs: sigma=$SIGMA ==="
  python3 run_pipeline.py --graphs "$GRAPHS" --owner "$OWNER" --sigma "$SIGMA" \
    --seed 0 --theta_list "$THETA_CSV" --outdir "$INP"

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
      --outdir results/calibration/tracerx_sigma${SIGMA}_theta${THETA}
  done
done
echo "=== calibration (TRACERx) complete -> results/calibration ==="

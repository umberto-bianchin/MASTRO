#!/bin/bash
# =============================================================================
# EXPERIMENT: Calibration: empirical FWER control on TRACERx.
# Same as the breastCancer calibration script, using the TRACERx transaction
# file (--graphs / --owner mode) to build the input files.
#
# Independent of the other experiment scripts, safe to run in parallel.
# =============================================================================
set -euo pipefail
# Every path below is relative to the code directory (<repo>/MASTRO), which
# holds this scripts/ folder; the datasets are in <repo>/data (../data).
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
OUT_ROOT=${OUT_ROOT:-results/calibration}
EXACT_WHEN_CHEAPER=${EXACT_WHEN_CHEAPER:-1}  # exact above the cutoff when placements <= MC_SAMPLES; 0 = always MC (pre-Oct-2026 runs)
EXACT_ARG=()
[ "$EXACT_WHEN_CHEAPER" = 1 ] && EXACT_ARG=(--exact_when_cheaper)
WORKDIR=${WORKDIR:-}             # scratch for null-dataset mining; empty = <outdir>/resamples
WORKDIR_ARG=()
[ -n "$WORKDIR" ] && WORKDIR_ARG=(--workdir "$WORKDIR")
GRAPHS=../data/TRACERx/graphs_tracerx.txt
OWNER=../data/TRACERx/owner_tracerx.txt
THETA_CSV=$(echo "$THETA_LIST" | tr ' ' ',')

[ -f "$GRAPHS" ] || { echo "ERROR: $GRAPHS not found" >&2; exit 1; }
[ -x lcm53/lcm ] || ( echo "building LCM"; cd lcm53 && make )

for SIGMA in $SIGMA_LIST; do
  INP=$OUT_ROOT/tracerx_inputs_sigma${SIGMA}
  echo "=== [1] build inputs: sigma=$SIGMA ==="
  # Only the input files are used below: --min_mine_sigma skips the theta
  # candidate mining at floor(theta*sigma) (sigma=1 for theta=0.5), which blows up
  python3 run_pipeline.py --graphs "$GRAPHS" --owner "$OWNER" --sigma "$SIGMA" \
    --seed 0 --theta_list "$THETA_CSV" --min_mine_sigma "$SIGMA" --outdir "$INP"

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
      ${EXACT_ARG[@]+"${EXACT_ARG[@]}"} \
      ${WORKDIR_ARG[@]+"${WORKDIR_ARG[@]}"} \
      --outdir $OUT_ROOT/tracerx_sigma${SIGMA}_theta${THETA}
  done
done
echo "=== calibration (TRACERx) complete -> $OUT_ROOT ==="

#!/bin/bash
# =============================================================================
# EXPERIMENT: Discovery on TRACERx (real-data significance).
# Same corrected pipeline as the breastCancer discovery script, but the input
# is a pre-computed transaction file + owner file (--graphs / --owner mode).
#
# Independent of the other experiment scripts, safe to run in parallel.
# =============================================================================
set -euo pipefail
# Locate the directory that holds run_pipeline.py, robustly to both layouts:
# scripts/ inside the package (cd ..) or scripts/ as a sibling of MASTRO/
# (cd ../MASTRO, e.g. the server 'code/' bundle).
SDIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SDIR/../run_pipeline.py" ]; then
  cd "$SDIR/.."
elif [ -f "$SDIR/../MASTRO/run_pipeline.py" ]; then
  cd "$SDIR/../MASTRO"
else
  echo "ERROR: cannot locate run_pipeline.py from $SDIR" >&2
  exit 1
fi

M=${M:-2000}
PAR=${PAR:-4}
SIGMA_LIST=${SIGMA_LIST:-"2 5"}
THETA_LIST=${THETA_LIST:-"0.5 1.0"}
NULL=${NULL:-perm}
MC_SAMPLES=${MC_SAMPLES:-3000}   # per-patient MC draws; dominant cost knob
MC_CUTOFF=${MC_CUTOFF:-8}        # M_i above which a patient uses MC
# Floor on the mining threshold of the theta-consensus family. The correct
# threshold is floor(theta*sigma); when that drops below this value the family
# is mined at sigma instead and run_pipeline.py warns it may be incomplete.
# Empty = no floor, i.e. mine at floor(theta*sigma) however low. Mining at
# support 1 is intractable on TRACERx (one transaction carries 1006 items), so
# set this to match the breastCancer script, which passes --min_mine_sigma 2.
MIN_MINE_SIGMA=${MIN_MINE_SIGMA:-}
OUT_ROOT=${OUT_ROOT:-results/discovery}
PLOT_ROOT=${PLOT_ROOT:-results/fdr_plots}
GRAPHS=../data/TRACERx/graphs_tracerx.txt
OWNER=../data/TRACERx/owner_tracerx.txt
THETA_CSV=$(echo "$THETA_LIST" | tr ' ' ',')
MIN_MINE_ARG=()
[ -n "$MIN_MINE_SIGMA" ] && MIN_MINE_ARG=(--min_mine_sigma "$MIN_MINE_SIGMA")

[ -f "$GRAPHS" ] || { echo "ERROR: $GRAPHS not found" >&2; exit 1; }
[ -x lcm53/lcm ] || ( echo "building LCM"; cd lcm53 && make )
mkdir -p "$PLOT_ROOT" "$OUT_ROOT"

for SIGMA in $SIGMA_LIST; do
  OUT="$OUT_ROOT/tracerx_sigma${SIGMA}"

  echo "=== [1] observed mining + significance: sigma=$SIGMA ==="
  python3 run_pipeline.py \
    --graphs "$GRAPHS" --owner "$OWNER" --sigma "$SIGMA" --seed 0 \
    --theta_list "$THETA_CSV" \
    ${MIN_MINE_ARG[@]+"${MIN_MINE_ARG[@]}"} \
    --significance --sig_null "$NULL" \
    --sig_mc_cutoff "$MC_CUTOFF" --sig_mc_samples "$MC_SAMPLES" --sig_n_jobs "$PAR" \
    --outdir "$OUT"

  SIGDIR="$OUT/significance"
  PVAL_EXP="$SIGDIR/alg1_expected_uniform_pvalues_exp.csv"

  for THETA in $THETA_LIST; do
    PVAL_THETA="$SIGDIR/alg3_theta${THETA}_pvalues_theta.csv"
    WYOUT="$OUT_ROOT/tracerx_wy_sigma${SIGMA}_theta${THETA}"

    echo "=== [2] WY/FDR: sigma=$SIGMA theta=$THETA M=$M par=$PAR null=$NULL ==="
    python3 run_wy_correction_ensemble.py \
      --graphs_all "$OUT/inputs/graphs_all.txt" \
      -w           "$OUT/inputs/weights_uniform.txt" \
      --owner      "$OUT/inputs/owner.txt" \
      --sigma      "$SIGMA" -M "$M" \
      --test both --null "$NULL" --theta "$THETA" --par "$PAR" \
      ${MIN_MINE_ARG[@]+"${MIN_MINE_ARG[@]}"} \
      --mc_cutoff "$MC_CUTOFF" --mc_samples "$MC_SAMPLES" \
      --outdir "$WYOUT" \
      --pvalues_exp   "$PVAL_EXP" \
      --pvalues_theta "$PVAL_THETA" \
      --save_resample_pvals

    echo "=== [3] plot FDR: sigma=$SIGMA theta=$THETA ==="
    python3 plot_fdr_v2.py \
      --fdr_exp   "$WYOUT/fdr_exp.csv" \
      --fdr_theta "$WYOUT/fdr_theta.csv" \
      --out "$PLOT_ROOT/tracerx_sigma${SIGMA}_theta${THETA}.pdf" \
      --title "TRACERx sigma=${SIGMA} theta=${THETA}"
  done
done
echo "=== discovery (TRACERx) complete -> $OUT_ROOT ==="

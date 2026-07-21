#!/bin/bash
# =============================================================================
# EXPERIMENT: Discovery on TRACERx (real-data significance).
# Same corrected pipeline as the breastCancer discovery script, but the input
# is a pre-computed transaction file + owner file (--graphs / --owner mode).
#
# Independent of the other experiment scripts, safe to run in parallel.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

M=${M:-2000}
PAR=${PAR:-4}
SIGMA_LIST=${SIGMA_LIST:-"2 5"}
THETA_LIST=${THETA_LIST:-"0.5 1.0"}
NULL=${NULL:-perm}
MC_SAMPLES=${MC_SAMPLES:-3000}   # per-patient MC draws; dominant cost knob
MC_CUTOFF=${MC_CUTOFF:-8}        # M_i above which a patient uses MC
GRAPHS=../data/TRACERx/graphs_tracerx.txt
OWNER=../data/TRACERx/owner_tracerx.txt
THETA_CSV=$(echo "$THETA_LIST" | tr ' ' ',')

[ -f "$GRAPHS" ] || { echo "ERROR: $GRAPHS not found" >&2; exit 1; }
[ -x lcm53/lcm ] || ( echo "building LCM"; cd lcm53 && make )
mkdir -p results/fdr_plots

for SIGMA in $SIGMA_LIST; do
  OUT=results/discovery/tracerx_sigma${SIGMA}

  echo "=== [1] observed mining + significance: sigma=$SIGMA ==="
  python3 run_pipeline.py \
    --graphs "$GRAPHS" --owner "$OWNER" --sigma "$SIGMA" --seed 0 \
    --theta_list "$THETA_CSV" \
    --significance --sig_null "$NULL" \
    --sig_mc_cutoff "$MC_CUTOFF" --sig_mc_samples "$MC_SAMPLES" --sig_n_jobs "$PAR" \
    --outdir "$OUT"

  SIGDIR="$OUT/significance"
  PVAL_EXP="$SIGDIR/alg1_expected_uniform_pvalues_exp.csv"

  for THETA in $THETA_LIST; do
    PVAL_THETA="$SIGDIR/alg3_theta${THETA}_pvalues_theta.csv"
    WYOUT=results/discovery/tracerx_wy_sigma${SIGMA}_theta${THETA}

    echo "=== [2] WY/FDR: sigma=$SIGMA theta=$THETA M=$M par=$PAR null=$NULL ==="
    python3 run_wy_correction_ensemble.py \
      --graphs_all "$OUT/inputs/graphs_all.txt" \
      -w           "$OUT/inputs/weights_uniform.txt" \
      --owner      "$OUT/inputs/owner.txt" \
      --sigma      "$SIGMA" -M "$M" \
      --test both --null "$NULL" --theta "$THETA" --par "$PAR" \
      --mc_cutoff "$MC_CUTOFF" --mc_samples "$MC_SAMPLES" \
      --outdir "$WYOUT" \
      --pvalues_exp   "$PVAL_EXP" \
      --pvalues_theta "$PVAL_THETA" \
      --save_resample_pvals

    echo "=== [3] plot FDR: sigma=$SIGMA theta=$THETA ==="
    python3 plot_fdr_v2.py \
      --fdr_exp   "$WYOUT/fdr_exp.csv" \
      --fdr_theta "$WYOUT/fdr_theta.csv" \
      --out "results/fdr_plots/tracerx_sigma${SIGMA}_theta${THETA}.pdf" \
      --title "TRACERx sigma=${SIGMA} theta=${THETA}"
  done
done
echo "=== discovery (TRACERx) complete -> results/discovery ==="

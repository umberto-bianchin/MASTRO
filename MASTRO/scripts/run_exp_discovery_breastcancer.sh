#!/bin/bash
# =============================================================================
# EXPERIMENT: Discovery on breastCancer (real-data significance).
#
#   1. run_pipeline.py  -> mine Alg 0/1/2/3 + observed significance
#                          (expected-support on Alg 1, theta-consensus on Alg 3)
#   2. run_wy_correction_ensemble.py -> WY-corrected thresholds + empirical FDR,
#      each test scored on ITS OWN family (no merged CSV):
#         exp   FDR  <- observed Alg 1  vs  null expected-maximal family
#         theta FDR  <- observed Alg 3  vs  null theta-maximal family
#   3. plot_fdr_v2.py   -> FDR(k) curves per (sigma, theta)
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
NPY=${NPY:-../data/breastCancer.npy}
THETA_CSV=$(echo "$THETA_LIST" | tr ' ' ',')

[ -f "$NPY" ] || { echo "ERROR: $NPY not found" >&2; exit 1; }
[ -x lcm53/lcm ] || ( echo "building LCM"; cd lcm53 && make )
mkdir -p results/fdr_plots

for SIGMA in $SIGMA_LIST; do
  OUT=results/discovery/breastCancer_sigma${SIGMA}

  echo "=== [1] observed mining + significance: sigma=$SIGMA ==="
  python3 run_pipeline.py \
    --npy "$NPY" --sigma "$SIGMA" --seed 0 \
    --theta_list "$THETA_CSV" \
    --min_mine_sigma 2 \
    --significance --sig_null "$NULL" \
    --sig_mc_cutoff "$MC_CUTOFF" --sig_mc_samples "$MC_SAMPLES" --sig_n_jobs "$PAR" \
    --outdir "$OUT"

  SIGDIR="$OUT/significance"
  PVAL_EXP="$SIGDIR/alg1_expected_uniform_pvalues_exp.csv"

  for THETA in $THETA_LIST; do
    PVAL_THETA="$SIGDIR/alg3_theta${THETA}_pvalues_theta.csv"
    WYOUT=results/discovery/breastCancer_wy_sigma${SIGMA}_theta${THETA}

    echo "=== [2] WY/FDR: sigma=$SIGMA theta=$THETA M=$M par=$PAR null=$NULL ==="
    python3 run_wy_correction_ensemble.py \
      --graphs_all "$OUT/inputs/graphs_all.txt" \
      -w           "$OUT/inputs/weights_uniform.txt" \
      --owner      "$OUT/inputs/owner.txt" \
      --sigma      "$SIGMA" -M "$M" \
      --test both --null "$NULL" --theta "$THETA" --par "$PAR" \
      --min_mine_sigma 2 \
      --mc_cutoff "$MC_CUTOFF" --mc_samples "$MC_SAMPLES" \
      --outdir "$WYOUT" \
      --pvalues_exp   "$PVAL_EXP" \
      --pvalues_theta "$PVAL_THETA" \
      --save_resample_pvals

    echo "=== [3] plot FDR: sigma=$SIGMA theta=$THETA ==="
    python3 plot_fdr_v2.py \
      --fdr_exp   "$WYOUT/fdr_exp.csv" \
      --fdr_theta "$WYOUT/fdr_theta.csv" \
      --out "results/fdr_plots/breastCancer_sigma${SIGMA}_theta${THETA}.pdf" \
      --title "breastCancer sigma=${SIGMA} theta=${THETA}"
  done
done
echo "=== discovery (breastCancer) complete -> results/discovery ==="

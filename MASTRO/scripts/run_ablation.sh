#!/bin/bash
# =============================================================================
# EXPERIMENT: ablations of Multi-MASTRO on real cohorts (R1 calibration + Q-Q,
# R2 discoveries). One design choice broken at a time, see
# ablation_calibration.py for the variants and the protocol:
#   maestri, perm_tree, count_trees, count_perm_tree, trees_as_patients.
#
# Per dataset and sigma: the real cohort, n_cal per-patient null datasets
# (WY of the 'perm' variants), n_cal per-tree null datasets (WY of the per-tree
# variants), n_val per-patient null datasets (FWER, Q-Q). Every dataset is
# mined once (Multi-MASTRO family, weighted support >= sigma) and scored by
# every variant.
#
# breastCancer is the cohort where the multi-tree choices matter (37809
# candidate trees); TRACERx is a control (367 of 394 patients have one tree).
#
# Run from the repo's MASTRO/ directory:  bash scripts/run_ablation.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."                     # -> <repo>/MASTRO, the code directory

# ---- knobs ------------------------------------------------------------------
DATASETS=${DATASETS:-"breastCancer tracerx"}
SIGMA_LIST=${SIGMA_LIST:-"5"}
N_CAL=${N_CAL:-500}
N_VAL=${N_VAL:-500}
PAR=${PAR:-50}
MC_SAMPLES=${MC_SAMPLES:-3000}   # per-patient MC draws, 'perm' variants only
MC_CUTOFF=${MC_CUTOFF:-8}
ALPHAS=${ALPHAS:-0.01,0.05,0.1}
OUT_BASE=${OUT_BASE:-results/ablation}
WORKDIR=${WORKDIR:-}             # mining scratch; empty = <outdir>/work

# The tracked lcm53/lcm is a macOS binary: rebuild when it cannot run here
# (exit 126/127 = not executable / wrong architecture)
set +e; lcm53/lcm >/dev/null 2>&1; RC=$?; set -e
if [ "$RC" -eq 126 ] || [ "$RC" -eq 127 ]; then
  echo "building LCM"; ( cd lcm53 && make -B )
fi

for DS in $DATASETS; do
  for SIGMA in $SIGMA_LIST; do
    INP=${OUT_BASE}/${DS}_inputs_sigma${SIGMA}
    if [ -f "$INP/inputs/graphs_all.txt" ]; then
      echo "=== [SKIP] inputs ${DS} sigma=${SIGMA} (exist) ==="
    else
      echo "=== [1] build inputs: ${DS} sigma=${SIGMA} ==="
      case "$DS" in
        breastCancer)
          python3 run_pipeline.py --npy ../data/breastCancer.npy --sigma "$SIGMA" --theta_list 1.0 \
            --seed 0 --outdir "$INP" ;;
        tracerx)
          python3 run_pipeline.py --graphs ../data/TRACERx/graphs_tracerx.txt \
            --owner ../data/TRACERx/owner_tracerx.txt --sigma "$SIGMA" --theta_list 1.0 \
            --seed 0 --outdir "$INP" ;;
        *) echo "unknown dataset $DS" >&2; exit 1 ;;
      esac
    fi

    OUT=${OUT_BASE}/${DS}_sigma${SIGMA}
    echo "=== [2] ablation: ${DS} sigma=${SIGMA} n_cal=${N_CAL} n_val=${N_VAL} ==="
    # Resumable: datasets whose p-values are already on disk are not recomputed
    python3 ablation_calibration.py \
      --graphs_all "$INP/inputs/graphs_all.txt" \
      -w           "$INP/inputs/weights_uniform.txt" \
      --owner      "$INP/inputs/owner.txt" \
      --sigma "$SIGMA" --n_cal "$N_CAL" --n_val "$N_VAL" --alphas "$ALPHAS" \
      --mc_cutoff "$MC_CUTOFF" --mc_samples "$MC_SAMPLES" --par "$PAR" \
      --outdir "$OUT" ${WORKDIR:+--workdir "$WORKDIR/${DS}_sigma${SIGMA}"}

    python3 plot_ablation_qq.py --dir "$OUT"
    python3 plot_ablation_qq.py --dir "$OUT" --family mined
  done
done
echo "=== ablation complete -> ${OUT_BASE} ==="

#!/bin/bash
# =============================================================================
# EXPERIMENT: Westfall-Young correction on the null cohorts of
# run_pottr_null_breastcancer.sh.
#
# That driver gives, on every null cohort, the trajectories POTTR returned with
# POTTR's p-value and the UNCORRECTED Multi-MASTRO p-values. An uncorrected
# p-value of a trajectory picked by POTTR's ILP is optimistic under any test,
# so on its own it says nothing about false discoveries. This driver computes
# the WY-corrected thresholds of the null cohort itself (resampled with the
# same per-patient 'perm' null) and counts the trajectories whose Multi-MASTRO
# p-value clears them.
#
# SIGMA=2 makes every POTTR trajectory with expected support >= 2 a member of
# the expected-support family, which is all of them (POTTR's k >= 2). The
# theta threshold covers the theta-maximal family only, which need not contain
# POTTR's trajectories: the expected-support threshold is the one the claim
# rests on.
#
# The breastCancer thresholds of the discovery experiment do NOT apply here:
# that cohort holds every candidate tree (37809), this one at most 2 per
# patient (1759 trees).
#
# Needs seed<S>/inputs and seed<S>/significance.csv from
# run_pottr_null_breastcancer.sh (same OUT_BASE, same seeds).
#
# Run from the repo's MASTRO/ directory:  bash scripts/run_pottr_null_wy.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."                     # -> <repo>/MASTRO, the code directory

# ---- knobs ------------------------------------------------------------------
OUT_BASE=${OUT_BASE:-results/pottr_null}
SEEDS=${SEEDS:-"1"}
SIGMA=${SIGMA:-2}
THETA=${THETA:-1.0}                          # the THETA of the null-cohort run
NULL=${NULL:-perm}
M=${M:-2000}                                 # WY resamples, as in the discovery runs
PAR=${PAR:-20}
MC_SAMPLES=${MC_SAMPLES:-20000}              # unused while every patient has <= MC_CUTOFF trees
MC_CUTOFF=${MC_CUTOFF:-8}
WY_NAME="wy_sigma${SIGMA}_theta${THETA}"

[ -x lcm53/lcm ] || ( echo "building LCM"; cd lcm53 && make )

for SEED in $SEEDS; do
  SD="${OUT_BASE}/seed${SEED}"
  [ -f "${SD}/inputs/graphs_all.txt" ] && [ -f "${SD}/significance.csv" ] \
    || { echo "[err] ${SD}: run run_pottr_null_breastcancer.sh first"; exit 1; }

  echo "=== WY seed=${SEED} sigma=${SIGMA} theta=${THETA} M=${M} par=${PAR} ==="
  python3 run_wy_correction_ensemble.py \
    --graphs_all "${SD}/inputs/graphs_all.txt" \
    -w           "${SD}/inputs/weights_uniform.txt" \
    --owner      "${SD}/inputs/owner.txt" \
    --sigma "$SIGMA" -M "$M" \
    --test both --null "$NULL" --theta "$THETA" --par "$PAR" \
    --min_mine_sigma 2 \
    --mc_cutoff "$MC_CUTOFF" --mc_samples "$MC_SAMPLES" \
    --outdir "${SD}/${WY_NAME}"
done

echo "=== summary with WY ==="
python3 summarize_pottr_null.py --csv "${OUT_BASE}"/seed*/significance.csv \
  --wy_name "$WY_NAME" --out "${OUT_BASE}/summary.csv"

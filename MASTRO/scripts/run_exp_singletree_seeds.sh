#!/bin/bash
# =============================================================================
# EXPERIMENT: single-tree baseline seed sweep across random seeds.
#
# The single-tree baseline (Alg 0) picks ONE tree per patient at random, so its
# results depend on the random seed. To report the mean and variance of how
# many trajectories it mines (FREQUENT) and how many survive correction
# (SIGNIFICANT), and which trajectories flip in or out across samplings,
# we repeat the whole single-tree pipeline over many seeds.
#
# For each (sigma, seed):
#   1. run_pipeline.py  -> sample one tree/patient with this seed, mine Alg 0,
#                          and score it with the (original) MASTRO test.
#   2. run_wy_correction_ensemble.py on the SAMPLED single-tree family
#                          (graphs_sampled / weights_sampled=1.0 / owner_sampled)
#                          -> WY threshold at alpha, i.e. the significant count.
# Then analyze_singletree_seeds.py aggregates mean/sd and the appear/disappear
# set-diff across seeds.
#
# COST: run_pipeline is invoked with --single_tree_only, so the seed-independent
# ensemble families (Alg 1/2/3) are NOT re-mined or re-scored each seed, only
# the cheap single-tree baseline (M_i = 1, exact null) is.
#
# PARALLELISM: seeds and sigmas are independent. Either raise PAR (per-run cores)
# or launch several (SIGMA,SEED) shards concurrently with disjoint SEEDS lists.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
SIGMA_LIST=${SIGMA_LIST:-"2 5"}
M=${M:-2000}                 # WY resamples
PAR=${PAR:-4}
NULL=${NULL:-perm}
MC_SAMPLES=${MC_SAMPLES:-3000}
MC_CUTOFF=${MC_CUTOFF:-8}
ALPHA=${ALPHA:-0.05}
NPY=../data/breastCancer.npy
ROOT=results/singletree_seeds

[ -f "$NPY" ] || { echo "ERROR: $NPY not found" >&2; exit 1; }
[ -x lcm53/lcm ] || ( echo "building LCM"; cd lcm53 && make )

for SIGMA in $SIGMA_LIST; do
  for SEED in $SEEDS; do
    OUT="$ROOT/sigma${SIGMA}/seed${SEED}"

    if [ -f "$OUT/significance/alg0_mastro_random_pvalues_exp.csv" ]; then
      echo "=== [SKIP] mining sigma=$SIGMA seed=$SEED (exists) ==="
    else
      echo "=== [1] single-tree mining + significance: sigma=$SIGMA seed=$SEED ==="
      python3 run_pipeline.py \
        --npy "$NPY" --sigma "$SIGMA" --seed "$SEED" \
        --single_tree_only \
        --significance --sig_null "$NULL" \
        --sig_mc_cutoff "$MC_CUTOFF" --sig_mc_samples "$MC_SAMPLES" --sig_n_jobs "$PAR" \
        --outdir "$OUT"
    fi

    PVAL_ALG0="$OUT/significance/alg0_mastro_random_pvalues_exp.csv"
    WYOUT="$ROOT/sigma${SIGMA}/seed${SEED}_wy"

    if [ -f "$WYOUT/wy_thresholds.txt" ]; then
      echo "=== [SKIP] WY sigma=$SIGMA seed=$SEED (exists) ==="
    else
      echo "=== [2] WY on single-tree family: sigma=$SIGMA seed=$SEED ==="
      python3 run_wy_correction_ensemble.py \
        --graphs_all "$OUT/inputs/graphs_sampled.txt" \
        -w           "$OUT/inputs/weights_sampled.txt" \
        --owner      "$OUT/inputs/owner_sampled.txt" \
        --sigma      "$SIGMA" -M "$M" \
        --test exp --null "$NULL" --par "$PAR" \
        --mc_cutoff "$MC_CUTOFF" --mc_samples "$MC_SAMPLES" \
        --outdir "$WYOUT" \
        --pvalues_exp "$PVAL_ALG0"
    fi
  done
done

echo "=== [3] aggregate across seeds ==="
python3 analyze_singletree_seeds.py \
  --root "$ROOT" --sigma_list "$SIGMA_LIST" --alpha "$ALPHA"

echo "=== single-tree seed sweep complete -> $ROOT/summary.txt ==="

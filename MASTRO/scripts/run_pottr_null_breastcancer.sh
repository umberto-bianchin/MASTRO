#!/bin/bash
# =============================================================================
# EXPERIMENT: does POTTR report significant trajectories on a cohort where
# nothing is conserved?
#
# The real breastCancer cohort (deduplicated, the same matched cohort as
# run_pottr_breastcancer.sh) is turned into null cohorts by relabelling every
# patient with one random bijection of its own alterations, drawn independently
# across patients and shared by all the trees of a patient
# (permute_pottr_cohort.py). Patients, trees and topologies survive; any
# recurrence across patients does not. On every null cohort this driver runs the
# same pipeline as the real-data comparison:
#   (B)  POTTR k-sweep,
#   (B2) POTTR's own p-value for every trajectory,
#   (C)  Multi-MASTRO p-values of the same trajectories,
# and finally (D) counts, per seed, the trajectories below alpha for each test.
#
# Cost: one null cohort costs as much as the real POTTR run (O(T^2) conflict
# graphs, rebuilt by POTTR at every k). Size SEEDS and K_LIST accordingly.
#
# Run from the repo's MASTRO/ directory:  bash scripts/run_pottr_null_breastcancer.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."                     # -> <repo>/MASTRO, the code directory

# ---- knobs ------------------------------------------------------------------
NPY=${NPY:-../data/breastCancer.npy}
POTTR_REPO=${POTTR_REPO:-../../POTTR}        # POTTR clone next to this repo (holds code/)
POTTR_ENV=${POTTR_ENV:-pottr_env}            # conda env with gurobi (environment.yaml)
OUT_BASE=${OUT_BASE:-results/pottr_null}
MAX_TREES=${MAX_TREES:-2}                    # same cohort as the real-data run
SEEDS=${SEEDS:-"1 2 3 4 5"}
K_MIN=${K_MIN:-2}
K_MAX=${K_MAX:-50}
K_LIST=${K_LIST:-"$(seq "$K_MIN" "$K_MAX" | tr '\n' ' ')"}
CORES=${CORES:-20}
POTTR_SIG_MAX_NODES=${POTTR_SIG_MAX_NODES:-6}
THETA=${THETA:-1.0}
NULL=${NULL:-perm}

# ---- (A) the real cohort, built once ----------------------------------------
echo "=== (A) prepare the matched real cohort ==="
python3 breastcancer_to_pottr.py \
  --npy "$NPY" --out "${OUT_BASE}/real/dags" --ensemble_out "${OUT_BASE}/real/inputs" \
  --max_trees "$MAX_TREES" --seed 0

if [ -n "$POTTR_ENV" ] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$POTTR_ENV"; then
    conda activate "$POTTR_ENV"
  else
    echo "[warn] conda env '$POTTR_ENV' not found; using current python3"
  fi
fi
( cd "${POTTR_REPO}/code" && python3 -c "import gurobipy, networkx, pandas" ) \
  || { echo "[err] POTTR deps missing (gurobipy/networkx/pandas) in python3."; exit 1; }

KMIN=$(echo $K_LIST | tr ' ' '\n' | sort -n | head -1)
KMAX=$(echo $K_LIST | tr ' ' '\n' | sort -n | tail -1)

for SEED in $SEEDS; do
  SD="${OUT_BASE}/seed${SEED}"
  echo "=== seed ${SEED} ==="
  python3 permute_pottr_cohort.py \
    --dags_in "${OUT_BASE}/real/dags" --ens_in "${OUT_BASE}/real/inputs" \
    --dags_out "${SD}/dags" --ens_out "${SD}/inputs" --seed "$SEED"

  DAGS_ABS="$(cd "${SD}/dags" && pwd)"
  POTTR_OUT_ABS="$(mkdir -p "${SD}/pottr" && cd "${SD}/pottr" && pwd)"

  # ---- (B) POTTR k-sweep ----------------------------------------------------
  for K in $K_LIST; do
    KOUT="${POTTR_OUT_ABS}/k${K}"
    if [ -f "${KOUT}/converted_graphs.txt" ]; then
      echo "[SKIP] seed=${SEED} k=$K already has converted_graphs.txt"
      continue
    fi
    echo "--- POTTR seed=${SEED} k=$K ---"
    ( cd "${POTTR_REPO}/code" && \
      python3 run_POTTR.py -o "$KOUT" -d "$DAGS_ABS" -k "$K" -c "$CORES" -parallel )
  done

  # ---- (B2) POTTR's own p-value, per trajectory -----------------------------
  python3 pottr_force_significance.py \
    --pottr_dir "${SD}/pottr" --k_range "${KMIN},${KMAX}" \
    --dags "$DAGS_ABS" --pottr_repo "$POTTR_REPO" \
    --max_nodes "$POTTR_SIG_MAX_NODES" --cores "$CORES" \
    --out "${SD}/pottr/significance_forced.txt"

  # ---- (C) Multi-MASTRO on the same trajectories ----------------------------
  python3 pottr_significance.py \
    --pottr_dir "${SD}/pottr" --k_range "${KMIN},${KMAX}" \
    --pottr_sig_global "${SD}/pottr/significance_forced.txt" \
    --graphs_all "${SD}/inputs/graphs_all.txt" \
    -w "${SD}/inputs/weights_uniform.txt" --owner "${SD}/inputs/owner.txt" \
    --theta "$THETA" --null "$NULL" --n_jobs "$CORES" --seed "$SEED" \
    --out "${SD}/significance.csv"
done

# ---- (D) false discoveries per seed -----------------------------------------
echo "=== (D) summary ==="
python3 summarize_pottr_null.py --csv "${OUT_BASE}"/seed*/significance.csv \
  --out "${OUT_BASE}/summary.csv"

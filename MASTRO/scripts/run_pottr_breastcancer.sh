#!/bin/bash
# =============================================================================
# EXPERIMENT: Run POTTR on breastCancer *the correct way*, then re-assess its
# trajectories with the ensemble MASTRO significance test.
#
# Why the naive breastcancer_dags run is wrong (see breastcancer_to_pottr.py):
#   1. Scale: POTTR builds a conflict graph for every pair of trees across
#      patients (O(T^2)). ~2700 candidate trees -> millions of pairs -> the ILP
#      never finishes.
#   2. Support double-counts trees as patients: compute_support.py tallies
#      support by graph name (per tree, "P<i>-<j>") and rescans ALL candidate
#      trees, so one patient's near-identical trees inflate recurrence.
#
# This driver:
#   (A) builds a MATCHED, deduplicated, capped, tractable cohort: POTTR dags
#       AND the ensemble inputs from the identical trees;
#   (B) runs POTTR over a k-sweep, one output dir per k (layout expected by
#       pottr_significance.py: <POTTR_OUT>/k<k>/converted_graphs.txt);
#   (C) re-evaluates the POTTR trajectories under the ensemble test.
#
# Run from the repo's MASTRO/ directory:  bash scripts/run_pottr_breastcancer.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../MASTRO"               # -> MASTRO/ (Python CWD for our tools)
MASTRO_DIR="$(pwd)"

# ---- knobs ------------------------------------------------------------------
NPY=${NPY:-../data/breastCancer.npy}
POTTR_REPO=${POTTR_REPO:-../POTTR}           # repo root that holds code/ (+ data/)
POTTR_ENV=${POTTR_ENV:-pottr_env}            # conda env with gurobi (environment.yaml)
DAGS_DIR=${DAGS_DIR:-${POTTR_REPO}/data/breastcancer_dags}
POTTR_OUT=${POTTR_OUT:-results/pottr_cmp/pottr_bc}      # <- k<k>/ subdirs written here
ENS_DIR=${ENS_DIR:-results/pottr_cmp/bc_inputs}         # matched ensemble inputs
SIG_OUT=${SIG_OUT:-results/pottr_cmp/pottr_bc_significance.csv}

MAX_TREES=${MAX_TREES:-5}                    # distinct trees/patient cap (drives O(T^2))
N_PATIENTS=${N_PATIENTS:-80}                 # 0 = all eligible (may be intractable!)
MULTITREE_ONLY=${MULTITREE_ONLY:-1}          # 1 = only patients with >=2 distinct trees
K_LIST=${K_LIST:-"2 3 5 8 10"}              # POTTR recurrence levels to sweep
CORES=${CORES:-20}
THETA=${THETA:-1.0}
NULL=${NULL:-perm}
SEED=${SEED:-0}

MT_FLAG=""
[ "$MULTITREE_ONLY" = "1" ] && MT_FLAG="--multitree_only"

# ---- (A) build matched, tractable cohort ------------------------------------
echo "=== (A) prepare matched POTTR + ensemble cohort ==="
python3 breastcancer_to_pottr.py \
  --npy "$NPY" --out "$DAGS_DIR" --ensemble_out "$ENS_DIR" \
  --max_trees "$MAX_TREES" --n_patients "$N_PATIENTS" $MT_FLAG --seed "$SEED"

# absolute paths for POTTR (run from POTTR/code with a different CWD)
DAGS_ABS="$(cd "$(dirname "$DAGS_DIR")" && pwd)/$(basename "$DAGS_DIR")"
POTTR_OUT_ABS="$(mkdir -p "$POTTR_OUT" && cd "$POTTR_OUT" && pwd)"

# Optional conda env for POTTR. If Gurobi is installed system-wide, leave
# POTTR_ENV empty (POTTR_ENV="" bash ...) to use the system python3 directly.
# Activation is best-effort: never abort the run just because the env is absent.
if [ -n "$POTTR_ENV" ] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$POTTR_ENV"; then
    conda activate "$POTTR_ENV"
  else
    echo "[warn] conda env '$POTTR_ENV' not found; using current python3"
  fi
fi
# Fail fast with a clear message if POTTR's deps are not importable.
( cd "${POTTR_REPO}/code" && python3 -c "import gurobipy, networkx, pandas" ) \
  || { echo "[err] POTTR deps missing (gurobipy/networkx/pandas) in python3."; \
       echo "      Set POTTR_ENV=<conda env> or install them for system python3."; \
       exit 1; }

# ---- (B) POTTR k-sweep ------------------------------------------------------
echo "=== (B) POTTR k-sweep: k in [$K_LIST] ==="
for K in $K_LIST; do
  KOUT="${POTTR_OUT_ABS}/k${K}"
  if [ -f "${KOUT}/converted_graphs.txt" ]; then
    echo "[SKIP] k=$K already has converted_graphs.txt"
    continue
  fi
  echo "--- POTTR k=$K -> ${KOUT} ---"
  ( cd "${POTTR_REPO}/code" && \
    python3 run_POTTR.py -o "$KOUT" -d "$DAGS_ABS" -k "$K" -c "$CORES" -parallel )
done

# ---- (C) re-assess POTTR trajectories under the ensemble test ---------------
echo "=== (C) ensemble significance of POTTR trajectories ==="
KMIN=$(echo $K_LIST | tr ' ' '\n' | sort -n | head -1)
KMAX=$(echo $K_LIST | tr ' ' '\n' | sort -n | tail -1)
python3 pottr_significance.py \
  --pottr_dir "$POTTR_OUT" --k_range "${KMIN},${KMAX}" \
  --graphs_all "${ENS_DIR}/graphs_all.txt" \
  -w "${ENS_DIR}/weights_uniform.txt" --owner "${ENS_DIR}/owner.txt" \
  --theta "$THETA" --null "$NULL" --n_jobs "$CORES" --seed "$SEED" \
  --out "$SIG_OUT"

echo "=== done: POTTR dags=$DAGS_ABS  pottr_out=$POTTR_OUT_ABS  sig=$SIG_OUT ==="

#!/bin/bash
# =============================================================================
# EXPERIMENT: wall-clock cost of extracting frequent trajectories,
#             Multi-MASTRO against POTTR, on the breast-cancer cohort.
#
# WHY A COHORT SWEEP AND NOT ONE NUMBER
# -------------------------------------
# POTTR builds a conflict graph for every pair of trees across patients, which
# is O(T^2) in the total number of candidate trees. breastCancer holds 37809
# trees, i.e. ~7x10^8 pairs. Running POTTR on that cohort therefore requires a
# deduplicated, capped subset - at 2 distinct trees per patient it becomes 1315
# patients and 1759 trees, 4.7% of the cohort, with the entire heavy tail of
# multi-tree patients removed. This experiment measures where the wall actually
# is, by rebuilding the matched cohort at several caps and timing both methods
# on each.
#
# WHAT IS MEASURED
# ----------------
# Frequent-trajectory extraction only, on both sides. No significance test:
# POTTR's inline MASTRO test is stubbed out by pottr_timing_run.py, and
# mastro_timing_run.py never calls compute_significance_ensemble. No theta
# post-filters either, since POTTR has no analogue of them - use
# run_pipeline.py for theta families.
#
# FAIRNESS
# --------
#  * Both methods read the SAME trees. breastcancer_to_pottr.py writes the
#    POTTR DAGs and the MASTRO transactions from one identical selection.
#  * They run one after the other, never concurrently, on the same machine.
#  * Threads are NOT equalised, because they cannot be: Multi-MASTRO's mining
#    is serial by construction (LCM is a serial C program, convert/filter are
#    serial Python), while POTTR takes -c cores for Gurobi and for building the
#    conflict graph. POTTR is therefore timed TWICE per cohort, at 1 thread and
#    at $CORES threads, and both numbers are reported. Reporting only the
#    multi-core POTTR number would understate it; reporting only the
#    single-core one would overstate it.
#  * Every POTTR invocation gets the same wall-clock budget, $TIMEOUT seconds.
#    A timeout is recorded as a data point, not as a failure.
#
# OUTPUT
# ------
#   $OUTDIR/timing.csv          one row per (cohort, method, threads, k),
#                               rebuilt from the JSON records after every cell,
#                               so it is correct even if the run is interrupted
#   $OUTDIR/cap<N>/...          per-run JSON records and POTTR output dirs
#
# Run from the repo's MASTRO/ directory:
#   bash scripts/run_timing_pottr_vs_mastro.sh
# Quick shakedown before committing the machine to the full sweep:
#   CAP_LIST=1 K_LIST="2 3" TIMEOUT=300 bash scripts/run_timing_pottr_vs_mastro.sh
# =============================================================================
set -euo pipefail

# Locate the code directory rather than assuming a layout: scripts/ sits
# beside MASTRO/ on the server but inside it in the local checkout, so a fixed
# relative path works in one place and silently breaks in the other.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MASTRO_DIR=""
for CAND in "${SCRIPT_DIR}/../MASTRO" "${SCRIPT_DIR}/.." "${SCRIPT_DIR}/../.."; do
  if [ -f "${CAND}/mastro_timing_run.py" ] && [ -x "${CAND}/lcm53/lcm" ]; then
    MASTRO_DIR="$(cd "$CAND" && pwd)"; break
  fi
done
[ -n "$MASTRO_DIR" ] || {
  echo "[err] cannot find the code directory (needs mastro_timing_run.py and lcm53/lcm)."
  echo "      Looked beside and above $SCRIPT_DIR."
  exit 1
}
cd "$MASTRO_DIR"
echo "[info] code dir: $MASTRO_DIR"

# ---- knobs ------------------------------------------------------------------
NPY=${NPY:-../data/breastCancer.npy}
# Interpreter for every Python stage. Override when the PATH python3 is
# not the one carrying numpy / the POTTR deps.
PY=${PY:-python3}
# POTTR lives beside the repo in some checkouts and one level up in others,
# so probe instead of defaulting. POTTR_REPO=<path> overrides the search.
if [ -z "${POTTR_REPO:-}" ]; then
  for CAND in ../POTTR ../../POTTR ../../../POTTR; do
    [ -f "${CAND}/code/run_POTTR.py" ] && { POTTR_REPO="$(cd "$CAND" && pwd)"; break; }
  done
fi
[ -n "${POTTR_REPO:-}" ] || {
  echo "[err] POTTR repo not found. Pass POTTR_REPO=/path/to/POTTR."; exit 1; }
echo "[info] POTTR repo: $POTTR_REPO"
POTTR_ENV=${POTTR_ENV:-pottr_env}
OUTDIR=${OUTDIR:-results/timing_pottr_vs_mastro}

# Distinct trees per patient. 0 means no cap at all, i.e. the full 37809-tree
# cohort, which is expected to time out on the POTTR side and is included
# precisely to record that.
CAP_LIST=${CAP_LIST:-"1 2 3 5 8 0"}

# POTTR solves one ILP per recurrence level, so the full sweep is the honest
# comparison: Multi-MASTRO returns the whole family for a given sigma in a
# single pass, POTTR has to be re-run for every k.
K_LIST=${K_LIST:-"$(seq 2 50 | tr '\n' ' ')"}

SIGMA_LIST=${SIGMA_LIST:-"2 5"}
CORES=${CORES:-20}
TIMEOUT=${TIMEOUT:-3600}
SEED=${SEED:-0}
# Run only one half. SKIP_POTTR=1 also skips the Gurobi dependency check, so
# the MASTRO side can be exercised on a machine without a Gurobi licence.
SKIP_POTTR=${SKIP_POTTR:-0}
SKIP_MASTRO=${SKIP_MASTRO:-0}
# Abandon the k-sweep of a cohort after this many CONSECUTIVE timeouts. The
# cost is dominated by the ILP, and how that scales with k is not known, so
# one slow k is not evidence that the rest are slow. 0 disables the cutoff.
MAX_CONSEC_TIMEOUTS=${MAX_CONSEC_TIMEOUTS:-3}
# Thread settings to time POTTR at. The single-threaded arm is worth one
# documented data point, but it did not finish even the easiest instance, so
# it is not worth repeating across the sweep: POTTR_THREADS="$CORES".
POTTR_THREADS=${POTTR_THREADS:-"1 $CORES"}
# Gurobi solution pool. run_POTTR defaults to 0, i.e. one optimal solution;
# the POTTR paper used 50000 when comparing against MASTRO, to collect the
# co-optimal trajectories. Timings at the two settings are not comparable.
POOL=${POOL:-0}
# Pass -v to POTTR so Gurobi prints its MIP log. Worth having when a run is
# expected to time out: the log shows the gap it was still sitting at.
POTTR_VERBOSE=${POTTR_VERBOSE:-1}

mkdir -p "$OUTDIR"
# timing.csv is rebuilt from the JSON records at the end of the run, not
# appended to as we go: appending duplicated every row on a resume.
CSV="${OUTDIR}/timing.csv"

# ---- POTTR environment ------------------------------------------------------
if [ -n "$POTTR_ENV" ] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$POTTR_ENV"; then
    conda activate "$POTTR_ENV"
  else
    echo "[warn] conda env '$POTTR_ENV' not found; using current python3"
  fi
fi
if [ "$SKIP_POTTR" != "1" ]; then
  ( cd "${POTTR_REPO}/code" && "$PY" -c "import gurobipy, networkx, pandas" ) \
    || { echo "[err] POTTR deps missing (gurobipy/networkx/pandas)."; \
         echo "      Set POTTR_ENV=<conda env> or install them for python3."; exit 1; }
fi

# Record what the numbers were measured on: without this the table is unusable
# in a paper, and a stale hardware sentence is easy to leave behind.
{
  echo "date          : $(date -Iseconds)"
  echo "host          : $(hostname)"
  echo "kernel        : $(uname -srm)"
  echo "cpu model     : $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "cpu cores     : $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  echo "memory        : $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo unknown)"
  echo "python        : $("$PY" --version 2>&1)"
  echo "gurobi        : $("$PY" -c 'import gurobipy; print(gurobipy.gurobi.version())' 2>/dev/null || echo unknown)"
  echo "cores knob    : $CORES"
  echo "timeout       : ${TIMEOUT}s"
  echo "cap list      : $CAP_LIST"
  echo "k list        : $(echo $K_LIST | tr ' ' '\n' | head -1)..$(echo $K_LIST | tr ' ' '\n' | tail -1)"
  echo "sigma list    : $SIGMA_LIST"
} > "${OUTDIR}/environment.txt"
cat "${OUTDIR}/environment.txt"

# ---- sweep ------------------------------------------------------------------
for CAP in $CAP_LIST; do
  echo
  echo "############ cohort: max_trees=${CAP} ############"
  CDIR="${OUTDIR}/cap${CAP}"
  ENS_DIR="${CDIR}/inputs"
  DAGS_DIR="${CDIR}/dags"
  mkdir -p "$CDIR"

  # --- (A) matched cohort: same trees to both methods ---
  if [ -f "${ENS_DIR}/manifest.csv" ]; then
    echo "[SKIP] cohort cap=${CAP} already built"
  else
    echo "=== (A) build matched cohort, max_trees=${CAP} ==="
    # max_trees 0 means "no cap"; the converter takes a large number instead.
    MT=$CAP; [ "$CAP" = "0" ] && MT=1000000
    "$PY" breastcancer_to_pottr.py \
      --npy "$NPY" --out "$DAGS_DIR" --ensemble_out "$ENS_DIR" \
      --max_trees "$MT" --n_patients 0 --seed "$SEED"
  fi

  N_PAT=$(awk -F, 'NR>1{print $1}' "${ENS_DIR}/manifest.csv" | sort -u | wc -l | tr -d ' ')
  N_TREE=$(grep -cve '^[[:space:]]*$' "${ENS_DIR}/graphs_all.txt" | tr -d ' ')
  echo "    cohort: ${N_PAT} patients, ${N_TREE} trees"

  # --- (B) Multi-MASTRO, one pass per sigma, serial ---
  for S in $SIGMA_LIST; do
    [ "$SKIP_MASTRO" = "1" ] && break
    J="${CDIR}/mastro_sigma${S}.json"
    if [ -f "$J" ]; then
      echo "[SKIP] mastro sigma=${S}"
    else
      echo "=== (B) Multi-MASTRO mining, sigma=${S} ==="
      rm -rf "${CDIR}/mastro_work_s${S}"
      if timeout "$TIMEOUT" "$PY" mastro_timing_run.py \
           --inputs "$ENS_DIR" --sigma "$S" \
           --workdir "${CDIR}/mastro_work_s${S}" \
           --label "cap${CAP}" --out "$J"; then
        :
      else
        echo "{\"method\":\"multi-mastro\",\"sigma\":${S},\"completed\":false,\"status\":\"timeout_or_error\",\"total_s\":${TIMEOUT}}" > "$J"
        echo "    [TIMEOUT/ERR] mastro sigma=${S}"
      fi
    fi
    # Refresh the CSV after every cell, not only at the end: these runs get
    # killed or time out mid-sweep, and a rebuild costs milliseconds against
    # cells that take hours. The CSV is then always current on disk.
    "$PY" "${MASTRO_DIR}/rebuild_timing_csv.py" --outdir "$OUTDIR" >/dev/null
  done

  # --- (C) POTTR, one ILP per k, at 1 thread and at $CORES threads ---
  for TH in $POTTR_THREADS; do
    [ "$SKIP_POTTR" = "1" ] && break
    CONSEC=0
    PFLAG=""; [ "$TH" != "1" ] && PFLAG="--parallel"
    for K in $K_LIST; do
      J="${CDIR}/pottr_t${TH}_k${K}.json"
      if [ -f "$J" ]; then
        echo "[SKIP] pottr threads=${TH} k=${K}"
      else
        echo "=== (C) POTTR k=${K}, threads=${TH} ==="
        KOUT="${CDIR}/pottr_t${TH}/k${K}"
        LOG="${CDIR}/pottr_t${TH}_k${K}.log"
        mkdir -p "$KOUT"
        VFLAG=""; [ "$POTTR_VERBOSE" = "1" ] && VFLAG="--pottr_verbose"
        if timeout "$TIMEOUT" "$PY" "${MASTRO_DIR}/pottr_timing_run.py" \
             --pottr_repo "$POTTR_REPO" --dags "$(cd "$DAGS_DIR" && pwd)" \
             -k "$K" -c "$TH" --pool "$POOL" $PFLAG $VFLAG \
             --output_path "$KOUT" --label "cap${CAP}" --out "$J" 2>&1 | tee "$LOG"; then
          :
        else
          # pottr_timing_run.py writes a partial record on SIGTERM. Only fall
          # back to a bare stub when it left nothing, or the stage timings the
          # timeout was meant to expose would be thrown away here.
          if ! "$PY" -c "import json,sys; d=json.load(open('$J')); sys.exit(0 if d.get('stages') else 1)" 2>/dev/null; then
            echo "{\"method\":\"pottr\",\"k\":${K},\"threads\":${TH},\"completed\":false,\"status\":\"timeout_no_partial\",\"total_s\":${TIMEOUT},\"stages\":[]}" > "$J"
          fi
          echo "    [TIMEOUT/ERR] pottr k=${K} threads=${TH} (log: $LOG)"
        fi
      fi
      "$PY" "${MASTRO_DIR}/rebuild_timing_csv.py" --outdir "$OUTDIR" >/dev/null
      if "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$J')).get('completed') else 1)"; then
        CONSEC=0
      else
        CONSEC=$((CONSEC + 1))
        if [ "$MAX_CONSEC_TIMEOUTS" != "0" ] && [ "$CONSEC" -ge "$MAX_CONSEC_TIMEOUTS" ]; then
          echo "    [STOP] ${CONSEC} consecutive timeouts; abandoning the k-sweep (cap=${CAP}, threads=${TH})"
          for KR in $K_LIST; do
            [ "$KR" -le "$K" ] && continue
            [ -f "${CDIR}/pottr_t${TH}_k${KR}.json" ] && continue
            echo "{\"method\":\"pottr\",\"k\":${KR},\"threads\":${TH},\"completed\":false,\"status\":\"not_attempted_after_timeouts\",\"total_s\":null,\"stages\":[]}" \
              > "${CDIR}/pottr_t${TH}_k${KR}.json"
          done
          break
        fi
      fi
    done
  done
done

echo
echo "=== done ==="
"$PY" "${MASTRO_DIR}/rebuild_timing_csv.py" --outdir "$OUTDIR"
echo "csv         : $CSV"
echo "environment : ${OUTDIR}/environment.txt"
column -s, -t "$CSV" | cut -c1-140 | head -40

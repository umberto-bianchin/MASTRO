#!/bin/bash
#SBATCH --job-name=pottr-null
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err
#SBATCH --partition=allgroups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --array=2-10
# =============================================================================
# scripts/run_pottr_null_breastcancer.sh on one DEI compute node per seed,
# inside ~/images/pottr.sif (built from scripts/dei/pottr.def). The array index
# is the seed; CORES (Gurobi threads, MASTRO jobs) follows --cpus-per-task.
#
# Needs: POTTR cloned next to the repo (~/POTTR), and a Gurobi license: one
# named academic license per node in ~/gurobi/<node>.lic (restrict the array to
# those nodes), else an Academic WLS one at ~/gurobi.lic (2 sessions at a time:
# submit with --array=...%2).
#
# Submit from the code directory (<repo>/MASTRO):
#   cd ~/MASTRO/MASTRO
#   sbatch scripts/dei/run_pottr_null.slurm                       # seeds 2..10
#   sbatch --array=2 --time=12:00:00 scripts/dei/run_pottr_null.slurm   # one seed
# 30 cpus fit the 32-core runners (2 cores per node are reserved).
#
# Resumable per k: a resubmitted seed skips every k with converted_graphs.txt.
# The summary over all seeds runs afterwards (SUMMARY=0 here):
#   python3 summarize_pottr_null.py --csv results/pottr_null/seed*/significance.csv \
#     --wy_file results/pottr_null/seed1/wy_sigma2_theta1.0/wy_thresholds.txt \
#     --out results/pottr_null/summary.csv
# =============================================================================
set -euo pipefail

SIF=${SIF:-$HOME/images/pottr.sif}
cd "$SLURM_SUBMIT_DIR"
[ -f scripts/run_pottr_null_breastcancer.sh ] || { echo "submit from <repo>/MASTRO" >&2; exit 1; }
[ -f "$SIF" ] || { echo "image $SIF not found" >&2; exit 1; }

# A named academic license is bound to one node: ~/gurobi/<node>.lic when it
# exists. The WLS fallback allows only 2 concurrent sessions ("Too many sessions").
if [ -z "${GRB_LICENSE_FILE:-}" ] && [ -f "$HOME/gurobi/$(hostname -s).lic" ]; then
  export GRB_LICENSE_FILE=$HOME/gurobi/$(hostname -s).lic
fi
export GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-$HOME/gurobi.lic}
[ -f "$GRB_LICENSE_FILE" ] || { echo "Gurobi license $GRB_LICENSE_FILE not found" >&2; exit 1; }

export SEEDS=${SLURM_ARRAY_TASK_ID}
export K_LIST=${K_LIST:-"2 3 5 8 12 17 22 29"}   # the seed-1 sweep
export CORES=${CORES:-$SLURM_CPUS_PER_TASK}
export POTTR_ENV=""
export SUMMARY=0
echo "node=$(hostname) seed=$SEEDS cpus=$SLURM_CPUS_PER_TASK K_LIST=$K_LIST license=$GRB_LICENSE_FILE"
srun apptainer exec "$SIF" bash scripts/run_pottr_null_breastcancer.sh

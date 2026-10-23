#!/bin/bash
#SBATCH --job-name=mastro-power
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=allgroups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
# =============================================================================
# scripts/run_exp_power.sh on one DEI compute node, inside ~/images/mastro.sif
# (built from scripts/dei/mastro.def). PAR follows --cpus-per-task; 30 cpus fit
# the 32-core runners (2 cores per node are reserved).
#
# Submit from the code directory (<repo>/MASTRO); the knobs of run_exp_power.sh
# pass through the environment:
#   cd ~/MASTRO/MASTRO
#   OUT_BASE=results/power_M10 sbatch scripts/dei/run_power.slurm
#
# The run is resumable per (N, f) cell through <OUT_BASE>/implant_theta*/
# recall.csv, and the WY threshold is cached in _null/thresholds.json: neither
# depends on M, so a run with a different M needs a fresh OUT_BASE.
# =============================================================================
set -euo pipefail

SIF=${SIF:-$HOME/images/mastro.sif}
cd "$SLURM_SUBMIT_DIR"
[ -f scripts/run_exp_power.sh ] || { echo "submit from <repo>/MASTRO" >&2; exit 1; }
[ -f "$SIF" ] || { echo "image $SIF not found" >&2; exit 1; }

export PAR=${PAR:-$SLURM_CPUS_PER_TASK}
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK PAR=$PAR MTREES=${MTREES:-10} OUT_BASE=${OUT_BASE:-results/power}"
srun apptainer exec "$SIF" bash scripts/run_exp_power.sh

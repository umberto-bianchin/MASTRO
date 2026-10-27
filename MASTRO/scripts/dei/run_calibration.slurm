#!/bin/bash
#SBATCH --job-name=mastro-calibration
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=allgroups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
# =============================================================================
# scripts/run_exp_calibration_<COHORT>.sh on one DEI compute node, inside
# ~/images/mastro.sif (built from scripts/dei/mastro.def). PAR follows
# --cpus-per-task; COHORT is breastcancer (default) or tracerx.
#
# Submit from the code directory (<repo>/MASTRO); the knobs of the calibration
# script pass through the environment:
#   cd ~/MASTRO/MASTRO
#   MC_SAMPLES=20000 OUT_ROOT=results/calibration_exact \
#     sbatch scripts/dei/run_calibration.slurm
#   COHORT=tracerx MC_SAMPLES=20000 OUT_ROOT=results/calibration_exact \
#     sbatch --cpus-per-task=16 --mem=64G scripts/dei/run_calibration.slurm
#
# The pool is NOT resumable: empirical_fwer_ensemble writes it only at the end
# of each theta.
# =============================================================================
set -euo pipefail

SIF=${SIF:-$HOME/images/mastro.sif}
COHORT=${COHORT:-breastcancer}
cd "$SLURM_SUBMIT_DIR"
[ -f "scripts/run_exp_calibration_${COHORT}.sh" ] || { echo "submit from <repo>/MASTRO" >&2; exit 1; }
[ -f "$SIF" ] || { echo "image $SIF not found" >&2; exit 1; }

export PAR=${PAR:-$SLURM_CPUS_PER_TASK}
# Null-dataset mining scratch in RAM on the node, not in the 50 GB home quota
# (sigma=2 mining can write tens of GB per dataset). tmpfs counts against --mem.
export WORKDIR=${WORKDIR:-/dev/shm/$USER-$SLURM_JOB_ID}
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK PAR=$PAR COHORT=$COHORT EXACT_WHEN_CHEAPER=${EXACT_WHEN_CHEAPER:-1} MC_SAMPLES=${MC_SAMPLES:-default} OUT_ROOT=${OUT_ROOT:-default} WORKDIR=$WORKDIR"
srun apptainer exec "$SIF" bash "scripts/run_exp_calibration_${COHORT}.sh"

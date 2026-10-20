#!/bin/bash
#SBATCH --job-name=mastro-ablation
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=allgroups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=94
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
# =============================================================================
# scripts/run_ablation.sh on one DEI compute node, inside ~/images/mastro.sif
# (built from scripts/dei/mastro.def). PAR follows --cpus-per-task.
#
# Submit from the code directory (<repo>/MASTRO); the knobs of run_ablation.sh
# pass through the environment, and sbatch flags override the defaults above:
#   cd ~/MASTRO/MASTRO
#   DATASETS=breastCancer N_CAL=20 N_VAL=20 \
#     sbatch --time=06:00:00 scripts/dei/run_ablation.slurm      # timing run
#   DATASETS=breastCancer sbatch scripts/dei/run_ablation.slurm   # full run
#   DATASETS=tracerx sbatch --cpus-per-task=30 --mem=64G scripts/dei/run_ablation.slurm
#   2 cores per node are reserved for the system: 94 fits a 96-core runner,
#   190 is the whole runner-11:  sbatch --cpus-per-task=190 --nodelist=runner-11 ...
#
# The run is resumable: if the job hits its time limit, submit it again.
# Check the peak memory afterwards with  sacct -j <JOBID> --format=MaxRSS,Elapsed
# =============================================================================
set -euo pipefail

SIF=${SIF:-$HOME/images/mastro.sif}
cd "$SLURM_SUBMIT_DIR"
[ -f scripts/run_ablation.sh ] || { echo "submit from <repo>/MASTRO" >&2; exit 1; }
[ -f "$SIF" ] || { echo "image $SIF not found" >&2; exit 1; }

export PAR=${PAR:-$SLURM_CPUS_PER_TASK}
# Mining scratch in RAM on the node, not in the 50 GB home quota: at sigma=2 a
# single null dataset wrote 22 GB of LCM output. tmpfs counts against --mem.
export WORKDIR=${WORKDIR:-/dev/shm/$USER-$SLURM_JOB_ID}
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK PAR=$PAR DATASETS=${DATASETS:-default} SIGMA_LIST=${SIGMA_LIST:-default} WORKDIR=$WORKDIR"
srun apptainer exec "$SIF" bash scripts/run_ablation.sh

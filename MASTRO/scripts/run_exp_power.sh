#!/bin/bash
# =============================================================================
# EXPERIMENT: Power / recovery of an implanted trajectory (ensemble).
#
# Implants a known trajectory into N carrier patients with per-patient
# consistency f, then measures recall at a WY-corrected FWER threshold.
# Demonstrates the exp-vs-theta regime: the expected-support test recovers P
# once N*f is large, the theta-consensus test recovers it only when f >= theta.
#
# Fully self-contained (synthesises its own cohorts), safe to run in parallel
# with everything else. Runs one sweep per theta value.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../MASTRO"

PAR=${PAR:-4}
THETA_LIST=${THETA_LIST:-"0.5 1.0"}
NULL=${NULL:-perm}
N_PATIENTS=${N_PATIENTS:-60}
MTREES=${MTREES:-6}
K=${K:-4}
N_LIST=${N_LIST:-10,20,30}
F_LIST=${F_LIST:-0.2,0.4,0.6,0.8,1.0}
SIGMA=${SIGMA:-2.0}
ALPHA=${ALPHA:-0.05}
N_TRIALS=${N_TRIALS:-200}
WY_M=${WY_M:-500}
# Output base: point to a disk with space (e.g. OUT_BASE=/mnt/ssd1/power_out)
# when /home is full. recall.csv + WY cache + trial scratch all go here.
OUT_BASE=${OUT_BASE:-results/power}

[ -x lcm53/lcm ] || ( echo "building LCM"; cd lcm53 && make )

for THETA in $THETA_LIST; do
  echo "=== power sweep: theta=$THETA null=$NULL ==="
  python3 implant_experiment_ensemble.py \
    --n_patients "$N_PATIENTS" --M "$MTREES" --k "$K" \
    --N_list "$N_LIST" --f_list "$F_LIST" \
    --theta "$THETA" --sigma "$SIGMA" --alpha "$ALPHA" \
    --null "$NULL" --test both \
    --n_trials "$N_TRIALS" --wy_M "$WY_M" \
    --mc_cutoff 8 --mc_samples 2000 --par "$PAR" \
    --outdir "${OUT_BASE}/implant_theta${THETA}"
done
echo "=== power experiment complete -> ${OUT_BASE} ==="

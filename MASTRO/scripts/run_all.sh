#!/bin/bash
# =============================================================================
# Launch ALL experiments in parallel under a 20-core budget.
# 5 jobs x PAR=4 workers = 20 cores total. Run this inside ONE screen session,
# then detach (Ctrl-A D) and close SSH, it keeps going and `wait`s for all.
#
#   screen -S mastro
#   bash scripts/run_all.sh
#   (Ctrl-A D to detach;  screen -r mastro to reattach)
#
# Logs: one per job under logs/. Override PAR to rebalance (keep the sum <= 20).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p ../logs
LOG=../logs

TOTAL=${TOTAL:-64}          # total core budget to divide across the 5 jobs
PAR=${PAR:-$((TOTAL / 5))}  # equal split (default 64/5 = 12 workers per job)
export PAR

echo "[run_all] launching 5 jobs at PAR=$PAR each ($((PAR*5)) of $TOTAL cores)"
bash run_exp_discovery_breastcancer.sh   > "$LOG/disc_bc.log"  2>&1 &
bash run_exp_discovery_tracerx.sh        > "$LOG/disc_tx.log"  2>&1 &
bash run_exp_calibration_breastcancer.sh > "$LOG/cal_bc.log"   2>&1 &
bash run_exp_calibration_tracerx.sh      > "$LOG/cal_tx.log"   2>&1 &
bash run_exp_power.sh                    > "$LOG/power.log"    2>&1 &
wait
echo "[run_all] all experiments finished. Outputs under results/, logs under logs/"

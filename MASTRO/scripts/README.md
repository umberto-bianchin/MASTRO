# Ensemble-MASTRO experiments

Each script is self-contained and writes under `results/` (relative to the code
dir, i.e. `MASTRO/MASTRO/results/`). All five are independent, so on a many-core
server you can launch them **in parallel** and split cores with the `PAR` env var.

These scripts live inside the code dir and `cd "$(dirname "$0")/.."` to it on
start; the datasets are read from `../data` (`MASTRO/data/`). `lcm53/lcm` is
built automatically on first run.

## The experiments

| Script | Experiment | Notes |
|---|---|---|
| `run_exp_discovery_breastcancer.sh` | Real-data discovery (Exp 3) | mining + significance + WY/FDR |
| `run_exp_discovery_tracerx.sh`      | Real-data discovery (Exp 3) | TRACERx, `--graphs`/`--owner` mode |
| `run_exp_calibration_breastcancer.sh` | Empirical FWER control (Exp 1) | validates the WY thresholds |
| `run_exp_calibration_tracerx.sh`      | Empirical FWER control (Exp 1) | TRACERx |
| `run_exp_power.sh`                  | Implant / recovery power (Exp 2) | fully synthetic, exp-vs-theta |

Each test is scored on **its own mined family** (expected-support on Alg 1,
theta-consensus on the Alg 3 theta-maximal post-filter) both on the observed
data and on every resample, so exp and theta FDR/WY are internally consistent.
p-values are add-one smoothed, so none is ever 0.

## Running on the server (20-core budget)

You can use at most **20 cores**. Each script defaults to `PAR=4`, so the five
jobs together use exactly 5 x 4 = 20 cores. RAM is a few GB per worker (tensors
are shared copy-on-write, not copied), so the total stays far under any limit.

**Recommended: one screen, one command.**

    cd MASTRO/MASTRO
    screen -S mastro          # start a detachable session
    bash scripts/run_all.sh   # launches all 5 jobs (PAR=4 each) and waits
    # Ctrl-A then D to detach; you can now close SSH.
    # screen -r mastro   to reattach;   tail -f logs/*.log   to watch.

`run_all.sh` backgrounds the five jobs and `wait`s, writing one log per job to
`logs/`. All jobs run from the **same folder**, safe now that each mines into
its own workdir (no shared LCM scratch).

**Alternatives**
- One job at a time, full speed: `PAR=18 bash scripts/run_exp_discovery_breastcancer.sh`
  (own screen, or back-to-back). Same total core-seconds; a single job finishes
  sooner.
- Rebalance the parallel mix by exporting `PAR` per job (keep the **sum <= 20**);
  calibration is heaviest (one full mining per null dataset), discovery lightest.

Other tunables (sane defaults): `M` (WY resamples), `SIGMA_LIST`, `THETA_LIST`,
`NULL` (perm|indep), and for calibration `N_DATASETS`/`M_CAL`/`N_TRIALS`. Quick
dev pass: `M=1000 N_DATASETS=200 M_CAL=100 N_TRIALS=200`.

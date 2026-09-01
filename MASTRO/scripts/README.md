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

## POTTR comparison (breast cancer)

`run_pottr_breastcancer.sh` rebuilds the whole POTTR side of the comparison from
scratch: matched cohort, POTTR k-sweep, POTTR's own permutation test, and the
re-scoring under the multi-tree test. Defaults reproduce the reported run
(whole cohort, `MAX_TREES=2`, `k = 2..50`), so:

    cd MASTRO/MASTRO
    screen -S pottr
    POTTR_ENV="" CORES=20 bash scripts/run_pottr_breastcancer.sh 2>&1 | tee logs/pottr_bc.log

Needs Gurobi: leave `POTTR_ENV=""` for a system-wide install, or set it to the
conda env built from `POTTR/code/environment.yaml`. Expect hours to a day — the
ILP is O(T^2) in the number of candidate trees (~1750 here).

Stages, in order, each resumable (step B skips any `k<k>/` that already has
`converted_graphs.txt`, so a killed run restarts where it stopped):

- **(A)** `breastcancer_to_pottr.py` → POTTR dags + the matched multi-tree
  inputs (`results/pottr_cmp/bc_inputs/`) from the identical trees.
- **(B)** `run_POTTR.py` once per k → `results/pottr_cmp/pottr_bc/k<k>/`.
- **(B2)** `pottr_force_significance.py` → `pottr_bc/significance_forced.txt`.
  `run_POTTR.py` runs its own significance test only when the *largest*
  trajectory of that k has fewer than 13 nodes, which skips whole k directories,
  small trajectories included — that is why most trajectories had no
  `p_POTTR` in the table. This re-applies the gate per trajectory
  (`POTTR_SIG_MAX_NODES`, default 6; cost grows as n! and C(tree_nodes, n)) and
  never overwrites POTTR's own files. Set `FORCE_POTTR_SIG=0` to skip it.
- **(C)** `pottr_significance.py` → `results/pottr_cmp/pottr_bc_significance.csv`,
  one row per trajectory: trees, patients, `s^exp`, the two multi-tree p-values,
  and POTTR's own p-value.

Then `plot_pottr_comparison.py` draws the trees-vs-patients figure from that CSV.

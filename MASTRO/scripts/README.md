# Experiments

One self-contained script per experiment reported in the paper. Each one runs
end to end from the datasets in `data/`, needs no arguments, and writes its
results under `results/`.

## Conventions shared by every script

**Configuration is by environment variable.** Every script runs a full
experiment with its defaults; exporting a variable narrows or rescales it. For
example `SIGMA_LIST=5 THETA_LIST=1.0 bash run_exp_discovery_breastcancer.sh`
restricts the discovery run to one support threshold and one consensus level.

**Runs are resumable.** Every stage skips itself when its output already
exists, so an interrupted run can be restarted with the same command and will
pick up where it stopped. Deleting an output file is how you force that stage
to be recomputed.

**`PAR` sets the number of worker processes** used for the per-patient null
distributions. It is the only knob that trades wall-clock time for cores; the
default is deliberately small so that several experiments can run side by side.
Memory per worker is modest, since the per-patient tensors are shared
copy-on-write rather than copied.

**`lcm53/lcm` is built on first use**, so a fresh checkout needs no setup step
beyond the Python dependencies.

## The experiments

### Discovery on real cohorts

`run_exp_discovery_breastcancer.sh`, `run_exp_discovery_tracerx.sh`

Mines the four trajectory families, scores each one under the null, then runs
the Westfall-Young correction and the empirical FDR estimate, and plots the
FDR(k) curves. This is the experiment that produces the reported significant
trajectories.

Writes `results/discovery/<cohort>_sigma<S>/` (mined families and p-values),
`results/discovery/<cohort>_wy_sigma<S>_theta<T>/` (corrected thresholds,
resample minima, FDR curves) and `results/fdr_plots/`.

Knobs: `SIGMA_LIST`, `THETA_LIST`, `M` (Westfall-Young resamples), `MC_SAMPLES`,
`MC_CUTOFF`, `NULL`, `PAR`.

### Calibration of the error control

`run_exp_calibration_breastcancer.sh`, `run_exp_calibration_tracerx.sh`

Checks that the corrected thresholds do what they claim. It builds a pool of
datasets from the global null, where no trajectory is truly conserved, then
repeatedly splits the pool into one half that fixes the threshold and an
independent half that measures how often anything is rejected. A valid
procedure rejects at most a fraction alpha of the time. The result is a mean
and standard deviation of the empirical family-wise error rate at each nominal
alpha, for both tests.

This is the heaviest experiment of the set: each null dataset has to be mined
in full.

Writes `results/calibration/<cohort>_sigma<S>_theta<T>/`, alongside the
materialised inputs in `results/calibration/<cohort>_inputs_sigma<S>/`.

Knobs: `N_DATASETS` (size of the null pool), `M_CAL` (datasets used to fix the
threshold), `N_TRIALS` (calibration/validation splits averaged over), `ALPHAS`,
plus the usual `SIGMA_LIST`, `THETA_LIST`, `MC_SAMPLES`, `MC_CUTOFF`, `NULL`,
`PAR`.

### Recovery power

`run_exp_power.sh`

Measures what each test can actually find. It synthesises cohorts, implants a
known trajectory into `N` carrier patients so that a fraction `f` of each
carrier's trees contains it, and reports how often the implanted trajectory is
recovered at a corrected threshold.

The two tests separate here by construction: the expected-support test recovers
the trajectory once `N*f` is large enough, whatever the split between `N` and
`f`, while the theta-consensus test can only recover it when `f >= theta`, no
matter how many carriers there are. Sweeping both axes shows the trade-off.

Fully synthetic, so it needs none of the cohorts in `data/`.

Writes `results/power/implant_theta<T>/recall.csv`.

Knobs: `N_LIST` (carriers), `F_LIST` (per-patient consistency), `THETA_LIST`,
`N_PATIENTS` (cohort size), `MTREES` (trees per patient), `K` (implanted
trajectory length), `SIGMA`, `ALPHA`, `N_TRIALS`, `WY_M`, `PAR`.

### Seed dependence of the single-tree baseline

`run_exp_singletree_seeds.sh`

Quantifies the cost of choosing one tree per patient. The single-tree baseline
samples one tree per patient at random, so everything it reports depends on
that draw. This repeats the entire single-tree pipeline over many seeds and
reports how many trajectories are mined and how many survive correction, as a
mean and standard deviation, plus which trajectories appear and disappear
between seeds.

Only the baseline is re-mined per seed: the ensemble families do not depend on
the seed and would be recomputed identically every time.

Writes `results/singletree_seeds/sigma<S>/seed<K>/` plus a `summary.txt`
aggregating across seeds.

Knobs: `SEEDS`, `SIGMA_LIST`, `ALPHA`, `M`, `MC_SAMPLES`, `MC_CUTOFF`, `NULL`,
`PAR`.

### Mining overhead

`run_exp_spurious_timing.sh`

Measures what the trajectory structure costs on top of plain itemset mining.
Most of what the miner enumerates is not a valid trajectory: an itemset is kept
only if it is relation-complete, meaning every pair of its alterations carries
a relation, and the rest is discarded downstream. This counts how many itemsets
are discarded at each support threshold and times the stages separately, so the
enumeration itself can be compared against the filtering that follows it.

Writes `results/spurious_analysis/` (`spurious_summary.csv`, a per-stage
breakdown in JSON, and the figures).

Knobs: `SIGMAS`, `MAXITEMS` (cap on enumerated itemsets), `TIMEOUT` (per mining
invocation).

Timings are hardware-dependent; record the machine alongside any number taken
from this script.

### Comparison against POTTR

`run_pottr_breastcancer.sh`

POTTR also mines recurrent trajectories from uncertain phylogenies, but it
handles the uncertainty at the discovery stage: an ILP selects at most one tree
per patient, and support is then counted over candidate trees. This experiment
takes POTTR's own trajectories and re-scores them under the multi-tree test on
exactly the same trees, so the two differ only in how patient evidence is
counted.

Four stages, each resumable:

1. Build a matched cohort: POTTR DAG files and the multi-tree inputs are
   written from one identical selection of trees, so the two sides describe the
   same data. Trees identical within a patient are deduplicated and the number
   of trees per patient is capped, which is what keeps the ILP tractable:
   POTTR builds a conflict graph for every pair of trees across patients.
2. Run POTTR once per recurrence level `k`.
3. Re-apply POTTR's own significance test per trajectory. POTTR runs it only
   when the largest trajectory of a given `k` is small enough, which leaves
   whole levels unscored, small trajectories included. This fills them in
   without touching POTTR's own output files.
4. Score every POTTR trajectory under the multi-tree test.

Writes `results/pottr_cmp/`: the matched inputs, one directory per `k`, a CSV
with one row per trajectory (its support in trees, its support in distinct
patients, its expected support, the multi-tree p-values and POTTR's own), and
the comparison figure.

Requires a POTTR checkout and a Gurobi licence. Point `POTTR_REPO` at the
checkout; set `POTTR_ENV` to a conda environment providing Gurobi, or to the
empty string to use the current interpreter. Expect the ILP to dominate the
runtime.

Knobs: `MAX_TREES` (trees per patient after deduplication), `K_MIN`/`K_MAX`,
`N_PATIENTS`, `MULTITREE_ONLY`, `CORES`, `THETA`, `FORCE_POTTR_SIG`,
`POTTR_SIG_MAX_NODES`, `SEED`.

### Runtime comparison against POTTR

`run_timing_pottr_vs_mastro.sh`

Times both methods on identical trees, one after the other so neither competes
with the other for cores. It rebuilds the matched cohort at several per-patient
tree caps and, for each one, times the Multi-MASTRO mining and POTTR at a range
of recurrence levels, with a wall-clock budget per invocation. A timeout is
recorded as a data point, including which stage it was in when it was stopped.

Writes `results/timing_pottr_vs_mastro/`: `timing.csv` with one row per
configuration and a per-stage breakdown, `environment.txt` recording the
machine, and a JSON record plus a solver log per run.

This one is a campaign of several runs rather than a single invocation. The
last section of this file documents each of them.

## A note on how significance is computed

Each test is scored on **its own mined family**, the expected-support test on
the expected-support family and the theta-consensus test on the theta-maximal
post-filter, on the observed data and on every resample alike. The observed
and null distributions therefore refer to the same set of hypotheses, and
neither test borrows the other's null. Monte-Carlo p-values are add-one
smoothed, so none is ever exactly zero.

## Runtime comparison: the campaign in detail

### The pieces

| File | Role |
|---|---|
| `scripts/run_timing_pottr_vs_mastro.sh` | the driver: builds matched cohorts, runs both methods one after the other, writes one CSV |
| `breastcancer_to_pottr.py` | builds a matched cohort: POTTR DAG files and Multi-MASTRO transactions from one identical selection of trees |
| `mastro_timing_run.py` | times the Multi-MASTRO mining stages on a prepared input directory |
| `pottr_timing_run.py` | times POTTR by wrapping the functions `run_POTTR` calls, then invoking it unchanged |
| `rebuild_timing_csv.py` | rebuilds `timing.csv` from the per-run JSON records; idempotent |
| `pottr_force_significance.py` | applies POTTR's own permutation test to each POTTR trajectory |
| `pottr_significance.py` | re-scores POTTR trajectories under the multi-tree tests |

Every run writes `<outdir>/timing.csv`, `<outdir>/environment.txt` (machine,
CPU, RAM, solver version, date) and, per cell, a JSON record with per-stage wall
time, CPU time, and the CPU-to-wall ratio. `timing.csv` is rebuilt from those
records after every cell, so it is correct on disk even if the run is
interrupted; `rebuild_timing_csv.py` is also runnable by hand, which is how
records written by an older version of the harness get corrected without
repeating the runs.

### What is measured, and what is not

Frequent-trajectory extraction only, on both sides. No significance test:
POTTR's inline test is stubbed out, and the Multi-MASTRO side never calls the
significance stage. No theta post-filters either, since POTTR has no analogue.

Threads are not equalised, because they cannot be. Multi-MASTRO's mining is
serial by construction: LCM is compiled without `MULTI_CORE`, and the
downstream stages are serial Python. POTTR takes a thread count for Gurobi and
for building its conflict graph. Rather than assume this, every stage records
its CPU-to-wall ratio, so the degree of parallelism is a measurement.

Cohorts are named by their per-patient tree cap. `cap=1` keeps one tree per
patient and is a pure single-tree cohort; `cap=2` keeps at most two distinct
trees and is the smallest cohort with genuine multi-tree structure; `cap=0`
means no cap, the full 37809-tree cohort.

### The runs, in order

#### 1. Shakedown

Smallest possible configuration, to check that both sides start, that Gurobi
picks up a licence, and to get a first cost per ILP.

```
CAP_LIST=1 K_LIST="2 3" TIMEOUT=600 CORES=20 \
  bash scripts/run_timing_pottr_vs_mastro.sh
```

#### 2. Probe: shape of the cost against k

Five values of k over two cohorts, to find where POTTR becomes expensive before
committing the machine to a full sweep.

```
CAP_LIST="1 2" K_LIST="2 5 10 20 50" TIMEOUT=1800 CORES=20 \
  POTTR_THREADS=20 MAX_CONSEC_TIMEOUTS=0 \
  OUTDIR=results/timing_pottr_vs_mastro \
  bash scripts/run_timing_pottr_vs_mastro.sh
```

Note on reading its output: the cells it marks as timeouts are cells that
exceeded a 30-minute budget, not cells that cannot be solved. Experiment 3
completed all of them. Do not read a timeout here as infeasibility.

#### 3. The expensive band, with a real budget

The k values that exceeded the probe's budget, given four hours each.

```
CAP_LIST="1 2" K_LIST="5 10" SIGMA_LIST="2 5" TIMEOUT=14400 CORES=20 \
  POTTR_THREADS=20 MAX_CONSEC_TIMEOUTS=0 \
  OUTDIR=results/timing_hardband \
  bash scripts/run_timing_pottr_vs_mastro.sh
```

and the one cell the probe had left incomplete:

```
CAP_LIST=2 K_LIST=2 SIGMA_LIST=5 TIMEOUT=14400 CORES=20 POTTR_THREADS=20 \
  OUTDIR=results/timing_hardband \
  bash scripts/run_timing_pottr_vs_mastro.sh
```

#### 4. Replicates

Three independent repetitions of the cells that complete quickly, to separate
signal from run-to-run variation. A separate `OUTDIR` per replicate is required,
since the driver skips cells whose output already exists.

```
for REP in 1 2 3; do
  CAP_LIST="1 2" K_LIST="20 50" SIGMA_LIST="2 5" TIMEOUT=3600 CORES=20 \
    POTTR_THREADS=20 OUTDIR=results/timing_rep$REP \
    bash scripts/run_timing_pottr_vs_mastro.sh
done
```

Two things this is for: error bars, and checking whether repeated runs return
the same trajectory. Do not compare timings across separate invocations of the
driver; the between-session spread is much larger than the within-session one.

#### 5. Full cohort

The complete cohort, one value of k, with the ten-hour budget the POTTR paper
uses for its own scaling benchmark.

```
CAP_LIST=0 K_LIST=2 SIGMA_LIST="2 5" TIMEOUT=36000 CORES=20 POTTR_THREADS=20 \
  OUTDIR=results/timing_fullcohort \
  bash scripts/run_timing_pottr_vs_mastro.sh
```

This cohort needs 6.6e8 cross-patient tree pairs in POTTR's conflict graph,
which `run_POTTR` materialises in memory before solving anything. Expect the
conflict stage to dominate. `pottr_timing_run.py` samples the resident memory
of the whole process tree every 15 seconds and rewrites its record, so a run
killed from outside still leaves the stage it was in and how much it held.

#### 6. Solution pool

By default `run_POTTR.py` passes `--solution-pool-size 0`, so Gurobi returns a
single optimal solution; when several trajectories share the maximum size, which
one comes back is not fixed. The POTTR paper's own comparison against MASTRO
instead sets `PoolSearchMode=2` with `PoolSolutions=50000`. The two settings are
different experiments and their timings are not interchangeable.

```
CAP_LIST=1 K_LIST="20 50" SIGMA_LIST=5 TIMEOUT=7200 CORES=20 \
  POTTR_THREADS=20 POOL=50000 \
  OUTDIR=results/timing_pool50k \
  bash scripts/run_timing_pottr_vs_mastro.sh
```

To count how many trajectories came back, and their support in trees against
their support in distinct patients, rebuild the CSV: those are separate columns.
`converted_graphs.txt` holds the trajectories themselves, two lines each.

#### 7. Both significance tests on the same trajectories

Applies POTTR's own permutation test and the multi-tree tests to the same
trajectories, so the two can be compared hypothesis by hypothesis. Run it on a
cohort with multi-tree patients: on a one-tree-per-patient cohort the multi-tree
test degenerates to the single-tree one and the comparison says nothing.

```
D=results/<outdir>/cap2

python3 pottr_force_significance.py \
  --pottr_dir $D/pottr_t20 --k_range 20,50 \
  --dags $D/dags --pottr_repo ../POTTR \
  --max_nodes 6 --cores 20 --out $D/significance_forced.txt

python3 pottr_significance.py \
  --pottr_dir $D/pottr_t20 --k_range 20,50 \
  --pottr_sig_global $D/significance_forced.txt \
  --graphs_all $D/inputs/graphs_all.txt \
  -w $D/inputs/weights_uniform.txt --owner $D/inputs/owner.txt \
  --theta 1.0 --null perm --n_jobs 20 --seed 0 \
  --out $D/pottr_vs_ours.csv
```

`pottr_force_significance.py` is needed because `run_POTTR` runs its own test
only when the largest trajectory of a given k is small enough, which leaves
whole k values unscored, small trajectories included. It re-applies the gate per
trajectory and never touches POTTR's own output files.

### Knobs specific to the runtime comparison

| Variable | Meaning |
|---|---|
| `CAP_LIST` | per-patient tree caps to build cohorts for; `0` means no cap |
| `K_LIST` | POTTR recurrence levels |
| `SIGMA_LIST` | Multi-MASTRO support thresholds |
| `POTTR_THREADS` | thread settings to time POTTR at |
| `POOL` | Gurobi solution pool size passed to POTTR |
| `TIMEOUT` | wall-clock budget per invocation |
| `MAX_CONSEC_TIMEOUTS` | abandon a k-sweep after this many consecutive timeouts; `0` disables |
| `SKIP_POTTR`, `SKIP_MASTRO` | run only one half |
| `PY` | interpreter, when the one on PATH is not the one carrying the dependencies |
| `POTTR_REPO` | POTTR checkout, when the automatic search picks the wrong one |

### Practical notes

Runs resume: every cell whose output exists is skipped, so an interrupted run
restarts with the same command. Deleting a JSON forces that cell to be redone.

Do not run two timing experiments at once, and do not run one alongside other
work. These are measurements.

The per-invocation budget is not a tight bound. The solver runs inside C and
does not return control to the Python signal handler, so a run can exceed its
`TIMEOUT` by a wide margin; observed overshoots reached 40 percent. Cells marked
as timeouts are therefore comparable to each other only as "did not finish".

When collecting results, filter by file name rather than by extension: the
cohort DAG files are `.txt` and there are tens of thousands of them.

```
rsync -avz --prune-empty-dirs \
  --include '*/' --include '*.json' --include '*.log' \
  --include 'timing.csv' --include 'manifest.csv' \
  --include 'converted_graphs.txt' --include 'environment.txt' \
  --exclude '*' \
  <host>:<path>/results/<outdir> <local destination>/
```

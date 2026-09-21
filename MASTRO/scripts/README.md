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
with the other for cores. It rebuilds the matched cohort at several
per-patient tree caps and, for each one, times the Multi-MASTRO mining and
POTTR at a range of recurrence levels, with a wall-clock budget per invocation.
A timeout is recorded as a data point, including which stage it was in when it
was stopped.

The two sides are not given the same number of threads, because they cannot be:
Multi-MASTRO's mining is serial by construction, while POTTR takes a thread
count for Gurobi and for building the conflict graph. POTTR is therefore timed
at one thread and at many, and both are reported.

Writes `results/timing_pottr_vs_mastro/`: `timing.csv` with one row per
configuration and a per-stage breakdown, `environment.txt` recording the
machine, and a JSON record plus a solver log per run.

Knobs: `CAP_LIST` (trees per patient; `0` means no cap), `K_LIST`, `SIGMA_LIST`,
`POTTR_THREADS`, `TIMEOUT`, `MAX_CONSEC_TIMEOUTS`, `SKIP_POTTR`, `SKIP_MASTRO`.

## A note on how significance is computed

Each test is scored on **its own mined family**, the expected-support test on
the expected-support family and the theta-consensus test on the theta-maximal
post-filter, on the observed data and on every resample alike. The observed
and null distributions therefore refer to the same set of hypotheses, and
neither test borrows the other's null. Monte-Carlo p-values are add-one
smoothed, so none is ever exactly zero.

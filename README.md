# Multi-MASTRO: Discovering Significant Evolutionary Trajectories from Sets of Cancer Phylogenies

This repository contains the implementation of Multi-MASTRO, our algorithm to discover statistically significant conserved evolutionary trajectories of alterations when each patient is described by a *set* of candidate phylogenetic tumor trees rather than by a single tree. Reconstructing a tumor phylogeny is an inverse problem and the data often do not determine one tree; methods that take a single tree per patient must pick a representative, and that choice can manufacture apparent recurrence. Multi-MASTRO keeps the whole set: every candidate tree of a patient contributes, weighted, so a patient's evidence for a trajectory is a quantity in [0,1] rather than a yes/no, and no patient is ever counted more than once. All details on Multi-MASTRO are in the accompanying paper.

Multi-MASTRO builds on MASTRO (Pellegrina and Vandin, Bioinformatics 2022, https://doi.org/10.1093/bioinformatics/btac467), which solves the single-tree version of the problem. The original implementation is kept in this repository and still runs (`run_MASTRO.py`, `run_wy_correction.py`, `compute_significance.py`, `plot_results.py`); it is what the single-tree baseline of our experiments uses.

Please address questions and bug reports to Umberto Bianchin (umberto.bianchin@phd.unipd.it). For the original MASTRO, to Leonardo Pellegrina (leonardo.pellegrina@unipd.it) and Fabio Vandin (fabio.vandin@unipd.it).

Our implementation leverages LCM (version 5.3 by Takeaki Uno http://research.nii.ac.jp/~uno/codes.htm ) to mine frequent itemsets.
The folder `/MASTRO/` contains the source files for the implementation of Multi-MASTRO and for reproducing the experiments described in the paper. The folder `/MASTRO/scripts/` contains one self-contained script per experiment, documented in its own README. The folder `/data/` contains the data analyzed by Multi-MASTRO (and scripts to preprocess it).

Multi-MASTRO needs Python 3.10 or later (tested on 3.12) and the packages in `requirements.txt` (`pip install -r requirements.txt`), plus a C compiler and `perl`: `lcm53/lcm` is built automatically on first run, and LCM's `transnum.pl` maps edge labels to integer item ids.

To run Multi-MASTRO use the python script `run_pipeline.py`. Such script accepts various parameters, listed and described by using the `-h` flag:

```
usage: run_pipeline.py [-h] (--npy NPY | --graphs GRAPHS) [--owner OWNER]
                       [--sigma SIGMA] [--seed SEED] [--theta_list THETA_LIST]
                       [--min_mine_sigma MIN_MINE_SIGMA] [--lcmdir LCMDIR]
                       [--outdir OUTDIR] [--keep_gl] [--significance]
                       [--single_tree_only] [--sig_null {indep,perm}]
                       [--sig_mc_cutoff SIG_MC_CUTOFF]
                       [--sig_mc_samples SIG_MC_SAMPLES]
                       [--sig_n_jobs SIG_N_JOBS]
```

A typical invocation on the multi-tree breast cancer cohort is:

```
python3 run_pipeline.py --npy ../data/breastCancer.npy --sigma 5 --theta_list 0.5,1.0 --significance --outdir results/demo
```

The `--npy` option specifies a dataset given as an object array indexed `data[patient][tree][edge] = [parent, child]`; alternatively `--graphs` takes a file of pre-computed transactions (see below for the input format), in which case `--owner` gives the patient owning each line. Without `--owner` every line is treated as a separate patient, which is the single-tree setting.

The parameter `--sigma` sets the minimum support of the trajectories to find. Because a patient's trees are weighted by `1/M_i`, support is measured in *expected patients* and the threshold may be fractional.

The parameter `--theta_list` sets the consensus thresholds. A trajectory is theta-frequent for a patient when it appears in at least a fraction `theta` of that patient's trees.

One run mines four families of trajectories. Output files carry a short prefix, kept stable so that results already on disk stay readable:

1. `alg0_` is the single-tree baseline: one tree per patient is sampled at random (controlled by `--seed`) and mined without weights.
2. `alg1_` is the expected-support family: all candidate trees are pooled, the trees of patient `i` weighted `1/M_i`, and mined with weighted frequent-itemset mining.
3. `alg2_` is the theta-frequent family: patterns whose per-patient weighted presence reaches `theta` in at least `sigma` patients.
4. `alg3_` is the theta-maximal family: the theta-frequent family reduced to its maximal elements.

The flag `--sig_null` specifies the null model used for evaluating the significance of a frequent trajectory:
`perm` draws a single random permutation of the alteration set of patient `i` and applies it **coherently to all of that patient's candidate trees**. Placements are therefore injective, and the relabeling is shared within a patient, which is what couples the trees of one patient and makes the patient-level null distribution a genuine lattice-valued quantity rather than a Bernoulli.
`indep` places each alteration of the trajectory independently and uniformly at random on the alteration set, matching the baseline denominator of the original MASTRO.
`perm` is the default and is the more conservative of the two: its placements cannot fail for a reason unrelated to ordering, whereas `indep` also draws non-injective placements that fail automatically. The third null model of the original MASTRO, drawing a random topology from the cohort, is not implemented for the multi-tree setting.

Two support notions are tested, each on its own mined family so that observed and null distributions refer to the same set of hypotheses. The *expected support* `s_exp(P)` is the expected number of patients carrying `P` if one tree were drawn per patient, and it aggregates a patient's duplicate trees back to that patient. The *theta-consensus support* `s_theta(P)` is the number of patients for which `P` appears in at least a fraction `theta` of their trees. The expected-support test is applied to the `alg0_` and `alg1_` families, the theta-consensus test to the `alg2_` and `alg3_` families.

The per-patient distributions are computed exactly when a patient has at most `--sig_mc_cutoff` candidate trees, and by Monte Carlo with `--sig_mc_samples` draws above that, since exact enumeration grows exponentially in the number of trees. Monte-Carlo p-values use the add-one estimator `(B*p + 1)/(B + 1)`, so none is ever exactly zero. Note that no estimated p-value can then fall below `1/(B+1)`: if the corrected significance threshold lands near that floor, the number of rejections is decided by the granularity of the estimate rather than by the data. Check where the threshold sits relative to `1/(B+1)` before reading anything into a count, and raise `--sig_mc_samples` if it is close.

To correct for multiple hypothesis testing, Multi-MASTRO uses resampling-based procedures: the Westfall-Young permutation testing procedure to bound the Family-Wise Error Rate (FWER) and an empirical estimate of the False Discovery Rate (FDR).
The script `run_wy_correction_ensemble.py` generates resampled datasets, re-mines each test's family on every resample, and writes the corrected thresholds and the empirical FDR.
The script `plot_fdr_v2.py` generates the FDR(k) curves. Reported `max k` values follow the prefix convention, the largest `k` before the curve first crosses the level; the curve is not monotone, so taking the global maximum instead gives different numbers.
The script `empirical_fwer_ensemble.py` measures the empirical FWER under the global null, to verify that the correction achieves its nominal level.

### Input format
Multi-MASTRO takes in input a file containing, in each line, the edges of the complete expanded tumor graph of the tumor phylogenetic trees, separated by a space.
Alterations can be of any type (e.g., SNV or CNA) but must have a distict id in each tree (e.g., `TP53_SNV` and `TP53_DEL` are allowed in the same tumor tree).
The complete expanded tumor graph is composed by three types of edges:
1. a directed edge `A->-B` denotes that the alteration `A` is an anchestor of the alteration `B`
2. an undirected edge `A-/-B` denotes that `A` and `B` belong to different branches of the tree
3. an undirected edge `A-?-B` denotes that the order between `A` and `B` is not known (they belong to the same node)

Note that the last two edges are undirected, therefore the alterations are meant to be sorted alphabetically (i.e., `A-?-B` is correct, `B-?-A` is not).
(it is not necessary to include the edges incident to the germline node to find trajetories composed by at least 2 alterations; otherwise, use `g` to denote the germline node)

For example, the following file denotes the input of three tumor trees with edges `{g->-A, A->-B, B->-C}`, `{g->-A, A->-B, A->-C}`, `{g->-A, g->-B, A->-C}`:
```
A->-B A->-C B->-C
A->-B A->-C B-/-C
A-/-B A->-C B-/-C
```

When a patient contributes several candidate trees, its trees occupy several consecutive lines and two further files, aligned line by line with the graphs file, describe the grouping:
`owner.txt` gives the 0-based index of the patient owning each line, and `weights_uniform.txt` gives the weight of each line, `1/M_i` for a patient with `M_i` trees. Both are written automatically in `--npy` mode.

### Output format
The output of Multi-MASTRO is written under the directory given by `--outdir`:
```
inputs/          graphs_all.txt, owner.txt, weights_uniform.txt, graphs_sampled.txt
runs/            per-family mining output, including the filtered families
significance/    per-trajectory p-values, one CSV per family
analysis/        pairwise Jaccard similarity and per-family unique patterns
```
The mined families are lists of frequent trajectories, represented with the same format as the input tumor trees. Each trajectory line is followed by a line listing the transactions in which it occurs.
The significance CSVs give, for each trajectory, its number of nodes and edges, its expected support `s_exp`, its theta-consensus support `s_theta`, and the p-value of whichever test applies to that family.
`run_wy_correction_ensemble.py` writes its own directory containing `wy_thresholds.txt` (corrected thresholds at alpha = 0.01, 0.05 and 0.10), `min_pvalues.csv` (one resample minimum per row) and `fdr_exp.csv` / `fdr_theta.csv` (the observed p-values in rank order with the empirical FDR).

### Reproducing the experiments
`MASTRO/scripts/` holds one self-contained script per experiment: calibration of the error control, recovery power on implanted trajectories, discovery on the two real cohorts, the single-tree seed sweep, the mining-overhead measurement, and the comparison against POTTR. Each writes under `results/`, resumes if interrupted, and takes its knobs from environment variables; see `MASTRO/scripts/README.md`.
As a check that the toolchain works end to end, on `data/breastCancer.npy` the expected-support family contains 2354 trajectories at `sigma = 2` and 372 at `sigma = 5`.

### License
MIT, see `LICENSE`. LCM v5.3 is redistributed under its own terms.

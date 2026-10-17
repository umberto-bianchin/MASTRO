"""
Ablations on a real cohort: calibration (empirical FWER, Q-Q) and discoveries
of Multi-MASTRO with one design choice broken at a time.

Variants (same as ablation_toys.py):
  maestri            weights 1/M_i, owner = patient, null 'perm'
  perm_tree          weights 1/M_i, owner = patient, null 'perm_tree'
  count_trees        weights 1,     owner = patient, null 'perm'
  count_perm_tree    weights 1,     owner = patient, null 'perm_tree'
  trees_as_patients  weights 1,     owner = tree,    null 'perm'

Every variant scores the SAME family, the complete trajectories with
Multi-MASTRO weighted support >= sigma: only the significance test changes.

Ground truth. The data-generating null is the per-patient relabelling ('perm'
generator): every candidate tree of a patient is a reconstruction of the same
mutations, so relabelling the mutations relabels all its trees together. A
variant whose own null differs from it calibrates its Westfall-Young threshold
on its OWN resamples (that is what a user of the variant would do), and its
FWER is then measured on datasets drawn from the per-patient null. WY with the
data-generating null would be calibrated by construction for every variant.

Pools of datasets, each mined once and scored by every variant:
  real      the observed cohort (discoveries)
  perm_cal  n_cal datasets, per-patient permutation (WY of the 'perm' variants)
  tree_cal  n_cal datasets, one permutation per tree (WY of the per-tree
            variants; for trees_as_patients a per-patient permutation IS a
            per-tree one)
  perm_val  n_val datasets, per-patient permutation (FWER and Q-Q)

The family mined on a null dataset is selected for high support, so its
p-values are not uniform even under a valid test. The Q-Q plot therefore also
scores, on every perm_val dataset, the FIXED family mined on the real cohort
(supports recounted on the null dataset): no selection, so a valid test stays
on or below the diagonal.

Outputs in --outdir:
  pvals/<pool>/<r>.csv.gz   pattern + one p-value column per variant
  pvals_fixed/perm_val/<r>.csv.gz   same, real-data family (Q-Q)
  minima.csv                per-dataset minimum p-value per variant
  real_pvalues.csv          p-values of the real-data family (plain CSV, so
                            that git carries it)
  fwer.csv                  WY threshold, empirical FWER (mean +- sd over
                            bootstrap calibration/validation splits) and
                            real-data discoveries per variant and alpha
  discoveries.csv           the real-data discoveries per variant and alpha

Usage:
    python3 ablation_calibration.py --graphs_all INP/graphs_all.txt \\
        -w INP/weights_uniform.txt --owner INP/owner.txt --sigma 5 \\
        --n_cal 500 --n_val 500 --par 50 --outdir results/ablation/breast_sigma5
    # recompute fwer.csv / discoveries.csv from the saved p-values:
    python3 ablation_calibration.py ... --analyze_only
"""

import argparse
import csv
import gzip
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from utils import PYTHON, SCRIPT_DIR, ensure_dir, load_weights_and_owner, run_cmd, set_verbose
from run_wy_correction_ensemble import mine_family, permute_transactions

VARIANTS = {
    # name: (weights, owner, significance null, WY calibration pool)
    "maestri": ("patient", "patient", "perm", "perm_cal"),
    "perm_tree": ("patient", "patient", "perm_tree", "tree_cal"),
    "count_trees": ("tree", "patient", "perm", "perm_cal"),
    "count_perm_tree": ("tree", "patient", "perm_tree", "tree_cal"),
    "trees_as_patients": ("tree", "tree", "perm", "tree_cal"),
}
POOL_GEN = {"real": None, "perm_cal": "perm", "tree_cal": "perm_tree", "perm_val": "perm"}
POOL_ID = {"real": 0, "perm_cal": 1, "tree_cal": 2, "perm_val": 3}


def variant_files(inputs, weights_path, owner_path, K):
    """weights/owner file pair of every variant, written once into *inputs*."""
    ones = inputs / "weights_ones.txt"
    tree_owner = inputs / "owner_tree.txt"
    if not ones.exists():
        ones.write_text("1.0\n" * K)
    if not tree_owner.exists():
        tree_owner.write_text("".join(f"{t}\n" for t in range(K)))
    w = {"patient": weights_path, "tree": ones}
    o = {"patient": owner_path, "tree": tree_owner}
    return {v: (w[wm], o[om], null) for v, (wm, om, null, _) in VARIANTS.items()}


def read_pvals(path):
    """{pattern: p} of one significance CSV."""
    with path.open() as f:
        return {r["pattern"]: float(r["pval_exp"]) for r in csv.DictReader(f)
                if r.get("pval_exp")}


def _run_dataset(args):
    (pool, r, graphs_path, weights_path, owner_path, owner0, n_patients,
     files, variants, sigma, lcmdir, workdir, outdir, mc_cutoff, mc_samples,
     seed, keep, verbose, fixed, n_jobs) = args
    set_verbose(verbose)
    out_csv = outdir / "pvals" / pool / f"{r}.csv.gz"
    fixed_csv = outdir / "pvals_fixed" / pool / f"{r}.csv.gz"
    if out_csv.exists() and (not fixed or fixed_csv.exists()):
        return pool, r, _minima(out_csv, variants)

    rng = np.random.default_rng(seed + 1_000_003 * POOL_ID[pool] + r)
    rdir = workdir / f"{pool}_{r}"
    ensure_dir(rdir)
    try:
        lines = [l.rstrip("\n") for l in open(graphs_path)]
        gen = POOL_GEN[pool]
        if gen is not None:
            lines = permute_transactions(lines, owner0, n_patients, rng,
                                         per_tree=(gen == "perm_tree"))
        data = rdir / "graphs.txt"
        data.write_text("".join(l + "\n" for l in lines))

        def score(family_file, tag):
            table = {}
            if family_file is None:
                return table
            for v in variants:
                w, o, null = files[v]
                sig = rdir / f"sig_{tag}_{v}.csv"
                run_cmd([PYTHON, str(SCRIPT_DIR / "compute_significance_ensemble.py"),
                         "-i", str(family_file), "-o", str(sig),
                         "-w", str(w), "--owner", str(o), "--graphs_all", str(data),
                         "--test", "exp", "--null", null,
                         "--mc_cutoff", str(mc_cutoff), "--mc_samples", str(mc_samples),
                         "--seed", str(seed + r), "--n_jobs", str(n_jobs)])
                for pat, p in read_pvals(sig).items():
                    table.setdefault(pat, {})[v] = p
            return table

        write_table(out_csv, score(mine_family(data, rdir, lcmdir, sigma, weights_path),
                                   "mined"), variants)
        if fixed:
            fam = rdir / "fixed_family.txt"
            write_family(fam, fixed, lines)
            write_table(fixed_csv, score(fam, "fixed"), variants)
        return pool, r, _minima(out_csv, variants)
    finally:
        if not keep:
            shutil.rmtree(rdir, ignore_errors=True)


def write_table(path, table, variants):
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pattern"] + variants)
        for pat, ps in table.items():
            wr.writerow([pat] + [f"{ps.get(v, 1.0):.6e}" for v in variants])
    tmp.rename(path)


def write_family(path, patterns, lines):
    """Result file (pattern / occurrence lines) of fixed patterns on *lines*."""
    index = {}
    for t, line in enumerate(lines):
        for item in line.split():
            index.setdefault(item, set()).add(t)
    with path.open("w") as f:
        for pat in patterns:
            occ = None
            for item in pat.split():
                ids = index.get(item, set())
                occ = set(ids) if occ is None else occ & ids
            occ = sorted(occ or ())
            f.write(f"{pat} ({len(occ)})\n{' '.join(map(str, occ))}\n")


def real_family(outdir):
    with gzip.open(outdir / "pvals" / "real" / "0.csv.gz", "rt") as f:
        return [row["pattern"] for row in csv.DictReader(f)]


def _minima(path, variants):
    m = {v: 1.0 for v in variants}
    with gzip.open(path, "rt") as f:
        for row in csv.DictReader(f):
            for v in variants:
                m[v] = min(m[v], float(row[v]))
    return m


def wy_threshold(minima, alpha):
    """alpha-quantile of the calibration minima, as in empirical_fwer_ensemble."""
    s = np.sort(np.asarray(minima))
    return s[max(0, int(np.floor(alpha * len(s))) - 1)]


def analyze(outdir, variants, alphas, n_trials, m_cal, seed):
    """Empirical FWER over bootstrap calibration/validation splits.

    In every trial a variant's WY threshold is the alpha-quantile of m_cal
    minima drawn from its own calibration pool, and the FWER is the fraction of
    validation datasets (per-patient null) whose minimum falls below it. For
    the 'perm' variants calibration and validation come from the same null, so
    the pool perm_cal + perm_val is split at random in every trial, as in
    empirical_fwer_ensemble.py; a single split leaves a quantile error of about
    sqrt(alpha/m) on the FWER. The discovery threshold uses the whole own pool.
    """
    mins = {}
    with (outdir / "minima.csv").open() as f:
        for row in csv.DictReader(f):
            mins.setdefault(row["pool"], []).append({v: float(row[v]) for v in variants})
    real_gz = outdir / "pvals" / "real" / "0.csv.gz"
    real_csv = outdir / "real_pvalues.csv"
    if real_gz.exists():
        with gzip.open(real_gz, "rt") as f:
            real = list(csv.DictReader(f))
        with real_csv.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["pattern"] + variants)
            wr.writeheader()
            wr.writerows(real)
    elif real_csv.exists():
        real = list(csv.DictReader(real_csv.open()))
    else:
        real = None
        print("[ABL] no real-data p-values here: FWER only, discoveries left blank")

    rng = np.random.default_rng(seed + 2)
    rows, disc = [], []
    for v in variants:
        own = VARIANTS[v][3]
        perm_all = np.array([d[v] for d in mins.get("perm_cal", []) + mins.get("perm_val", [])])
        own_all = perm_all if own == "perm_cal" else np.array([d[v] for d in mins[own]])
        n_val = len(perm_all) - m_cal if own == "perm_cal" else len(perm_all)
        for a in alphas:
            fw = []
            for _ in range(n_trials):
                if own == "perm_cal":
                    idx = rng.permutation(len(perm_all))
                    cal, val = perm_all[idx[:m_cal]], perm_all[idx[m_cal:]]
                else:
                    cal = own_all[rng.choice(len(own_all), size=m_cal, replace=False)]
                    val = perm_all
                fw.append(float(np.mean(val <= wy_threshold(cal, a))))
            d = wy_threshold(own_all, a)
            hits = [row["pattern"] for row in real or [] if float(row[v]) <= d]
            rows.append({"variant": v, "alpha": a, "wy_pool": own,
                         "threshold": f"{d:.6e}",
                         "fwer_mean": f"{np.mean(fw):.4f}", "fwer_sd": f"{np.std(fw):.4f}",
                         "m_cal": m_cal, "n_val": n_val, "n_trials": n_trials,
                         "n_family_real": "" if real is None else len(real),
                         "n_discoveries": "" if real is None else len(hits)})
            disc += [{"variant": v, "alpha": a, "pattern": h} for h in hits]
            print(f"[ABL] {v:18s} alpha={a:.2f} delta={d:.3e} (WY on {own})  "
                  f"FWER={np.mean(fw):.4f} +- {np.std(fw):.4f}  "
                  + ("" if real is None else f"discoveries={len(hits)}/{len(real)}"))

    with (outdir / "fwer.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    if real is None:
        print(f"[ABL] written -> {outdir / 'fwer.csv'}")
        return
    with (outdir / "discoveries.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["variant", "alpha", "pattern"])
        wr.writeheader()
        wr.writerows(disc)
    print(f"[ABL] written -> {outdir / 'fwer.csv'}, {outdir / 'discoveries.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs_all", required=True)
    ap.add_argument("-w", "--weights", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n_cal", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--alphas", default="0.01,0.05,0.1")
    ap.add_argument("--mc_cutoff", type=int, default=8)
    ap.add_argument("--mc_samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lcmdir", default="./lcm53")
    ap.add_argument("--par", type=int, default=1)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--n_trials", type=int, default=1000,
                    help="Bootstrap calibration/validation splits for the FWER")
    ap.add_argument("--m_cal", type=int, default=None,
                    help="Calibration minima per split (default: n_cal // 2)")
    args = ap.parse_args()

    set_verbose(args.verbose)
    variants = [v for v in args.variants.split(",") if v]
    alphas = [float(a) for a in args.alphas.split(",")]
    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)

    if not args.analyze_only:
        graphs_path = Path(args.graphs_all).resolve()
        weights_path = Path(args.weights).resolve()
        owner_path = Path(args.owner).resolve()
        _, owner0, n_patients, K = load_weights_and_owner(weights_path, owner_path)
        inputs = outdir / "inputs"
        ensure_dir(inputs)
        files = variant_files(inputs, weights_path, owner_path, K)
        workdir = outdir / "work"
        ensure_dir(workdir)

        def task(pool, r, fixed=None, n_jobs=1):
            return (pool, r, graphs_path, weights_path, owner_path, owner0, n_patients,
                    files, variants, args.sigma, Path(args.lcmdir).resolve(), workdir,
                    outdir, args.mc_cutoff, args.mc_samples, args.seed, args.keep,
                    args.verbose, fixed, n_jobs)

        # The real cohort first: its family is the fixed family of the Q-Q plot.
        # It runs alone, so its significance step uses every core (parallel
        # over trajectories) instead of one
        print("[ABL] real cohort first (significance on all cores)", flush=True)
        results = [_run_dataset(task("real", 0, n_jobs=args.par))]
        fixed = real_family(outdir)
        pools = [("perm_cal", args.n_cal), ("perm_val", args.n_val)]
        if any(VARIANTS[v][3] == "tree_cal" for v in variants):
            pools.append(("tree_cal", args.n_cal))
        tasks = [task(pool, r, fixed if pool == "perm_val" else None)
                 for pool, n in pools for r in range(n)]
        print(f"[ABL] n_patients={n_patients} trees={K} sigma={args.sigma} "
              f"variants={variants} real family={len(fixed)} "
              f"null datasets={len(tasks)} par={args.par}", flush=True)

        step = max(1, len(tasks) // 100)
        with Pool(args.par) as pool:
            for done, res in enumerate(pool.imap_unordered(_run_dataset, tasks), 1):
                results.append(res)
                if done % step == 0 or done == len(tasks):
                    print(f"[ABL] dataset {done}/{len(tasks)} done", flush=True)

        with (outdir / "minima.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["pool", "r"] + variants)
            for pool_name, r, m in sorted(results, key=lambda x: (x[0], x[1])):
                wr.writerow([pool_name, r] + [f"{m[v]:.6e}" for v in variants])

    analyze(outdir, variants, alphas, args.n_trials,
            args.m_cal or args.n_cal // 2, args.seed)


if __name__ == "__main__":
    main()

"""Calibration: empirical FWER control for the ensemble tests.

Under the global null (no trajectory is genuinely conserved) every reported
trajectory is a false discovery, so a correctly calibrated Westfall-Young
threshold must yield an empirical FWER of at most alpha. This script measures
that empirical rate for both ensemble tests.

Protocol:
  1. Draw a pool of N null datasets with the coherent per-patient null ('perm'
     or 'indep'), mine each, and score both tests on their own family (expected
     support on the maximal family, theta-consensus on the theta-maximal
     family), the same pipeline used by
     run_wy_correction_ensemble._run_resample. Record the minimum p-value of
     each null dataset, for each test.
  2. Bootstrap R trials. In every trial the pool is split into a calibration set
     of size m (used to fix the WY threshold delta_hat at each alpha) and a
     disjoint validation set (used to measure the empirical FWER as the fraction
     of validation datasets whose minimum p-value is <= delta_hat). Reusing one
     expensive pool of minings across many trials gives an independent
     calibration/validation split without paying N*R minings.

Output: empirical_fwer.csv with (test, alpha, m, mean_fwer, std_fwer) and a
console summary. A well-calibrated procedure gives mean_fwer <= alpha.

Usage
-----
    python3 empirical_fwer_ensemble.py \
        --graphs_all inputs/graphs_all.txt \
        -w inputs/weights_uniform.txt --owner inputs/owner.txt \
        --sigma 2.0 --theta 1.0 --null perm \
        --n_datasets 200 --m 100 --n_trials 200 \
        --par 64 --outdir test_cal/breastCancer_sigma2_theta1.0
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from utils import ensure_dir, load_weights_and_owner, set_verbose
from run_wy_correction_ensemble import _run_resample


def _min_or_one(vals):
    return min(vals) if vals else 1.0


def wy_threshold(sorted_minima, alpha, m):
    """WY-corrected threshold: the alpha-quantile of the calibration minima."""
    idx = max(0, int(np.floor(alpha * m)) - 1)
    return sorted_minima[idx]


def main():
    ap = argparse.ArgumentParser(
        description="Empirical FWER calibration for the ensemble tests"
    )
    ap.add_argument("--graphs_all", required=True)
    ap.add_argument("-w", "--weights", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--null", choices=["perm", "indep"], default="perm")
    ap.add_argument("--test", choices=["exp", "theta", "both"], default="both")
    ap.add_argument("--n_datasets", type=int, default=200,
                    help="Size N of the null-dataset pool")
    ap.add_argument("--m", type=int, default=100,
                    help="Calibration-set size m (< n_datasets)")
    ap.add_argument("--n_trials", type=int, default=200,
                    help="Bootstrap trials R over calibration/validation splits")
    ap.add_argument("--alphas", default="0.01,0.05,0.1")
    ap.add_argument("--mc_cutoff", type=int, default=8)
    ap.add_argument("--mc_samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--lcmdir", default="./lcm53")
    ap.add_argument("--par", type=int, default=1)
    ap.add_argument("--keep", action="store_true",
                    help="Keep resample working dirs (default: clean up)")
    ap.add_argument("--verbose", action="store_true",
                    help="Echo every per-dataset subcommand (default: quiet)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    if args.m >= args.n_datasets:
        ap.error("--m must be smaller than --n_datasets")

    set_verbose(args.verbose)

    lcmdir = Path(args.lcmdir).resolve()
    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)
    workdir = outdir / "resamples"
    ensure_dir(workdir)

    graphs_path = Path(args.graphs_all).resolve()
    weights_path = Path(args.weights).resolve()
    owner_path = Path(args.owner).resolve()

    graphs_lines = [l.rstrip("\n") for l in open(graphs_path)]
    _, _owner0, n_patients, K = load_weights_and_owner(weights_path, owner_path)
    assert len(graphs_lines) == K, (
        f"graphs has {len(graphs_lines)} lines but weights/owner has {K}")

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    print(f"[CAL] graphs={graphs_path.name} n_patients={n_patients} K={K}")
    print(f"[CAL] null={args.null} sigma={args.sigma} theta={args.theta}")
    print(f"[CAL] pool N={args.n_datasets} m={args.m} trials R={args.n_trials} "
          f"par={args.par}")

    # Step 1: build the null-dataset pool and record per-dataset minima
    worker_args = []
    for r in range(args.n_datasets):
        worker_args.append((
            r, graphs_path, weights_path, owner_path,
            args.sigma, lcmdir, workdir,
            args.test, args.null, args.theta,
            args.mc_cutoff, args.mc_samples,
            args.seed, not args.keep, args.verbose,
        ))

    total = args.n_datasets
    step = max(1, total // 100)
    results = []
    if args.par > 1:
        with Pool(args.par) as pool:
            for done, res in enumerate(
                    pool.imap_unordered(_run_resample, worker_args), 1):
                results.append(res)
                if done % step == 0 or done == total:
                    print(f"[CAL] null dataset {done}/{total} done", flush=True)
    else:
        for done, wa in enumerate(worker_args, 1):
            results.append(_run_resample(wa))
            if done % step == 0 or done == total:
                print(f"[CAL] null dataset {done}/{total} done", flush=True)
    results.sort(key=lambda x: x[0])

    min_exp = np.array([_min_or_one(pv_exp) for _, pv_exp, _ in results])
    min_theta = np.array([_min_or_one(pv_theta) for _, _, pv_theta in results])

    # Persist the raw pool so the bootstrap can be rerun/replotted offline.
    with (outdir / "pool_min_pvalues.csv").open("w") as f:
        f.write("dataset,min_pval_exp,min_pval_theta\n")
        for i in range(args.n_datasets):
            f.write(f"{i},{min_exp[i]:.6e},{min_theta[i]:.6e}\n")

    # Step 2: bootstrap calibration/validation splits
    rng = np.random.default_rng(args.seed + 1)
    tests = []
    if args.test in ("exp", "both"):
        tests.append(("exp", min_exp))
    if args.test in ("theta", "both"):
        tests.append(("theta", min_theta))

    rows = []
    N, m = args.n_datasets, args.m
    for tname, minima in tests:
        for alpha in alphas:
            fwers = []
            for _ in range(args.n_trials):
                perm = rng.permutation(N)
                cal = np.sort(minima[perm[:m]])
                val = minima[perm[m:]]
                delta = wy_threshold(cal, alpha, m)
                fwers.append(float(np.mean(val <= delta)))
            fwers = np.array(fwers)
            rows.append({
                "test": tname, "alpha": f"{alpha:.2f}", "m": m,
                "mean_fwer": f"{fwers.mean():.4f}",
                "std_fwer": f"{fwers.std():.4f}",
            })
            print(f"[CAL] {tname:5s} alpha={alpha:.2f}  "
                  f"empirical FWER = {fwers.mean():.4f} +/- {fwers.std():.4f}"
                  f"   ({'OK' if fwers.mean() <= alpha + fwers.std() else 'CHECK'})")

    out_csv = outdir / "empirical_fwer.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test", "alpha", "m", "mean_fwer", "std_fwer"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CAL] written -> {out_csv}")


if __name__ == "__main__":
    main()

"""Power: recovery of an implanted trajectory (ensemble version).

A ground-truth experiment designed to expose the difference between the two
tests. A known linear trajectory P = X1 -> X2 -> ... -> Xk over fresh
alterations is implanted into N randomly chosen "carrier" patients. In each
carrier we implant P perfectly in a fraction f of that patient's M trees (as a
real ancestor chain) and randomly in the remaining (1 - f) trees (the same
alterations placed so that P is not observed). Non-carrier patients never see
the X alterations.

With uniform weights w = 1/M this gives, by construction:
    * expected support    s^exp(P)   is about N * f
    * theta-consensus      s^theta(P) = N if f >= theta, else 0
so the expected-support test should recover P once N*f is large enough, while
the theta-consensus test should recover P only when the per-patient consistency
f reaches theta. This directly illustrates the incomparability of the two
support notions.

For each (N, f) we measure the recall, the fraction of trials in which P is
reported significant at a WY-corrected FWER threshold delta_hat(alpha). The
threshold is calibrated once, on a background-only null cohort of the same
shape, and reused across trials (it depends on the null, not the implant).

Output: recall.csv with (N, f, test, recall).
"""

import argparse
import csv
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from utils import (
    SCRIPT_DIR, ensure_dir, build_inputs, load_weights_and_owner,
    read_result_pairs, parse_items_from_pattern_line, run_cmd, set_verbose,
)
from run_pipeline import run_pipeline
from run_wy_correction_ensemble import _run_resample
from empirical_fwer_ensemble import wy_threshold


# =====================================================================
# Cohort synthesis
# =====================================================================
def _chain_edges(root, seq):
    """Edges of a linear chain root -> seq[0] -> seq[1] -> ..."""
    edges = []
    prev = root
    for node in seq:
        edges.append([prev, node])
        prev = node
    return edges


def _background_tree(rng, bg_labels, min_n=2, max_n=4):
    """Random linear background tree over a random subset of bg_labels."""
    n = rng.integers(min_n, max_n + 1)
    n = min(n, len(bg_labels))
    chosen = list(rng.choice(bg_labels, size=n, replace=False))
    return np.array(_chain_edges("GL", chosen), dtype=object)


def _perfect_tree(rng, traj, bg_labels):
    """Carrier tree containing P as a real ancestor chain, plus background."""
    seq = list(traj)
    n_bg = int(rng.integers(0, 3))
    if n_bg:
        seq = seq + list(rng.choice(bg_labels, size=min(n_bg, len(bg_labels)),
                                    replace=False))
    return np.array(_chain_edges("GL", seq), dtype=object)


def _noisy_tree(rng, traj, bg_labels):
    """Carrier tree in which P's alterations are present but placed in a random
    order (a random-insertion null for the trajectory).

    The k trajectory alterations are laid out as a random chain. P (which needs
    all C(k,2) forward pairs) is realised only when that random order happens to
    be the conserved one, an event of probability 1/k!. This is deliberate: the
    resulting mild, controlled contamination makes the per-patient presence a
    random variable slightly above f, which turns the recovery curves into
    smooth gradients (and keeps the null non-degenerate) rather than sharp steps.
    A moderate k (we use k = 4, so the accidental rate is 1/24) keeps this noise
    small enough not to saturate the low-theta regime."""
    perm = list(traj)
    rng.shuffle(perm)  # equals the conserved order (and thus realises P) w.p. 1/k!
    n_bg = int(rng.integers(1, 3))
    bg = list(rng.choice(bg_labels, size=min(n_bg, len(bg_labels)), replace=False))
    return np.array(_chain_edges("GL", perm + bg), dtype=object)


def build_cohort(rng, n_patients, M, traj, bg_labels, carriers, f):
    """Build a patients->trees object array with P implanted in `carriers`."""
    carriers = set(int(c) for c in carriers)
    data = np.empty(n_patients, dtype=object)
    for i in range(n_patients):
        trees = []
        if i in carriers:
            n_perfect = int(np.ceil(f * M))
            for j in range(M):
                if j < n_perfect:
                    trees.append(_perfect_tree(rng, traj, bg_labels))
                else:
                    trees.append(_noisy_tree(rng, traj, bg_labels))
        else:
            for _ in range(M):
                trees.append(_background_tree(rng, bg_labels))
        data[i] = trees
    return data


def traj_items(traj):
    """Ancestor items X_a->-X_b (a before b) that define the implanted P."""
    items = set()
    for a in range(len(traj)):
        for b in range(a + 1, len(traj)):
            items.add(f"{traj[a]}->-{traj[b]}")
    return items


# =====================================================================
# Mining + significance for one implanted dataset
# =====================================================================
def _recovered(sig_csv, p_items, col, delta):
    """True if some tested trajectory contains P and has p-value <= delta."""
    if not sig_csv.exists():
        return False
    with sig_csv.open() as f:
        for row in csv.DictReader(f):
            v = row.get(col, "")
            if not v:
                continue
            items = set(row["pattern"].split())
            if p_items.issubset(items) and float(v) <= delta:
                return True
    return False


def run_trial(trial_dir, data, traj, sigma, theta, lcmdir,
              delta_exp, delta_theta, null_model, mc_cutoff, mc_samples,
              seed, want_exp, want_theta):
    """Build inputs, mine, score, and report recovery flags for one dataset."""
    ensure_dir(trial_dir)
    inputs_dir = trial_dir / "inputs"
    ensure_dir(inputs_dir)
    graphs_all, weights_uniform, owner_txt, _ = build_inputs(
        list(data), inputs_dir, seed=seed)

    # Alg 1: expected-support maximal family
    alg1 = run_pipeline(graphs_txt=graphs_all, sigma=float(sigma), lcmdir=lcmdir,
                        workdir=trial_dir / "alg1", tag="alg1",
                        weights_txt=weights_uniform, weighted=True)

    p_items = traj_items(traj)
    rec_exp = rec_theta = False

    if want_exp:
        sig_exp = trial_dir / "pvalues_exp.csv"
        run_cmd([
            "python3", str(SCRIPT_DIR / "compute_significance_ensemble.py"),
            "-i", str(alg1), "-o", str(sig_exp),
            "-w", str(weights_uniform), "--owner", str(owner_txt),
            "--graphs_all", str(graphs_all),
            "--test", "exp", "--null", null_model,
            "--mc_cutoff", str(mc_cutoff), "--mc_samples", str(mc_samples),
            "--seed", str(seed),
        ])
        rec_exp = _recovered(sig_exp, p_items, "pval_exp", delta_exp)

    if want_theta:
        # Alg 3: theta-maximal family
        alg3 = trial_dir / "alg3.txt"
        run_cmd([
            "python3", str(SCRIPT_DIR / "postfilter_theta.py"),
            "-i", str(alg1), "-o", str(alg3),
            "-w", str(weights_uniform), "-owner", str(owner_txt),
            "-theta", str(theta), "-st", str(int(round(sigma))), "--maximal",
        ])
        if alg3.exists() and alg3.stat().st_size > 0:
            sig_theta = trial_dir / "pvalues_theta.csv"
            run_cmd([
                "python3", str(SCRIPT_DIR / "compute_significance_ensemble.py"),
                "-i", str(alg3), "-o", str(sig_theta),
                "-w", str(weights_uniform), "--owner", str(owner_txt),
                "--graphs_all", str(graphs_all),
                "--test", "theta", "--theta", str(theta), "--null", null_model,
                "--mc_cutoff", str(mc_cutoff), "--mc_samples", str(mc_samples),
                "--seed", str(seed),
            ])
            rec_theta = _recovered(sig_theta, p_items, "pval_theta", delta_theta)

    return rec_exp, rec_theta


# =====================================================================
# WY threshold on a background-only null cohort (calibrated once)
# =====================================================================
def calibrate_threshold(null_dir, rng, n_patients, M, bg_labels, sigma, theta,
                        lcmdir, null_model, mc_cutoff, mc_samples, wy_M, alpha,
                        par, seed, test, verbose=False):
    """Build a background-only cohort and derive delta_hat(alpha) via WY."""
    ensure_dir(null_dir)
    inputs_dir = null_dir / "inputs"
    ensure_dir(inputs_dir)
    data = build_cohort(rng, n_patients, M, traj=[], bg_labels=bg_labels,
                        carriers=set(), f=0.0)
    graphs_all, weights_uniform, owner_txt, _ = build_inputs(
        list(data), inputs_dir, seed=seed)

    _, _o, _np_, _K = load_weights_and_owner(weights_uniform, owner_txt)
    workdir = null_dir / "resamples"
    ensure_dir(workdir)
    worker_args = [(
        r, graphs_all.resolve(), weights_uniform.resolve(), owner_txt.resolve(),
        float(sigma), lcmdir, workdir, test, null_model, theta,
        mc_cutoff, mc_samples, seed, True, verbose,
    ) for r in range(wy_M)]

    if par > 1:
        with Pool(par) as pool:
            results = list(pool.imap_unordered(_run_resample, worker_args))
    else:
        results = [_run_resample(wa) for wa in worker_args]
    results.sort(key=lambda x: x[0])
    print(f"[POW] WY calibration done ({wy_M} null datasets)", flush=True)

    def _min(vals):
        return min(vals) if vals else 1.0
    min_exp = sorted(_min(pv) for _, pv, _ in results)
    min_theta = sorted(_min(pv) for _, _, pv in results)
    d_exp = wy_threshold(min_exp, alpha, wy_M) if min_exp else 0.0
    d_theta = wy_threshold(min_theta, alpha, wy_M) if min_theta else 0.0
    return d_exp, d_theta


def main():
    ap = argparse.ArgumentParser(description="Ensemble implant / power experiment")
    ap.add_argument("--n_patients", type=int, default=60)
    ap.add_argument("--M", type=int, default=6, help="trees per patient")
    ap.add_argument("--k", type=int, default=4, help="implanted trajectory length")
    ap.add_argument("--n_background", type=int, default=8)
    ap.add_argument("--N_list", default="10,20,30", help="carrier counts")
    ap.add_argument("--f_list", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--null", choices=["perm", "indep"], default="perm")
    ap.add_argument("--test", choices=["exp", "theta", "both"], default="both")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--wy_M", type=int, default=200,
                    help="resamples for the one-off WY threshold calibration")
    ap.add_argument("--mc_cutoff", type=int, default=8)
    ap.add_argument("--mc_samples", type=int, default=2000)
    ap.add_argument("--lcmdir", default="./lcm53")
    ap.add_argument("--par", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true",
                    help="Echo every per-trial subcommand (default: quiet)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    set_verbose(args.verbose)

    lcmdir = Path(args.lcmdir).resolve()
    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)

    want_exp = args.test in ("exp", "both")
    want_theta = args.test in ("theta", "both")

    bg_labels = [f"B{i}" for i in range(args.n_background)]
    traj = [f"X{i}" for i in range(args.k)]
    N_list = [int(x) for x in args.N_list.split(",") if x.strip()]
    f_list = [float(x) for x in args.f_list.split(",") if x.strip()]

    print(f"[POW] n_patients={args.n_patients} M={args.M} k={args.k} "
          f"theta={args.theta} sigma={args.sigma} alpha={args.alpha}")
    print(f"[POW] N_list={N_list} f_list={f_list} trials={args.n_trials}")

    # Calibrate the WY threshold once on a background-only cohort (cached).
    thr_path = outdir / "_null" / "thresholds.json"
    if thr_path.exists():
        thr = json.loads(thr_path.read_text())
        delta_exp, delta_theta = thr["delta_exp"], thr["delta_theta"]
        print(f"[POW] WY thresholds loaded from cache {thr_path}")
    else:
        cal_rng = np.random.default_rng(args.seed + 999)
        delta_exp, delta_theta = calibrate_threshold(
            outdir / "_null", cal_rng, args.n_patients, args.M, bg_labels,
            args.sigma, args.theta, lcmdir, args.null, args.mc_cutoff,
            args.mc_samples, args.wy_M, args.alpha, args.par, args.seed,
            test=args.test, verbose=args.verbose)
        ensure_dir(thr_path.parent)
        thr_path.write_text(json.dumps(
            {"delta_exp": delta_exp, "delta_theta": delta_theta}))
    print(f"[POW] WY thresholds @ alpha={args.alpha}: "
          f"delta_exp={delta_exp:.3e} delta_theta={delta_theta:.3e}")

    rng = np.random.default_rng(args.seed)
    out_csv = outdir / "recall.csv"
    rows = []
    done = set()
    if out_csv.exists():
        for r in csv.DictReader(open(out_csv)):
            rows.append(r)
            done.add((int(r["N"]), float(r["f"])))
        print(f"[POW] resume: {len(done)} (N,f) cells already done")

    def _flush():
        with out_csv.open("w", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=["N", "f", "test", "recall"])
            w.writeheader()
            w.writerows(rows)

    step = max(1, args.n_trials // 4)
    for N in N_list:
        for f in f_list:
            if (N, f) in done:
                print(f"[POW] N={N} f={f:.2f}  [SKIP] already done")
                continue
            n_exp = n_theta = 0
            for t in range(args.n_trials):
                carriers = rng.choice(args.n_patients, size=N, replace=False)
                data = build_cohort(rng, args.n_patients, args.M, traj,
                                    bg_labels, carriers, f)
                trial_dir = outdir / f"trials/N{N}_f{f}/t{t}"
                rec_e, rec_t = run_trial(
                    trial_dir, data, traj, args.sigma, args.theta, lcmdir,
                    delta_exp, delta_theta, args.null, args.mc_cutoff,
                    args.mc_samples, args.seed + t, want_exp, want_theta)
                n_exp += int(rec_e)
                n_theta += int(rec_t)
                # keep disk small
                _cleanup(trial_dir)
                if (t + 1) % step == 0:
                    print(f"[POW]   N={N} f={f:.2f}  trial {t+1}/{args.n_trials}",
                          flush=True)
            if want_exp:
                rows.append({"N": N, "f": f, "test": "exp",
                             "recall": f"{n_exp / args.n_trials:.4f}"})
            if want_theta:
                rows.append({"N": N, "f": f, "test": "theta",
                             "recall": f"{n_theta / args.n_trials:.4f}"})
            done.add((N, f))
            _flush()          # persist after every (N,f) so a crash can resume
            print(f"[POW] N={N} f={f:.2f}  "
                  f"recall_exp={n_exp/args.n_trials:.2f}  "
                  f"recall_theta={n_theta/args.n_trials:.2f}")

    print(f"[POW] written -> {out_csv}")


def _cleanup(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()

"""Power: recovery of an implanted trajectory (ensemble version).

A ground-truth experiment designed to expose the difference between the two
tests. A known linear trajectory P = X1 -> X2 -> ... -> Xk over fresh
alterations is implanted into N randomly chosen "carrier" patients. Every
patient holds one alteration set, the X alterations plus 1-2 background ones
drawn once per patient, and every tree is a chain over that whole set. In each
carrier we implant P perfectly in a fraction f of that patient's M trees (X in
the implanted order, background at random positions) and scramble the
remaining (1 - f) trees (a uniformly random order of the whole set).
Non-carriers hold only scrambled trees, so presence and co-occurrence of the X
alterations are identical across the whole cohort and only the *conserved
order* of P separates carriers from non-carriers. A scrambled tree is a uniform
relabeling of any other tree of its patient, so a cohort without carriers
follows the per-patient permutation null exactly.

With uniform weights w = 1/M this gives, by construction:
    * expected support    s^exp(P)   is about N * f
    * theta-consensus      s^theta(P) = N if f >= theta, else 0
so the expected-support test should recover P once N*f is large enough, while
the theta-consensus test should recover P only when the per-patient consistency
f reaches theta. This directly illustrates the incomparability of the two
support notions.

For each (N, f) we measure the recall, the fraction of trials in which P is
reported significant at a WY-corrected FWER threshold delta_hat(alpha). With
--calibration global the threshold is calibrated once, on a carrier-free cohort
of the same shape, and reused in every cell; with --calibration cell it is
calibrated once per (N, f), on an implanted cohort of that cell whose resamples
relabel every patient, carriers included. The per-patient null is conditional
on the trees of each patient, so the agreeing trees of the carriers change the
resamples, and only the per-cell threshold sees that.

Output: recall.csv with (N, f, test, recall); trials.csv with the smallest
p-value of a tested trajectory containing P, per trial and test, from which the
recall is recomputed for any threshold without rerunning the trials; and the
thresholds with their WY minima in _null/ (global) or _cal_m<wy_M>/ (cell).
"""

import argparse
import csv
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from utils import (
    PYTHON,
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


def _perfect_tree(rng, traj, bg):
    """Carrier tree containing P as a real ancestor chain, with the patient's
    background alterations inserted at random positions."""
    seq = list(traj)
    for b in bg:
        seq.insert(int(rng.integers(0, len(seq) + 1)), b)
    return np.array(_chain_edges("GL", seq), dtype=object)


def _noisy_tree(rng, traj, bg):
    """Tree in which the patient's alterations, P's and the background, are
    laid out as a chain in a uniformly random order.

    P (which needs all C(k,2) forward pairs) is realised only when the relative
    order of its k alterations happens to be the conserved one, an event of
    probability 1/k!. This is deliberate: the
    resulting mild, controlled contamination makes the per-patient presence a
    random variable slightly above f, which turns the recovery curves into
    smooth gradients (and keeps the null non-degenerate) rather than sharp steps.
    A moderate k (we use k = 4, so the accidental rate is 1/24) keeps this noise
    small enough not to saturate the low-theta regime."""
    seq = list(traj) + list(bg)
    rng.shuffle(seq)  # realises P w.p. 1/k!, as the null does
    return np.array(_chain_edges("GL", seq), dtype=object)


def build_cohort(rng, n_patients, M, traj, bg_labels, carriers, f):
    """Build a patients->trees object array with P implanted in `carriers`."""
    carriers = set(int(c) for c in carriers)
    data = np.empty(n_patients, dtype=object)
    for i in range(n_patients):
        trees = []
        # One background set per patient: all trees of a patient carry the
        # same alterations, as the per-patient permutation null assumes
        n_bg = int(rng.integers(1, 3))
        bg = list(rng.choice(bg_labels, size=min(n_bg, len(bg_labels)),
                             replace=False))
        if i in carriers:
            n_perfect = int(round(f * M))
            for j in range(M):
                if j < n_perfect:
                    trees.append(_perfect_tree(rng, traj, bg))
                else:
                    trees.append(_noisy_tree(rng, traj, bg))
        else:
            # Non-carriers now also carry the trajectory alterations, but always
            # scrambled: presence and co-occurrence match the carriers, so only
            # the conserved order distinguishes a carrier from a non-carrier. A
            # non-carrier supports P only by the same accidental 1/k! chance as a
            # carrier's noisy trees. When traj is empty this degenerates to a
            # pure background tree.
            for _ in range(M):
                if len(traj):
                    trees.append(_noisy_tree(rng, traj, bg))
                else:
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
def _min_p_containing(sig_csv, p_items, col):
    """Smallest p-value among the tested trajectories that contain P, or inf
    when none is tested. P is recovered at threshold delta iff this is <= delta,
    so the value alone lets the recall be recomputed for any threshold."""
    best = float("inf")
    if not sig_csv.exists():
        return best
    with sig_csv.open() as f:
        for row in csv.DictReader(f):
            v = row.get(col, "")
            if not v:
                continue
            if p_items.issubset(set(row["pattern"].split())):
                best = min(best, float(v))
    return best


def run_trial(trial_dir, data, traj, sigma, theta, lcmdir,
              null_model, mc_cutoff, mc_samples, seed, want_exp, want_theta):
    """Build inputs, mine, score, and return, per test, the smallest p-value of
    a tested trajectory containing P (inf if none)."""
    ensure_dir(trial_dir)
    inputs_dir = trial_dir / "inputs"
    ensure_dir(inputs_dir)
    graphs_all, weights_uniform, owner_txt, _ = build_inputs(
        list(data), inputs_dir, seed=seed)

    # expected-support maximal family
    alg1 = run_pipeline(graphs_txt=graphs_all, sigma=float(sigma), lcmdir=lcmdir,
                        workdir=trial_dir / "alg1", tag="alg1",
                        weights_txt=weights_uniform, weighted=True)

    p_items = traj_items(traj)
    minp_exp = minp_theta = float("inf")

    if want_exp:
        sig_exp = trial_dir / "pvalues_exp.csv"
        run_cmd([
            PYTHON, str(SCRIPT_DIR / "compute_significance_ensemble.py"),
            "-i", str(alg1), "-o", str(sig_exp),
            "-w", str(weights_uniform), "--owner", str(owner_txt),
            "--graphs_all", str(graphs_all),
            "--test", "exp", "--null", null_model,
            "--mc_cutoff", str(mc_cutoff), "--mc_samples", str(mc_samples),
            "--seed", str(seed),
        ])
        minp_exp = _min_p_containing(sig_exp, p_items, "pval_exp")

    if want_theta:
        # theta-maximal family
        alg3 = trial_dir / "alg3.txt"
        run_cmd([
            PYTHON, str(SCRIPT_DIR / "postfilter_theta.py"),
            "-i", str(alg1), "-o", str(alg3),
            "-w", str(weights_uniform), "-owner", str(owner_txt),
            "-theta", str(theta), "-st", str(int(round(sigma))), "--maximal",
        ])
        if alg3.exists() and alg3.stat().st_size > 0:
            sig_theta = trial_dir / "pvalues_theta.csv"
            run_cmd([
                PYTHON, str(SCRIPT_DIR / "compute_significance_ensemble.py"),
                "-i", str(alg3), "-o", str(sig_theta),
                "-w", str(weights_uniform), "--owner", str(owner_txt),
                "--graphs_all", str(graphs_all),
                "--test", "theta", "--theta", str(theta), "--null", null_model,
                "--mc_cutoff", str(mc_cutoff), "--mc_samples", str(mc_samples),
                "--seed", str(seed),
            ])
            minp_theta = _min_p_containing(sig_theta, p_items, "pval_theta")

    return minp_exp, minp_theta


# =====================================================================
# WY threshold from the resamples of one cohort
# =====================================================================
def calibrate_threshold(null_dir, rng, n_patients, M, bg_labels, traj, sigma, theta,
                        lcmdir, null_model, mc_cutoff, mc_samples, wy_M, alpha,
                        par, seed, test, verbose=False, N=0, f=0.0):
    """Build one cohort and derive delta_hat(alpha) via WY on its resamples.

    With N = 0 (calibration "global") the cohort has no carriers, so every
    patient carries the trajectory alterations but always scrambled: presence
    and co-occurrence match the implanted cohorts, and no conserved order is
    present anywhere. With N > 0 (calibration "cell") the cohort is an implanted
    cohort of the (N, f) cell, and each resample relabels every patient,
    carriers included, as the per-patient null of the real pipeline does: the
    agreeing trees of the carriers then make chance consensus more likely in the
    resamples too, which a carrier-free cohort misses.

    Returns (delta_exp, delta_theta, sorted exp minima, sorted theta minima)."""
    ensure_dir(null_dir)
    inputs_dir = null_dir / "inputs"
    ensure_dir(inputs_dir)
    carriers = rng.choice(n_patients, size=N, replace=False) if N else set()
    data = build_cohort(rng, n_patients, M, traj=traj, bg_labels=bg_labels,
                        carriers=carriers, f=f)
    graphs_all, weights_uniform, owner_txt, _ = build_inputs(
        list(data), inputs_dir, seed=seed)

    _, _o, _np_, _K = load_weights_and_owner(weights_uniform, owner_txt)
    workdir = null_dir / "resamples"
    ensure_dir(workdir)
    worker_args = [(
        r, graphs_all.resolve(), weights_uniform.resolve(), owner_txt.resolve(),
        float(sigma), lcmdir, workdir, test, null_model, theta,
        mc_cutoff, mc_samples, seed, True, verbose, None,  # theta_cand_sigma: keep legacy behaviour
        False,  # exact_when_cheaper: every synthetic patient is below mc_cutoff
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
    return d_exp, d_theta, min_exp, min_theta


def main():
    ap = argparse.ArgumentParser(description="Ensemble implant / power experiment")
    ap.add_argument("--n_patients", type=int, default=60)
    ap.add_argument("--M", type=int, default=10, help="trees per patient")
    ap.add_argument("--k", type=int, default=4, help="implanted trajectory length")
    ap.add_argument("--n_background", type=int, default=8)
    ap.add_argument("--N_list", default="2,3,5,10,20,30", help="carrier counts")
    ap.add_argument("--f_list", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--null", choices=["perm", "indep"], default="perm")
    ap.add_argument("--test", choices=["exp", "theta", "both"], default="both")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--wy_M", type=int, default=200,
                    help="resamples per WY threshold calibration")
    ap.add_argument("--calibration", choices=["global", "cell"], default="global",
                    help="global: one WY threshold from a carrier-free cohort, "
                         "reused in every cell; cell: one threshold per (N, f), "
                         "from the resamples of an implanted cohort of that cell")
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
    for f in f_list:
        if abs(f * args.M - round(f * args.M)) > 1e-9:
            ap.error(f"f={f} with M={args.M} gives f*M={f * args.M:g}, not an "
                     f"integer: the realised fraction of perfect trees would "
                     f"differ from f")

    print(f"[POW] n_patients={args.n_patients} M={args.M} k={args.k} "
          f"theta={args.theta} sigma={args.sigma} alpha={args.alpha}")
    print(f"[POW] N_list={N_list} f_list={f_list} trials={args.n_trials} "
          f"calibration={args.calibration} wy_M={args.wy_M}")

    def _thresholds(N, f):
        """WY thresholds of cell (N, f), cached on disk. 'global' calibrates
        once, on a carrier-free cohort; 'cell' once per (N, f), on an implanted
        cohort of that cell drawn from a seed fixed by (N, f). Neither the cohort
        nor its resamples depend on theta, so the exp threshold of a cell is the
        same in the theta = 0.5 and theta = 1 runs."""
        if args.calibration == "global":
            cal_dir, N_cal, f_cal = outdir / "_null", 0, 0.0
            cal_rng = np.random.default_rng(args.seed + 999)
        else:
            cal_dir = outdir / f"_cal_m{args.wy_M}" / f"N{N}_f{f:g}"
            N_cal, f_cal = N, f
            cal_rng = np.random.default_rng(
                [args.seed, 999, N, int(round(f * args.M))])
        thr_path = cal_dir / "thresholds.json"
        if thr_path.exists():
            thr = json.loads(thr_path.read_text())
        else:
            d_exp, d_theta, min_exp, min_theta = calibrate_threshold(
                cal_dir, cal_rng, args.n_patients, args.M, bg_labels, traj,
                args.sigma, args.theta, lcmdir, args.null, args.mc_cutoff,
                args.mc_samples, args.wy_M, args.alpha, args.par, args.seed,
                test=args.test, verbose=args.verbose, N=N_cal, f=f_cal)
            # the minima are kept so the threshold can be recomputed offline
            # at another alpha
            thr = {"delta_exp": d_exp, "delta_theta": d_theta,
                   "wy_M": args.wy_M, "min_exp": min_exp, "min_theta": min_theta}
            thr_path.write_text(json.dumps(thr))
            print(f"[POW] WY thresholds @ alpha={args.alpha} ({cal_dir.name}): "
                  f"delta_exp={d_exp:.3e} delta_theta={d_theta:.3e}", flush=True)
        return thr["delta_exp"], thr["delta_theta"]

    rng = np.random.default_rng(args.seed)
    out_csv = outdir / "recall.csv"
    rows = []

    # Per-trial smallest p-value of a tested trajectory containing P: the
    # recall of a cell follows from these and its thresholds, so a cell whose
    # trials are cached is not rerun (resume, or a recalibration with another
    # wy_M or --calibration on the same trials).
    trials_csv = outdir / "trials.csv"
    trial_fields = ["N", "f", "trial", "minp_exp", "minp_theta"]
    trials = {}
    if trials_csv.exists():
        for r in csv.DictReader(open(trials_csv)):
            trials.setdefault((int(r["N"]), float(r["f"])), []).append(r)
        print(f"[POW] trials cached for {len(trials)} (N,f) cells")

    def _flush():
        with out_csv.open("w", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=["N", "f", "test", "recall"])
            w.writeheader()
            w.writerows(rows)
        with trials_csv.open("w", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=trial_fields)
            w.writeheader()
            for cell_rows in trials.values():
                w.writerows(cell_rows)

    step = max(1, args.n_trials // 4)
    for N in N_list:
        for f in f_list:
            delta_exp, delta_theta = _thresholds(N, f)
            # The cohorts are drawn here, in trial order, from the one shared
            # rng, also for a cached cell, so the random stream (and every
            # result) is the same with or without --par or a resume; only
            # mining + scoring run in parallel.
            tasks = []
            for t in range(args.n_trials):
                carriers = rng.choice(args.n_patients, size=N, replace=False)
                data = build_cohort(rng, args.n_patients, args.M, traj,
                                    bg_labels, carriers, f)
                tasks.append((outdir / f"trials/N{N}_f{f}/t{t}", data, traj,
                              args.sigma, args.theta, lcmdir, args.null,
                              args.mc_cutoff, args.mc_samples, args.seed + t,
                              want_exp, want_theta))
            cached = trials.get((N, f), [])
            if len(cached) == args.n_trials:
                minp = [(float(r["minp_exp"]), float(r["minp_theta"]))
                        for r in sorted(cached, key=lambda r: int(r["trial"]))]
                print(f"[POW] N={N} f={f:.2f}  trials loaded from cache")
            else:
                minp = []
                if args.par > 1:
                    with Pool(args.par) as pool:
                        for done_t, res in enumerate(
                                pool.imap(_trial_worker, tasks), 1):
                            minp.append(res)
                            if done_t % step == 0:
                                print(f"[POW]   N={N} f={f:.2f}  trial {done_t}/{args.n_trials}",
                                      flush=True)
                else:
                    for t, task in enumerate(tasks):
                        minp.append(_trial_worker(task))
                        if (t + 1) % step == 0:
                            print(f"[POW]   N={N} f={f:.2f}  trial {t+1}/{args.n_trials}",
                                  flush=True)
                trials[(N, f)] = [
                    {"N": N, "f": f, "trial": t, "minp_exp": e, "minp_theta": th}
                    for t, (e, th) in enumerate(minp)]
            n_exp = sum(e <= delta_exp for e, _ in minp)
            n_theta = sum(th <= delta_theta for _, th in minp)
            if want_exp:
                rows.append({"N": N, "f": f, "test": "exp",
                             "recall": f"{n_exp / args.n_trials:.4f}"})
            if want_theta:
                rows.append({"N": N, "f": f, "test": "theta",
                             "recall": f"{n_theta / args.n_trials:.4f}"})
            _flush()          # persist after every (N,f) so a crash can resume
            print(f"[POW] N={N} f={f:.2f}  "
                  f"recall_exp={n_exp/args.n_trials:.2f}  "
                  f"recall_theta={n_theta/args.n_trials:.2f}")

    print(f"[POW] written -> {out_csv}")


def _trial_worker(task):
    """run_trial on one prepared cohort, then delete its scratch directory."""
    rec = run_trial(*task)
    # keep disk small
    _cleanup(task[0])
    return rec


def _cleanup(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()

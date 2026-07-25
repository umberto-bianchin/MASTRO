"""
Westfall-Young FWER correction and empirical FDR estimation for MASTRO
ensemble significance tests.

Creates M null datasets using the selected null model, runs the full
weighted mining + ensemble significance pipeline on each, and computes:
  - FWER-corrected significance thresholds (Westfall-Young): the alpha-quantile
    of the per-resample minimum p-values.
  - Empirical FDR curve (Storey-Tibshirani style) when the original dataset's
    p-values CSV is provided via --pvalues_csv (or --pvalues_exp/--pvalues_theta).

Usage:
    # WY only
    python3 run_wy_correction_ensemble.py --graphs_all inputs/graphs_all.txt \
        -w inputs/weights_uniform.txt --owner inputs/owner.txt \
        --sigma 10.0 -M 100 --test both --null perm --theta 1.0 \
        --outdir wy_results --par 4

    # WY + FDR (pass the p-values CSV from the original dataset)
    python3 run_wy_correction_ensemble.py --graphs_all inputs/graphs_all.txt \
        -w inputs/weights_uniform.txt --owner inputs/owner.txt \
        --sigma 10.0 -M 100 --test both --null perm --theta 1.0 \
        --outdir wy_results --par 4 \
        --pvalues_csv original_run/pvalues.csv
"""

import argparse
import csv
import gzip
import json
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from utils import (
    EDGE_SEPARATORS,
    SCRIPT_DIR,
    ensure_dir,
    load_weights_and_owner,
    run_cmd,
    run_transnum,
    run_lcm,
    set_verbose,
    read_result_pairs,
    parse_items_from_pattern_line,
)


# =====================================================================
# Null dataset generation
# =====================================================================
REL_ANC_FWD = 1
REL_ANC_REV = 2
REL_INC = 3
REL_UNC = 4


def split_edge_item(item):
    """Split one transaction item into (left, separator, right)"""
    for sep in EDGE_SEPARATORS:
        if sep in item:
            left, right = item.split(sep, 1)
            return left, sep, right
    return None


def transaction_relation_matrix(line):
    """Build the pairwise relation matrix encoded by one transaction"""
    parsed_items = []
    labels = set()

    for item in line.split():
        parsed = split_edge_item(item)
        if parsed is None:
            continue
        left, sep, right = parsed
        labels.add(left)
        labels.add(right)
        parsed_items.append((left, sep, right))

    labels = sorted(labels)
    idx = {label: pos for pos, label in enumerate(labels)}
    n = len(labels)

    # Missing relations behave as incomparable, matching the old graph logic
    rel = np.full((n, n), REL_INC, dtype=np.int8)
    np.fill_diagonal(rel, REL_UNC)

    for left, sep, right in parsed_items:
        i = idx[left]
        j = idx[right]
        if sep == "->-":
            rel[i, j] = REL_ANC_FWD
            rel[j, i] = REL_ANC_REV
        elif sep == "-?-":
            rel[i, j] = REL_UNC
            rel[j, i] = REL_UNC
        else:
            rel[i, j] = REL_INC
            rel[j, i] = REL_INC

    return labels, rel


def relation_to_item(left, right, code):
    """Encode one sampled pairwise relation back as a transaction item"""
    if code == REL_ANC_FWD:
        return f"{left}->-{right}"
    if code == REL_ANC_REV:
        return f"{right}->-{left}"

    x, y = (left, right) if left < right else (right, left)
    if code == REL_UNC:
        return f"{x}-?-{y}"
    return f"{x}-/-{y}"


def independent_assignment_transaction(line, rng, mapping=None):
    """Sample one transaction under the independent assignment null"""
    labels, rel = transaction_relation_matrix(line)
    n = len(labels)
    if n < 2:
        return line

    idx = {label: pos for pos, label in enumerate(labels)}
    if mapping is None:
        # Each alteration chooses a target position independently and uniformly
        targets = rng.integers(0, n, size=n)
    else:
        # Reuse the patient-level map across all candidate trees
        targets = []
        for label in labels:
            target_label = mapping.get(label)
            if target_label in idx:
                targets.append(idx[target_label])
            else:
                targets.append(int(rng.integers(0, n)))

    sampled_items = []
    for i in range(n):
        for j in range(i + 1, n):
            left = labels[i]
            right = labels[j]
            pos_left = int(targets[i])
            pos_right = int(targets[j])
            code = REL_UNC if pos_left == pos_right else int(rel[pos_left, pos_right])
            sampled_items.append(relation_to_item(left, right, code))

    return " ".join(sorted(sampled_items))


def independent_assignment_transactions(lines, owner0, n_patients, rng):
    """Create null transactions with patient-coherent independent placement"""
    patient_indices = [[] for _ in range(n_patients)]
    for t in range(len(lines)):
        patient_indices[owner0[t]].append(t)

    sampled = [None] * len(lines)
    for i in range(n_patients):
        indices = patient_indices[i]
        if not indices:
            continue

        labels = set()
        for t in indices:
            for item in lines[t].split():
                parsed = split_edge_item(item)
                if parsed is None:
                    continue
                left, _, right = parsed
                labels.add(left)
                labels.add(right)

        labels = sorted(labels)
        if not labels:
            for t in indices:
                sampled[t] = lines[t]
            continue

        # Same map for all trees of the patient, with replacement
        targets = rng.choice(labels, size=len(labels), replace=True)
        mapping = dict(zip(labels, targets))

        for t in indices:
            sampled[t] = independent_assignment_transaction(
                lines[t], rng, mapping=mapping
            )

    return sampled


def permute_transactions(lines, owner0, n_patients, rng):
    """Create permuted transaction lines with per-patient coherent label permutation.

    For each patient i:
      1. Collect all mutation labels across all their trees
      2. Draw a random bijection sigma_i of those labels
      3. Apply sigma_i to every edge-item in every transaction of patient i

    The tree topology is preserved; only the assignment of mutation names
    to tree positions changes.
    """
    patient_indices = [[] for _ in range(n_patients)]
    for t in range(len(lines)):
        patient_indices[owner0[t]].append(t)

    permuted = [None] * len(lines)

    for i in range(n_patients):
        indices = patient_indices[i]
        if not indices:
            continue

        # Collect every mutation label that appears in any tree of patient i.
        # Using the union across trees ensures labels unique to a single tree
        # are included in the permutation domain
        labels = set()
        for t in indices:
            for item in lines[t].split():
                parsed = split_edge_item(item)
                if parsed is None:
                    continue
                left, _, right = parsed
                labels.add(left)
                labels.add(right)

        # Build a single random bijection sigma_i for patient i.
        # Applying the same sigma_i to all trees of the patient preserves
        # inter-tree consistency (coherent null), matching the 'perm' null
        # model used in compute_significance_ensemble.py
        labels_sorted = sorted(labels)
        shuffled = labels_sorted.copy()
        rng.shuffle(shuffled)
        mapping = dict(zip(labels_sorted, shuffled))

        for t in indices:
            new_items = []
            for item in lines[t].split():
                parsed = split_edge_item(item)
                if parsed is None:
                    new_items.append(item)
                    continue

                left, sep, right = parsed
                a = mapping[left]
                b = mapping[right]
                if sep == "->-":
                    # Keep the original direction while changing label identities
                    new_items.append(f"{a}->-{b}")
                else:
                    # Keep symmetric items in canonical lexicographic order
                    x, y = (a, b) if a < b else (b, a)
                    new_items.append(f"{x}{sep}{y}")
            permuted[t] = " ".join(new_items)

    return permuted


# =====================================================================
# Single-resample pipeline
# =====================================================================
def _run_resample(args):
    """Worker: null sample -> mine -> significance -> extract p-values"""
    (r, graphs_path, weights_path, owner_path,
     sigma, lcmdir, workdir, test, null_model, theta,
     mc_cutoff, mc_samples, seed, cleanup, verbose, theta_cand_sigma) = args

    # Explicitly set verbosity in the worker: multiprocessing may use 'spawn'
    # (fresh import, VERBOSE resets to the default) rather than 'fork', so we
    # cannot rely on inheriting the parent's setting. Quiet by default keeps the
    # per-resample mining/significance out of the log.
    set_verbose(verbose)

    rng = np.random.default_rng(seed + r * 1000)
    rdir = workdir / f"resample_{r}"
    ensure_dir(rdir)

    try:
        # Load data independently in each worker
        graphs_lines = [l.rstrip("\n") for l in open(graphs_path)]
        _, owner0, n_patients, K = load_weights_and_owner(
            weights_path, owner_path
        )

        # Step 1: generate the null dataset
        if null_model == "perm":
            null_lines = permute_transactions(graphs_lines, owner0, n_patients, rng)
        elif null_model == "indep":
            null_lines = independent_assignment_transactions(
                graphs_lines, owner0, n_patients, rng
            )
        else:
            raise ValueError(f"Unknown null model: {null_model}")

        null_path = rdir / f"graphs_{null_model}.txt"
        with null_path.open("w") as f:
            for line in null_lines:
                f.write(line + "\n")

        want_exp = test in ("exp", "both")
        want_theta = test in ("theta", "both")
        # Mine the null at the lower theta-candidate threshold when requested,
        # so the null theta family is complete (Section 4.3); exp is recovered
        # afterwards by filtering the mined patterns to s_exp >= sigma. When
        # theta_cand_sigma is None (theta=1, exp-only, or infeasible) we mine at
        # sigma exactly, as before.
        sigma_mine = sigma
        if want_theta and theta_cand_sigma is not None and theta_cand_sigma < sigma:
            sigma_mine = float(theta_cand_sigma)

        # Step 2: mine (transnum -> LCM -> convert_results -> filter_results)
        # Standard pipeline identical to run_pipeline.py, applied to the
        # sampled dataset so we mine under the null distribution.
        table_file = rdir / "table_file"
        graphs_ids = rdir / "graphs_ids.txt"
        lcm_out = rdir / "lcm_out"
        converted = rdir / "converted.txt"
        filtered = rdir / "filtered.txt"
        log_path = rdir / "lcm.log"

        run_transnum(lcmdir, table_file, null_path, graphs_ids)
        run_lcm(lcmdir, graphs_ids, sigma_mine, lcm_out, log_path,
                weights_txt=weights_path)

        if not lcm_out.exists() or lcm_out.stat().st_size == 0:
            return r, [], []

        run_cmd(["python3", str(SCRIPT_DIR / "convert_results.py"),
                 "-m", str(table_file), "-i", str(lcm_out),
                 "-o", str(converted)])

        if not converted.exists() or converted.stat().st_size == 0:
            return r, [], []

        run_cmd(["python3", str(SCRIPT_DIR / "filter_results.py"),
                 "-i", str(converted), "-o", str(filtered)])

        if not filtered.exists() or filtered.stat().st_size == 0:
            return r, [], []

        # Step 3: significance
        # ONE significance pass over the expected-maximal family computes both
        # p-values per pattern (single data load + tensor build). Each test is
        # still scored on ITS OWN family: exp uses every pattern, theta keeps
        # only the theta-maximal subset, which we identify from the post-filter
        # and SELECT here. A pattern's theta p-value is a function of the
        # pattern, the data and theta alone, so selecting after the fact is
        # identical to re-running theta on the post-filtered file, at half the
        # cost.
        sig_seed = seed + r * 1000

        sig_csv = rdir / "pvalues.csv"
        sig_cmd = [
            "python3", str(SCRIPT_DIR / "compute_significance_ensemble.py"),
            "-i", str(filtered), "-o", str(sig_csv),
            "-w", str(weights_path), "--owner", str(owner_path),
            "--graphs_all", str(null_path),
            "--test", test, "--null", null_model,
            "--mc_cutoff", str(mc_cutoff), "--mc_samples", str(mc_samples),
            "--seed", str(sig_seed),
        ]
        if want_theta:
            sig_cmd += ["--theta", str(theta)]
        run_cmd(sig_cmd)

        rows = []
        if sig_csv.exists():
            with sig_csv.open() as f:
                rows = list(csv.DictReader(f))

        all_exp = []
        if want_exp:
            # exp family is s_exp >= sigma; when sigma_mine < sigma the mined
            # patterns also include s_exp in [sigma_mine, sigma), which belong
            # only to the theta family and must be excluded from the exp test.
            all_exp = [
                float(row["pval_exp"]) for row in rows
                if row.get("pval_exp")
                and float(row.get("s_exp", 0.0)) >= sigma - 1e-9
            ]

        all_theta = []
        if want_theta:
            # theta-maximal family for this resample (same post-filter as Alg 3);
            # select the theta p-values of those patterns from the single pass.
            # A failure of the post-filter on a single pathological resample must
            # NOT crash the whole run: treat it as "this resample contributed no
            # theta discoveries" (conservative) and carry on.
            filtered_theta = rdir / "filtered_theta.txt"
            try:
                run_cmd([
                    "python3", str(SCRIPT_DIR / "postfilter_theta.py"),
                    "-i", str(filtered), "-o", str(filtered_theta),
                    "-w", str(weights_path), "-owner", str(owner_path),
                    "-theta", str(theta), "-st", str(int(round(sigma))),
                    "--maximal",
                ])
            except subprocess.CalledProcessError as e:
                print(f"[WY][warn] postfilter_theta failed on resample {r} "
                      f"({e}); skipping its theta discoveries", flush=True)
                filtered_theta = None
            theta_max = set()
            if filtered_theta is not None:
                for pl, _ol in read_result_pairs(filtered_theta):
                    items = parse_items_from_pattern_line(pl)
                    if items:
                        theta_max.add(" ".join(sorted(items)))
            all_theta = [
                float(row["pval_theta"]) for row in rows
                if row.get("pval_theta") and row.get("pattern") in theta_max
            ]

        return r, all_exp, all_theta

    finally:
        if cleanup and rdir.exists():
            shutil.rmtree(rdir, ignore_errors=True)


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Westfall-Young FWER correction for MASTRO ensemble significance"
    )
    ap.add_argument("--graphs_all", required=True,
                    help="graphs_all.txt (all trees, all patients)")
    ap.add_argument("-w", "--weights", required=True, help="weights.txt")
    ap.add_argument("--owner", required=True, help="owner.txt")
    ap.add_argument("--sigma", type=float, required=True,
                    help="Support threshold for weighted LCM")
    ap.add_argument("-M", "--n_resamples", type=int, default=100,
                    help="Number of WY resamples (default: 100)")
    ap.add_argument("--test", choices=["exp", "theta", "both"], default="both",
                    help="Which significance test (default: both)")
    ap.add_argument("--null", choices=["perm", "indep"], default="perm",
                    help="Null model for null resampling and significance "
                         "testing (default: perm)")
    ap.add_argument("--theta", type=float, default=1.0,
                    help="Theta for theta-consensus test")
    ap.add_argument("--min_mine_sigma", type=int, default=None,
                    help="Lower bound on the candidate mining threshold for the "
                         "NULL theta-consensus family in each resample. The "
                         "correctness rule is sigma_exp = floor(theta*sigma); when it falls below this bound the null "
                         "theta family is mined at sigma instead (possibly "
                         "incomplete, see Chapter 6). Must match the value passed "
                         "to run_pipeline.py so observed and null families agree. "
                         "Set e.g. 2 on breastCancer; leave unset on TRACERx.")
    ap.add_argument("--mc_cutoff", type=int, default=8,
                    help="M_i above which to use MC sampling (default: 8)")
    ap.add_argument("--mc_samples", type=int, default=10000,
                    help="MC samples per patient (default: 10000)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    ap.add_argument("--lcmdir", default="./lcm53",
                    help="Path to lcm53 directory")
    ap.add_argument("--outdir", required=True,
                    help="Output directory for WY results")
    ap.add_argument("--par", type=int, default=1,
                    help="Number of parallel workers (default: 1)")
    ap.add_argument("--keep", action="store_true",
                    help="Keep resample working directories (default: clean up)")
    ap.add_argument("--verbose", action="store_true",
                    help="Echo every per-resample subcommand (default: quiet, "
                         "showing only the resample progress counter)")
    ap.add_argument("--save_resample_pvals", action="store_true",
                    help="Save the full per-resample p-value lists to disk "
                         "(all_pvals_exp.json.gz, all_pvals_theta.json.gz). "
                         "Allows FDR curves to be recomputed offline "
                         "without re-running the WY permutations.")
    ap.add_argument("--pvalues_exp",
                    help="Observed expected-support p-values CSV (Alg 1 family) "
                         "for the exp FDR curve.")
    ap.add_argument("--pvalues_theta",
                    help="Observed theta-consensus p-values CSV (Alg 3 family) "
                         "for the theta FDR curve.")
    ap.add_argument("--pvalues_csv",
                    help="[deprecated] combined p-values CSV; used as a fallback "
                         "for both tests when --pvalues_exp/--pvalues_theta are "
                         "not given. Prefer the split files: the exp and theta "
                         "tests must be scored on their own mined families.")
    args = ap.parse_args()

    # Quiet by default: workers inherit VERBOSE=False via fork, so per-resample
    # mining/significance stops flooding the log; only the progress counter and
    # the final WY/FDR summary are printed.
    set_verbose(args.verbose)

    lcmdir = Path(args.lcmdir).resolve()
    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)
    workdir = outdir / "resamples"
    ensure_dir(workdir)

    graphs_path = Path(args.graphs_all).resolve()
    weights_path = Path(args.weights).resolve()
    owner_path = Path(args.owner).resolve()

    # Validate inputs
    graphs_lines = [l.rstrip("\n") for l in open(graphs_path)]
    _, owner0, n_patients, K = load_weights_and_owner(weights_path, owner_path)
    assert len(graphs_lines) == K, (
        f"graphs has {len(graphs_lines)} lines but weights/owner has {K}"
    )

    print(f"[WY] graphs = {graphs_path}")
    print(f"[WY] n_patients = {n_patients}   transactions = {K}")
    print(f"[WY] sigma = {args.sigma}")
    print(f"[WY] resamples = {args.n_resamples}")
    print(f"[WY] test={args.test} null={args.null} theta={args.theta}")
    print(f"[WY] parallel = {args.par}")

    # Candidate mining threshold for the NULL theta family (Section 4.3):
    # sigma_exp = floor(theta*sigma). None keeps the old behaviour (mine at
    # sigma) for theta = 1, for the exp-only test, or when the required
    # threshold is below --min_mine_sigma (infeasible, Chapter 6).
    sigma_int = int(round(args.sigma))
    theta_cand = None
    if args.test in ("theta", "both") and args.theta < 1.0:
        sc = max(1, int(args.theta * args.sigma))
        if sc < sigma_int:
            if args.min_mine_sigma is not None and sc < args.min_mine_sigma:
                print(f"[WY][WARN] theta={args.theta}: correct null candidate "
                      f"threshold sigma_exp={sc} is below --min_mine_sigma="
                      f"{args.min_mine_sigma}; mining the null theta family at "
                      f"sigma={sigma_int} instead (may be incomplete, Chapter 6).",
                      flush=True)
            else:
                theta_cand = sc
                print(f"[WY] null theta family mined at sigma_exp={sc} "
                      f"(= floor(theta*sigma)) for completeness.", flush=True)

    worker_args = []
    for r in range(args.n_resamples):
        worker_args.append((
            r, graphs_path, weights_path, owner_path,
            args.sigma, lcmdir, workdir,
            args.test, args.null, args.theta,
            args.mc_cutoff, args.mc_samples,
            args.seed, not args.keep, args.verbose, theta_cand,
        ))

    total = args.n_resamples
    step = max(1, total // 100)  # ~1% progress ticks
    results = []
    if args.par > 1:
        with Pool(args.par) as pool:
            for done, res in enumerate(
                    pool.imap_unordered(_run_resample, worker_args), 1):
                results.append(res)
                if done % step == 0 or done == total:
                    print(f"[WY] resample {done}/{total} done", flush=True)
        results.sort(key=lambda x: x[0])
    else:
        for done, wa in enumerate(worker_args, 1):
            res = _run_resample(wa)
            results.append(res)
            if done % step == 0 or done == total:
                min_e = min(res[1]) if res[1] else 1.0
                min_t = min(res[2]) if res[2] else 1.0
                print(f"[WY] resample {done}/{total} done  "
                      f"(min_exp={min_e:.3e} min_theta={min_t:.3e})", flush=True)

    # Collect per-resample p-value lists and min p-values
    M = args.n_resamples
    all_pvals_exp = []    # list of lists, one per resample
    all_pvals_theta = []
    min_exp_vals = []
    min_theta_vals = []
    for r_id, pv_exp, pv_theta in sorted(results):
        all_pvals_exp.append(pv_exp)
        all_pvals_theta.append(pv_theta)
        min_exp_vals.append(min(pv_exp) if pv_exp else 1.0)
        min_theta_vals.append(min(pv_theta) if pv_theta else 1.0)

    # Write raw min p-values
    raw_path = outdir / "min_pvalues.csv"
    with raw_path.open("w") as f:
        f.write("resample,min_pval_exp,min_pval_theta\n")
        for r_id in range(M):
            f.write(f"{r_id},{min_exp_vals[r_id]:.6e},{min_theta_vals[r_id]:.6e}\n")

    # ---- Optional: persist full per-resample p-value lists ----
    # When --save_resample_pvals is set, dump all_pvals_exp / all_pvals_theta
    # to compressed JSON, so FDR curves can be recomputed offline
    # without re-running the M permutations.  Useful when:
    #   - --pvalues_csv was forgotten on the original run
    #   - the observed p-values change (e.g. after a refinement) but the
    #     null distribution is unchanged
    if args.save_resample_pvals:
        for label, data in (("exp", all_pvals_exp), ("theta", all_pvals_theta)):
            jpath = outdir / f"all_pvals_{label}.json.gz"
            with gzip.open(jpath, "wt") as f:
                json.dump(data, f)
            print(f"[WY] saved per-resample {label} p-values -> {jpath}")

    # ---- FWER: Westfall-Young corrected thresholds ----
    # WY threshold at level alpha = alpha-th quantile of the resample minima
    # Reject H_P iff p_P <= threshold(alpha)
    # Because we use the *minimum* across all patterns in each resample, the
    # procedure controls FWER over the full family of tested trajectories.
    min_exp_sorted = sorted(min_exp_vals)
    min_theta_sorted = sorted(min_theta_vals)

    print(f"\n[WY] === FWER-corrected thresholds ({M} resamples) ===")
    thresholds_path = outdir / "wy_thresholds.txt"
    with thresholds_path.open("w") as f:
        for alpha in [0.01, 0.05, 0.1]:
            idx = max(0, int(np.floor(alpha * M)) - 1)
            t_exp = min_exp_sorted[idx]
            t_theta = min_theta_sorted[idx]
            line = (f"alpha={alpha:.2f}  "
                    f"threshold_exp={t_exp:.6e}  "
                    f"threshold_theta={t_theta:.6e}")
            print(f"[WY] {line}")
            f.write(line + "\n")

    # ---- FDR: empirical estimation ----
    # Requires observed p-values. The exp and theta curves each use their own
    # observed family (Alg 1 vs Alg 3); a combined --pvalues_csv is accepted
    # only as a fallback for back-compatibility.
    if args.pvalues_exp or args.pvalues_theta or args.pvalues_csv:
        def _load_col(path, col):
            vals = []
            if not path:
                return vals
            with open(path) as f:
                for row in csv.DictReader(f):
                    v = row.get(col, "")
                    if v:
                        vals.append(float(v))
            return vals

        obs_exp = _load_col(args.pvalues_exp or args.pvalues_csv, "pval_exp")
        obs_theta = _load_col(args.pvalues_theta or args.pvalues_csv, "pval_theta")

        def compute_fdr_curve(obs_pvals, null_pval_lists, label):
            """Compute FDR(delta) at each observed p-value threshold

            FDR_hat(delta) = average null discoveries at delta
                             / max(observed discoveries at delta, 1)

            Numerator: average number of null discoveries at threshold delta
            Denominator: observed discoveries at the same threshold
            """
            if not obs_pvals:
                return []
            # Sort observed p-values so we can enumerate thresholds in order;
            # rank k (1-based) equals the denominator at threshold obs_sorted[k-1].
            obs_sorted = sorted(obs_pvals)
            obs_arr = np.array(obs_sorted)
            null_arrays = [np.array(pv) for pv in null_pval_lists]
            m = len(null_pval_lists)
            rows = []
            for k, delta in enumerate(obs_sorted, 1):
                denom = max(int((obs_arr <= delta).sum()), 1)
                # Cap individual resample contributions at 1.0 to avoid
                # FDR > 1 when a permuted dataset happens to yield more
                # discoveries than the original.
                ratios = [
                    min(int((arr <= delta).sum()) / denom, 1.0) if len(arr) > 0 else 0.0
                    for arr in null_arrays
                ]
                fdr_est = sum(ratios) / float(m)
                rows.append({"rank": k, "pvalue": f"{delta:.6e}",
                             f"fdr_{label}": f"{fdr_est:.6f}"})
            return rows

        fdr_rows_exp = compute_fdr_curve(obs_exp, all_pvals_exp, "exp")
        fdr_rows_theta = compute_fdr_curve(obs_theta, all_pvals_theta, "theta")

        if fdr_rows_exp:
            fdr_path = outdir / "fdr_exp.csv"
            with fdr_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["rank", "pvalue", "fdr_exp"])
                writer.writeheader()
                for row in fdr_rows_exp:
                    writer.writerow(row)
            print(f"[FDR] exp curve written to {fdr_path} ({len(fdr_rows_exp)} entries)")

        if fdr_rows_theta:
            fdr_path = outdir / "fdr_theta.csv"
            with fdr_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["rank", "pvalue", "fdr_theta"])
                writer.writeheader()
                for row in fdr_rows_theta:
                    writer.writerow(row)
            print(f"[FDR] theta curve written to {fdr_path} ({len(fdr_rows_theta)} entries)")
    else:
        print("\n[FDR] Skipped (no --pvalues_exp/--pvalues_theta provided)")

    print(f"\n[WY/FDR] Results written to {outdir}")


if __name__ == "__main__":
    main()

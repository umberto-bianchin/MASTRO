"""End-to-end runner for the ensemble MASTRO pipeline.

Given a dataset (either a .npy object array indexed data[patient][tree][edge] =
[parent, child], or a pre-built graphs .txt plus an owner file), this script
builds the transaction inputs and mines four families of trajectories:

  Algorithm 0 : single-tree baseline. One tree per patient is sampled at random
                and mined with unweighted frequent-itemset mining.
  Algorithm 1 : expected support. All candidate trees are pooled, each weighted
                by 1/M_i (M_i is the number of trees of patient i), and mined
                with weighted frequent-itemset mining.
  Algorithm 2 : theta-frequent post-filter applied on top of Algorithm 1.
  Algorithm 3 : theta-maximal post-filter applied on top of Algorithm 1.

The four mined families are then compared (Jaccard similarity, per-method unique
patterns). With --significance every family is scored under the ensemble null
(compute_significance_ensemble.py): the expected-support test on Alg 0 and Alg 1,
the theta-consensus test on Alg 2 and Alg 3. With --single_tree_only only
Algorithm 0 is built, mined and scored; the seed-independent ensemble families
(Alg 1/2/3) are skipped, which is what the single-tree seed sweep needs since
those families would be recomputed identically for every seed.

Every stage skips itself when its output file already exists, so an interrupted
run can be resumed by re-invoking it with the same --outdir.

Example:
  python3 run_pipeline.py --npy ../data/breastCancer.npy --sigma 5 --seed 0 \\
      --theta_list 0.5,1.0 --significance --sig_null perm
"""

import argparse
import json
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
from utils import (
    SCRIPT_DIR,
    ensure_dir,
    run_cmd,
    load_npy,
    build_inputs,
    run_transnum,
    run_lcm,
    read_result_itemsets,
    compute_dataset_stats,
)

# -------------------------
# Run pipelines
# -------------------------
def run_pipeline(graphs_txt: Path, sigma: float, lcmdir: Path, workdir: Path, tag: str, weights_txt: Optional[Path] = None, weighted: bool = False):
    """Algorithm 0-1 pipeline: FIM on one-tree-per-patient/all trees data

    Steps:  transnum -> LCM (no -w) -> convert_results -> filter_results
    """
    ensure_dir(workdir)

    # LCM scratch files live in this run's workdir (not the shared lcmdir), so
    # independent pipelines can mine from the same folder in parallel without
    # clobbering each other's table-file / lcm-out intermediates.
    table_file_ids = workdir / f"table-file-{graphs_txt.name}"
    file_graphs_ids = workdir / f"lcm-out-{graphs_txt.stem}_ids.txt"
    output_lcm = workdir / f"lcm-out-{graphs_txt.name}"

    results_converted = workdir / f"{tag}_convres.txt"
    results_filtered = workdir / f"{tag}_filtered.txt"

    run_transnum(lcmdir, table_file_ids, graphs_txt, file_graphs_ids)

    log_path = workdir / f"{tag}_lcm.log"

    if results_converted.exists():
        print(f"[SKIP] {results_converted.name} already exists - skipping run_lcm")
    else:
        if weighted:
            run_lcm(lcmdir, file_graphs_ids, sigma, output_lcm, log_path, weights_txt=weights_txt)
        else:
            run_lcm(lcmdir, file_graphs_ids, sigma, output_lcm, log_path, weights_txt=None)

        run_cmd(["python3", str(SCRIPT_DIR / "convert_results.py"),
                 "-m", str(table_file_ids), "-i", str(output_lcm), "-o", str(results_converted)])

    if results_filtered.exists():
        print(f"[SKIP] {results_filtered.name} already exists - skipping filter_results")
    else:
        run_cmd(["python3", str(SCRIPT_DIR / "filter_results.py"),
                 "-i", str(results_converted), "-o", str(results_filtered)])

    return results_filtered

def run_postfilters(expected_filtered: Path, weights_txt: Path, owner_txt: Path, theta: float, st: int, workdir: Path, tag: str):
    """Stage-2 post-filters (Algorithms 2 and 3) on already-filtered Alg-1 output

    Runs both Algorithm 2 (theta-frequent) and Algorithm 3 (theta-maximal),
    returning the paths to both output files
    """
    out_alg2 = workdir / f"{tag}_alg2_theta{theta}_st{st}.txt"
    out_alg3 = workdir / f"{tag}_alg3_theta{theta}_st{st}.txt"

    run_cmd(["python3", str(SCRIPT_DIR / "postfilter_theta.py"),
             "-i", str(expected_filtered),
             "-o", str(out_alg2),
             "-w", str(weights_txt),
             "-owner", str(owner_txt),
             "-theta", str(theta),
             "-st", str(st)])

    run_cmd(["python3", str(SCRIPT_DIR / "postfilter_theta.py"),
             "-i", str(expected_filtered),
             "-o", str(out_alg3),
             "-w", str(weights_txt),
             "-owner", str(owner_txt),
             "-theta", str(theta),
             "-st", str(st),
             "--maximal"])

    return out_alg2, out_alg3


# -------------------------
# Significance stage
# -------------------------
def _run_sig(input_path, output_path, npy_path, graphs_all_path,
             weights_txt, owner_txt,
             test, null_model, mc_cutoff, mc_samples, seed, theta=None, n_jobs=1):
    """Invoke compute_significance_ensemble.py for one (filtered file, test) pair"""
    cmd = [
        "python3", str(SCRIPT_DIR / "compute_significance_ensemble.py"),
        "-i", str(input_path),
        "-o", str(output_path),
        "-w", str(weights_txt),
        "--owner", str(owner_txt),
        "--test", test,
        "--null", null_model,
        "--mc_cutoff", str(mc_cutoff),
        "--mc_samples", str(mc_samples),
        "--seed", str(seed),
        "--n_jobs", str(n_jobs),
    ]
    if npy_path is not None:
        cmd += ["--npy", str(npy_path)]
    else:
        cmd += ["--graphs_all", str(graphs_all_path)]
    if theta is not None:
        cmd += ["--theta", str(theta)]
    run_cmd(cmd)


def run_significance_tests(sig_dir, npy_path, npy_sampled_path,
                           graphs_all_path, graphs_sampled_path,
                           weights_txt, owner_txt,
                           weights_sampled_txt, owner_sampled_txt,
                           alg0_filtered, alg1_filtered, alg2_paths, alg3_paths,
                           thetas, null_model, mc_cutoff, mc_samples, seed, n_jobs=1,
                           single_tree_only=False):
    """Run significance tests on every pipeline output

    Alg 0 uses the sampled weights/owner (1.0 per patient, 1 transaction per patient)
    Alg 1/2/3 use the uniform weights/owner (1/M_i per tree, M_i transactions per patient)

    Either npy_path or graphs_all_path must be set (not both)
    """
    jobs = []

    # Alg 0 -> expected-support test, with ITS OWN weights/owner
    out = sig_dir / "alg0_mastro_random_pvalues_exp.csv"
    jobs.append(dict(
        input_path=alg0_filtered, output_path=out,
        npy_path=npy_sampled_path,
        graphs_all_path=graphs_sampled_path,
        weights_txt=weights_sampled_txt, owner_txt=owner_sampled_txt,
        test="exp", null_model=null_model,
        mc_cutoff=mc_cutoff, mc_samples=mc_samples, seed=seed,
    ))

    if single_tree_only:
        # Only the single-tree baseline is scored; skip the ensemble families.
        print(f"[INFO] Running {len(jobs)} significance test (single-tree only), "
              f"{n_jobs} workers")
        for j in jobs:
            _run_sig(n_jobs=n_jobs, **j)
        return

    # Alg 1 -> expected-support test, with uniform weights/owner
    out = sig_dir / "alg1_expected_uniform_pvalues_exp.csv"
    jobs.append(dict(
        input_path=alg1_filtered, output_path=out,
        npy_path=npy_path,
        graphs_all_path=graphs_all_path,
        weights_txt=weights_txt, owner_txt=owner_txt,
        test="exp", null_model=null_model,
        mc_cutoff=mc_cutoff, mc_samples=mc_samples, seed=seed,
    ))

    # Alg 2 / Alg 3 -> theta-consensus test, once per theta
    for th in thetas:
        for prefix, paths in [("alg2", alg2_paths), ("alg3", alg3_paths)]:
            key = f"{prefix}_theta{th}"
            if key not in paths:
                continue
            out = sig_dir / f"{key}_pvalues_theta.csv"
            jobs.append(dict(
                input_path=paths[key], output_path=out,
                npy_path=npy_path,
                graphs_all_path=graphs_all_path,
                weights_txt=weights_txt, owner_txt=owner_txt,
                test="theta", null_model=null_model,
                mc_cutoff=mc_cutoff, mc_samples=mc_samples, seed=seed,
                theta=th,
            ))

    # Each significance job is itself parallel over trajectories (--n_jobs),
    # so we run the (few) jobs sequentially and give every job the full worker
    # budget rather than fanning out the handful of CSVs across a shallow pool.
    print(f"[INFO] Running {len(jobs)} significance tests, {n_jobs} workers each")
    for j in jobs:
        _run_sig(n_jobs=n_jobs, **j)

# -------------------------
# Result comparison
# -------------------------
def compare_results(result_paths: dict, outdir: Path):
    """Compare mined pattern sets across all algorithm variants

    Outputs:
      - comparison_summary.json : pattern count per method
      - pairwise_jaccard.csv    : Jaccard similarity for every method pair
      - unique_patterns.txt     : patterns found exclusively by each method
    """
    sets = {name: set(read_result_itemsets(p)) for name, p in result_paths.items()}
    summary = {name: {"n_patterns": len(s)} for name, s in sets.items()}

    names = list(sets.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            A, B = sets[a], sets[b]
            inter = len(A & B)
            uni = len(A | B)
            jac = (inter / uni) if uni else 1.0
            rows.append({"A": a, "B": b, "intersection": inter, "union": uni, "jaccard": jac})

    (outdir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))

    csv_path = outdir / "pairwise_jaccard.csv"
    with csv_path.open("w") as f:
        f.write("A,B,intersection,union,jaccard\n")
        for r in rows:
            f.write(f"{r['A']},{r['B']},{r['intersection']},{r['union']},{r['jaccard']}\n")

    uniq_path = outdir / "unique_patterns.txt"
    with uniq_path.open("w") as f:
        for name in names:
            others = set().union(*[sets[n] for n in names if n != name]) if len(names) > 1 else set()
            only = sets[name] - others
            f.write(f"## Only in {name}: {len(only)} patterns\n")
            for it in sorted(only, key=lambda x: (len(x), sorted(list(x)))):
                f.write("  - " + " ".join(sorted(it)) + "\n")
            f.write("\n")

    print("[OK] Wrote:", outdir / "comparison_summary.json", csv_path, uniq_path)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    input_grp = ap.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--npy", help="Path to .npy dataset (object array data[patient][tree][edge]=[parent,child])")
    input_grp.add_argument("--graphs", help="Path to a .txt file where each line is a pre-computed transaction")
    ap.add_argument("--owner", default=None,
                    help="Owner file for --graphs mode: one patient-ID per line (same #lines as graphs). "
                         "If omitted, each line is treated as a separate patient (1 tree per patient)")
    ap.add_argument("--sigma", type=int, default=10, help="Support threshold sigma")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for picking one tree per patient")
    ap.add_argument("--theta_list", default="0.25,0.5,0.75,1.0", help="Comma-separated list of theta values")
    ap.add_argument("--lcmdir", default="./lcm53", help="Path to lcm53 directory")
    ap.add_argument("--outdir", default=None, help="Output directory (default: results_<timestamp>)")
    ap.add_argument("--keep_gl", action="store_true", help="Keep GL in stats and transactions")
    ap.add_argument("--significance", action="store_true",
                    help="Run ensemble significance tests after mining "
                         "(exp test on alg0/alg1, theta test on alg2/alg3)")
    ap.add_argument("--single_tree_only", action="store_true",
                    help="Only build/mine/score the single-tree baseline (Alg 0). "
                         "Skips the seed-independent ensemble families (Alg 1/2/3) "
                         "and their significance -- for the single-tree seed sweep, "
                         "where redoing the ensemble every seed is pure waste.")
    ap.add_argument("--sig_null", choices=["indep", "perm"], default="perm",
                    help="Null model for significance testing (default: perm)")
    ap.add_argument("--sig_mc_cutoff", type=int, default=8,
                    help="M_i above which to use MC in significance testing")
    ap.add_argument("--sig_mc_samples", type=int, default=10000,
                    help="Monte-Carlo samples per patient in significance testing")
    ap.add_argument("--sig_n_jobs", type=int, default=1,
                    help="Parallel workers for per-patient phi_i computation "
                         "in significance testing (default 1 = sequential)")
    args = ap.parse_args()

    lcmdir = Path(args.lcmdir).resolve()
    drop_gl = not args.keep_gl

    if args.outdir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(f"results_{ts}").resolve()
    else:
        outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)

    inputs_dir = outdir / "inputs"
    ensure_dir(inputs_dir)

    npy_path = None
    npy_sampled_path = None

    if args.npy:
        npy_path = Path(args.npy).resolve()
        data = load_npy(npy_path)
        patients_trees = list(data)

        stats = compute_dataset_stats(patients_trees, drop_gl=drop_gl)
        (outdir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
        print("[OK] Dataset stats written to", outdir / "dataset_stats.json")

        graphs_all, weights_uniform, owner_txt, graphs_sampled = build_inputs(
            patients_trees, inputs_dir, seed=args.seed, drop_gl=drop_gl
        )

        # Alg 0 (single-tree baseline) is scored as an ensemble of size one per
        # patient: exactly one transaction per patient, with weight 1.0. We build
        # a dedicated one-tree-per-patient .npy and matching weights/owner files so
        # the significance test sees the sampled family, not the pooled ensemble.
        weights_sampled_txt = inputs_dir / "weights_sampled.txt"
        owner_sampled_txt = inputs_dir / "owner_sampled.txt"
        with weights_sampled_txt.open("w") as fw, owner_sampled_txt.open("w") as fo:
            for i, tlist in enumerate(patients_trees):
                if not tlist:
                    continue
                fw.write("1.0\n")
                fo.write(f"{i}\n")

        npy_sampled_path = inputs_dir / "sampled.npy"
        rng_sample = np.random.default_rng(args.seed)
        n = len(patients_trees)
        sampled_data = np.empty(n, dtype=object)
        for i, tlist in enumerate(patients_trees):
            if not tlist:
                sampled_data[i] = np.empty((0, 2), dtype=object)
                continue
            j = int(rng_sample.integers(0, len(tlist)))
            sampled_data[i] = np.array([tlist[j]], dtype=object)
        np.save(npy_sampled_path, sampled_data, allow_pickle=True)
        print(f"[OK] Sampled .npy with 1 tree/patient written to {npy_sampled_path}")

    else:
        graphs_src = Path(args.graphs).resolve()
        lines = [l.rstrip("\n") for l in graphs_src.open()]
        n_lines = len(lines)

        if args.owner:
            owner_ids = [l.strip() for l in open(args.owner)]
            assert len(owner_ids) == n_lines, (
                f"owner file has {len(owner_ids)} lines but graphs has {n_lines}")
            # Map owner labels -> 0-based patient indices (preserve order of first appearance)
            label_to_idx = {}
            for lab in owner_ids:
                if lab not in label_to_idx:
                    label_to_idx[lab] = len(label_to_idx)
            owner_ints = [label_to_idx[lab] for lab in owner_ids]
        else:
            # 1 tree per patient
            owner_ints = list(range(n_lines))

        n_patients = max(owner_ints) + 1
        # Group lines by patient
        patient_lines = [[] for _ in range(n_patients)]
        for idx, line in zip(owner_ints, lines):
            patient_lines[idx].append(line)

        # Write graphs_all.txt, weights, owner
        graphs_all = inputs_dir / "graphs_all.txt"
        weights_uniform = inputs_dir / "weights_uniform.txt"
        owner_txt = inputs_dir / "owner.txt"
        with graphs_all.open("w") as fg, weights_uniform.open("w") as fw, owner_txt.open("w") as fo:
            for pid, plines in enumerate(patient_lines):
                Mi = len(plines)
                w = 1.0 / Mi
                for line in plines:
                    fg.write(line + "\n")
                    fw.write(f"{w}\n")
                    fo.write(f"{pid}\n")

        # Write graphs_sampled.txt (one random tree per patient)
        graphs_sampled = inputs_dir / "graphs_sampled.txt"
        rng_sample = np.random.default_rng(args.seed)
        with graphs_sampled.open("w") as fs:
            for plines in patient_lines:
                j = int(rng_sample.integers(0, len(plines)))
                fs.write(plines[j] + "\n")

        # Sampled weights/owner (1 per patient, weight=1.0)
        weights_sampled_txt = inputs_dir / "weights_sampled.txt"
        owner_sampled_txt = inputs_dir / "owner_sampled.txt"
        with weights_sampled_txt.open("w") as fw, owner_sampled_txt.open("w") as fo:
            for pid in range(n_patients):
                fw.write("1.0\n")
                fo.write(f"{pid}\n")

        # Basic stats for .txt mode
        stats = {
            "n_patients": n_patients,
            "n_transactions": n_lines,
            "trees_per_patient": {
                "min": int(min(len(pl) for pl in patient_lines)),
                "median": float(np.median([len(pl) for pl in patient_lines])),
                "max": int(max(len(pl) for pl in patient_lines)),
                "mean": float(np.mean([len(pl) for pl in patient_lines])),
            },
            "input_mode": "graphs_txt",
        }
        (outdir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"[OK] Loaded {n_lines} transactions for {n_patients} patients from {graphs_src.name}")

    runs_dir = outdir / "runs"
    ensure_dir(runs_dir)

    sigma = args.sigma
    thetas = [float(x.strip()) for x in args.theta_list.split(",") if x.strip()]

    # --- Algorithm 0: random sampling, unweighted FIM ---
    alg0_dir = runs_dir / "alg0_mastro_random"
    alg0_filtered = run_pipeline(
        graphs_txt=graphs_sampled,
        sigma=sigma,
        lcmdir=lcmdir,
        workdir=alg0_dir,
        tag="mastro_random"
    )

    alg1_filtered = None
    alg2_paths = {}
    alg3_paths = {}

    if not args.single_tree_only:
        # --- Algorithm 1: all trees, weighted FIM (expected support) ---
        alg1_dir = runs_dir / "alg1_expected_uniform"
        alg1_filtered = run_pipeline(
            graphs_txt=graphs_all,
            sigma=float(sigma),
            lcmdir=lcmdir,
            workdir=alg1_dir,
            tag="expected_uniform",
            weights_txt=weights_uniform,
            weighted=True
        )

        # --- Algorithms 2 & 3: theta-frequent / theta-maximal post-filters ---
        alg23_dir = runs_dir / "alg2_alg3_postfilters"
        ensure_dir(alg23_dir)

        for th in thetas:
            a2, a3 = run_postfilters(
                expected_filtered=alg1_filtered,
                weights_txt=weights_uniform,
                owner_txt=owner_txt,
                theta=th,
                st=sigma,
                workdir=alg23_dir,
                tag="expected_uniform"
            )
            alg2_paths[f"alg2_theta{th}"] = a2
            alg3_paths[f"alg3_theta{th}"] = a3

    analysis_dir = outdir / "analysis"
    ensure_dir(analysis_dir)

    if args.single_tree_only:
        result_paths = {"alg0_mastro_random": alg0_filtered}
    else:
        result_paths = {
            "alg0_mastro_random": alg0_filtered,
            "alg1_expected_uniform": alg1_filtered,
            **alg2_paths,
            **alg3_paths,
        }
    compare_results(result_paths, analysis_dir)

    if args.significance:
        sig_dir = outdir / "significance"
        ensure_dir(sig_dir)
        # In --npy mode: npy_path/npy_sampled_path are set, graphs_all_path is None
        # In --graphs mode: vice-versa
        graphs_all_path = None if npy_path else graphs_all
        graphs_sampled_path = None if npy_path else graphs_sampled
        run_significance_tests(
            sig_dir=sig_dir,
            npy_path=npy_path,
            npy_sampled_path=npy_sampled_path,
            graphs_all_path=graphs_all_path,
            graphs_sampled_path=graphs_sampled_path,
            weights_txt=weights_uniform,
            owner_txt=owner_txt,
            weights_sampled_txt=weights_sampled_txt,
            owner_sampled_txt=owner_sampled_txt,
            alg0_filtered=alg0_filtered,
            alg1_filtered=alg1_filtered,
            alg2_paths=alg2_paths,
            alg3_paths=alg3_paths,
            thetas=thetas,
            null_model=args.sig_null,
            mc_cutoff=args.sig_mc_cutoff,
            mc_samples=args.sig_mc_samples,
            seed=args.seed,
            n_jobs=args.sig_n_jobs,
            single_tree_only=args.single_tree_only,
        )

    print("\n[DONE] Outputs are under:", outdir)


if __name__ == "__main__":
    main()

# =============================================================================
# Breast Cancer Experiment Runner
#
# End-to-end orchestrator that loads a .npy dataset, builds the
# MASTRO input files, and runs all four algorithm variants:
#
#   Algorithm 0  –  Random sampling (one tree per patient, unweighted FIM)
#   Algorithm 1  –  Expected support (all trees, weighted FIM, w_t = 1/M_i)
#   Algorithm 2  –  θ-frequent post-filter  (on top of Alg 1 output)
#   Algorithm 3  –  θ-maximal  post-filter  (on top of Alg 1 output)
#
# After mining, the results are compared via Jaccard similarity and unique-
# pattern analysis.
#
# Usage:
#   python3 breastCancer_experiments.py --npy ../data/breastCancer.npy --sigma 10 --seed 0 --theta_list 0.25,0.5,0.75,1.0
# =============================================================================

import argparse
import json
from pathlib import Path
from datetime import datetime
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
def run_pipeline(graphs_txt: Path, sigma: float, lcmdir: Path, workdir: Path, tag: str, weights_txt: Path = None, weighted: bool = False):
    """Algorithm 0-1 pipeline: FIM on one-tree-per-patient/all trees data.

    Steps:  transnum → LCM (no -w) → convert_results → filter_results
    """
    ensure_dir(workdir)

    table_file_ids = lcmdir / f"table-file-{graphs_txt.name}"
    file_graphs_ids = lcmdir / f"lcm-out-{graphs_txt.stem}_ids.txt"
    output_lcm = lcmdir / f"lcm-out-{graphs_txt.name}"

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
    """Stage-2 post-filters (Algorithms 2 and 3) on already-filtered Alg-1 output.

    Runs both Algorithm 2 (θ-frequent) and Algorithm 3 (θ-maximal),
    returning the paths to both output files.
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
# Result comparison
# -------------------------
def compare_results(result_paths: dict, outdir: Path):
    """Compare mined pattern sets across all algorithm variants.

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
    ap.add_argument("--npy", required=True, help="Path to breastCancer.npy")
    ap.add_argument("--sigma", type=int, default=10, help="Support threshold sigma")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for picking one tree per patient")
    ap.add_argument("--theta_list", default="0.25,0.5,0.75,1.0", help="Comma-separated list of theta values")
    ap.add_argument("--lcmdir", default="./lcm53", help="Path to lcm53 directory")
    ap.add_argument("--outdir", default=None, help="Output directory (default: results_<timestamp>)")
    ap.add_argument("--keep_gl", action="store_true", help="Keep GL in stats and transactions")
    args = ap.parse_args()

    lcmdir = Path(args.lcmdir).resolve()
    npy_path = Path(args.npy).resolve()
    drop_gl = not args.keep_gl

    if args.outdir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(f"results_{ts}").resolve()
    else:
        outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)

    data = load_npy(npy_path)
    patients_trees = list(data)

    stats = compute_dataset_stats(patients_trees, drop_gl=drop_gl)
    (outdir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print("[OK] Dataset stats written to", outdir / "dataset_stats.json")

    inputs_dir = outdir / "inputs"
    ensure_dir(inputs_dir)

    graphs_all, weights_uniform, owner_txt, graphs_sampled = build_inputs(
        patients_trees, inputs_dir, seed=args.seed, drop_gl=drop_gl
    )

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

    # --- Algorithms 2 & 3: θ-frequent / θ-maximal post-filters ---
    # Run for each θ value in the user-supplied list.
    alg23_dir = runs_dir / "alg2_alg3_postfilters"
    ensure_dir(alg23_dir)
    alg2_paths = {}
    alg3_paths = {}

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

    result_paths = {
        "alg0_mastro_random": alg0_filtered,
        "alg1_expected_uniform": alg1_filtered,
        **alg2_paths,
        **alg3_paths,
    }
    compare_results(result_paths, analysis_dir)

    print("\n[DONE] Outputs are under:", outdir)


if __name__ == "__main__":
    main()
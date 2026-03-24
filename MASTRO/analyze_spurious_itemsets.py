# Analyze Spurious Itemsets in MASTRO
# =============================================================================
#
# Quantifies how many frequent itemsets produced by the LCM algorithm are
# "spurious" — i.e., they do not correspond to valid evolutionary trajectories
# and would be discarded by the trajectory-validity filter in filter_results.py.
#
# A valid trajectory of n mutations requires exactly n*(n-1)/2 pairwise ordering
# edges; any itemset that fails this check is spurious.
#
# Pipeline executed for each support threshold (sigma):
#  1. Prepare transaction inputs from the .npy dataset (or reuse existing ones).
#  2. Run LCM frequent-itemset mining with an optional cap on the number of
#     solutions (-# flag) and/or a wall-clock timeout.
#  3. Count itemsets at each stage of the MASTRO filtering pipeline:
#       a) RAW     — all frequent itemsets output by LCM.
#       b) VALID   — itemsets passing the trajectory-validity check
#                    (numedges == n*(n-1)/2).
#       c) MAXIMAL — valid itemsets that are also maximal.
#  4. Produce a summary CSV, a detailed JSON, and (optionally) plots.
#
# Usage:
#  python3 analyze_spurious_itemsets.py --npy ../data/breastCancer.npy --sigmas 2,3,4,5,6,8,10,15,20 --seed 0 --max_itemsets 5000000 --timeout 120 --mode weighted --outdir spurious_analysis
#
# Modes:
#  - 'weighted'   : use all mutation trees per patient with uniform weights
#  - 'unweighted' : sample one tree per patient at random
# =============================================================================

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from utils import (
    SCRIPT_DIR,
    EDGE_SEPARATORS,
    ensure_dir,
    load_npy,
    run_cmd,
    build_inputs,
    run_transnum,
    run_lcm_limited,
    parse_pattern_nodes_and_edges,
    is_valid_trajectory,
)



# =====================================================================
# Parsing and analysis of LCM output
# =====================================================================
def parse_lcm_output(lcm_out_path: Path):
    """Parse the label-converted LCM output file into structured records.

    The LCM output alternates between two line types:
      - Pattern line:  "edge1 edge2 ... (support)"  — the frequent itemset.
      - Occurrence line: "tid1 tid2 ..."             — IDs of supporting transactions.

    For each itemset, this function extracts the individual mutation nodes
    and directed edges using `parse_pattern_nodes_and_edges`, then checks
    whether the edge count matches a complete ordering (valid trajectory).

    Returns a list of dicts with keys:
      'line', 'items_raw', 'support', 'occ_ids', 'nodes',
      'numedges', 'n_nodes', 'is_valid'.
    """
    records = []
    if not lcm_out_path.exists():
        return records

    with lcm_out_path.open("r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            # Pattern lines contain parenthesized support; occurrence lines do not
            if "(" in line:
                m = re.search(r"\((\d+(?:\.\d+)?)\)", line)
                sup = float(m.group(1)) if m else 0

                # Strip the "(support)" portion to isolate the space-separated items
                items_str = re.sub(r"\(.*?\)", "", line).strip()
                items_raw = [tok for tok in items_str.split() if tok.strip()]

                # Decompose items into unique mutation nodes and pairwise ordering edges
                nodes, numedges = parse_pattern_nodes_and_edges(items_raw, EDGE_SEPARATORS)
                n = len(nodes)
                valid = is_valid_trajectory(nodes, numedges)

                # The next line lists the transaction IDs that support this itemset
                occ_line = f.readline()
                occ_ids = frozenset()
                if occ_line:
                    occ_ids = frozenset(
                        int(x) for x in occ_line.strip().split() if x.strip().isdigit()
                    )

                records.append({
                    "line": line,
                    "items_raw": items_raw,
                    "support": sup,
                    "occ_ids": occ_ids,
                    "nodes": nodes,
                    "numedges": numedges,
                    "n_nodes": n,
                    "is_valid": valid,
                })
            else:
                # Orphan occurrence line (no preceding pattern) — skip
                pass

    return records

def apply_maximality_filter(valid_records, support_tol=1e-9):
    """Keep only support-preserving maximal trajectories.

    A valid itemset A is discarded if there exists another valid itemset B such that:
      1. nodes(A) is a strict subset of nodes(B), and
      2. support(B) == support(A) up to numerical tolerance.
    """
    if not valid_records:
        return []

    # Sort larger patterns first, so potential dominators are encountered early
    ordered = sorted(
        valid_records,
        key=lambda r: (-r["n_nodes"], -r["support"])
    )

    maximal = []
    for rec in ordered:
        dominated = False
        for kept in maximal:
            same_support = abs(kept["support"] - rec["support"]) <= support_tol
            strict_superset = rec["nodes"] < kept["nodes"]   # strict subset
            if same_support and strict_superset:
                dominated = True
                break
        if not dominated:
            maximal.append(rec)

    return maximal

def analyze_itemsets(records):
    """Compute summary statistics for a single LCM run.

    Classifies every itemset as valid or invalid (spurious), applies the
    maximality filter to the valid subset, and collects:
      - Counts at each pipeline stage (raw -> valid -> maximal).
      - Breakdown of invalidity reasons (no edges, missing edges, etc.).
      - Size distributions (number of mutation nodes) for valid/invalid sets.

    Returns a flat dict suitable for CSV/JSON serialization.
    """
    n_total = len(records)
    valid = [r for r in records if r["is_valid"]]
    invalid = [r for r in records if not r["is_valid"]]
    maximal = apply_maximality_filter(valid)

    # Categorize why each invalid itemset fails the trajectory check:
    # a complete ordering of n nodes requires exactly n*(n-1)/2 edges.
    invalid_reasons = defaultdict(int)
    for r in invalid:
        n = r["n_nodes"]
        expected = n * (n - 1) / 2
        actual = r["numedges"]
        if actual == 0:
            invalid_reasons["no_edges"] += 1
        elif actual < expected:
            invalid_reasons["missing_edges"] += 1
        elif actual > expected:
            invalid_reasons["too_many_edges"] += 1
        else:
            invalid_reasons["other"] += 1

    # Histogram of itemset sizes (number of distinct mutation nodes)
    size_dist_all = defaultdict(int)
    size_dist_valid = defaultdict(int)
    size_dist_invalid = defaultdict(int)

    for r in records:
        size_dist_all[r["n_nodes"]] += 1
    for r in valid:
        size_dist_valid[r["n_nodes"]] += 1
    for r in invalid:
        size_dist_invalid[r["n_nodes"]] += 1

    return {
        "n_raw": n_total,
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "n_maximal": len(maximal),
        "pct_spurious": round(100.0 * len(invalid) / max(1, n_total), 2),
        "pct_removed_by_maximality": round(
            100.0 * (len(valid) - len(maximal)) / max(1, len(valid)), 2
        ),
        "pct_final_vs_raw": round(100.0 * len(maximal) / max(1, n_total), 2),
        "invalid_reasons": dict(invalid_reasons),
        "size_dist_all": dict(sorted(size_dist_all.items())),
        "size_dist_valid": dict(sorted(size_dist_valid.items())),
        "size_dist_invalid": dict(sorted(size_dist_invalid.items())),
    }


# =====================================================================
# End-to-end analysis pipeline for a single support threshold
# =====================================================================
def run_analysis_for_sigma(sigma, mode, lcmdir, workdir,
                           file_graphs_ids, table_file_ids,
                           weights_txt=None,
                           max_itemsets=None, timeout_sec=None):
    """Run the full analysis pipeline for one support value (sigma).

    Steps:
      1. Execute LCM on the numeric-ID transaction file with the given
         support threshold, optional itemset cap, and optional timeout.
      2. Convert the numeric LCM output back to human-readable mutation
         labels using convert_results.py and the mapping table.
      3. Parse the converted output into structured records and compute
         spurious-itemset statistics via `analyze_itemsets`.

    Returns a dict merging LCM execution metadata (timing, completion
    status) with the analysis statistics, or None if LCM produced no output.
    """
    tag = f"sigma{sigma}"
    ensure_dir(workdir)

    output_lcm = workdir / f"{tag}_lcm_raw.txt"
    log_path = workdir / f"{tag}_lcm.log"
    results_converted = workdir / f"{tag}_converted.txt"

    # Step 1: Run LCM frequent-itemset mining
    lcm_info = run_lcm_limited(
        lcmdir, file_graphs_ids, sigma, output_lcm, log_path,
        weights_txt=weights_txt,
        max_itemsets=max_itemsets,
        timeout_sec=timeout_sec,
    )

    if not output_lcm.exists():
        print(f"  [WARN] No LCM output for sigma={sigma}")
        return None

    raw_line_count = sum(1 for _ in output_lcm.open("r"))

    # Step 2: Map numeric item IDs back to mutation-edge labels
    run_cmd(["python3", str(SCRIPT_DIR / "convert_results.py"),
         "-m", str(table_file_ids),
         "-i", str(output_lcm),
         "-o", str(results_converted)])

    # Step 3: Parse converted output and compute spurious-itemset statistics
    records = parse_lcm_output(results_converted)
    stats = analyze_itemsets(records)

    # Record output file sizes for the summary report
    raw_file_size_mb = output_lcm.stat().st_size / (1024 * 1024)
    conv_file_size_mb = results_converted.stat().st_size / (1024 * 1024) if results_converted.exists() else 0

    result = {
        "sigma": sigma,
        "mode": mode,
        **lcm_info,
        "raw_file_lines": raw_line_count,
        "raw_file_size_mb": round(raw_file_size_mb, 2),
        "converted_file_size_mb": round(conv_file_size_mb, 2),
        **stats,
    }

    print(f"\n  === sigma={sigma} ===")
    print(f"  LCM: {lcm_info['elapsed_s']}s | completed={lcm_info['completed']} | timed_out={lcm_info['timed_out']}")
    print(f"  Raw file: {raw_file_size_mb:.2f} MB ({raw_line_count} lines)")
    print(f"  Total itemsets (raw):    {stats['n_raw']}")
    print(f"  Valid trajectories:      {stats['n_valid']}")
    print(f"  Spurious itemsets:       {stats['n_invalid']}  ({stats['pct_spurious']}%)")
    print(f"  After maximality:        {stats['n_maximal']}")
    print(f"  Final / raw:             {stats['pct_final_vs_raw']}%")

    return result


# =====================================================================
# Plotting
# =====================================================================
def plot_results(results, outdir: Path):
    """Generate summary plots of the spurious-itemset analysis.

    Produces two figures saved as PNG files:
      1. `spurious_analysis.png` — a 2x2 grid showing:
         (a) Itemset counts at each pipeline stage (log scale) vs. sigma.
         (b) Percentage of spurious itemsets vs. sigma.
         (c) LCM wall-clock time vs. sigma (red bars = timed out).
         (d) Raw LCM output file size vs. sigma.
      2. `spurious_by_size.png` — side-by-side bar charts comparing the
         size distribution (number of mutation nodes) of valid vs. spurious
         itemsets for a representative subset of sigma values.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")
        return

    results = [r for r in results if r is not None]
    if not results:
        return

    sigmas = [r["sigma"] for r in results]
    n_raw = [r["n_raw"] for r in results]
    n_valid = [r["n_valid"] for r in results]
    n_maximal = [r["n_maximal"] for r in results]
    n_invalid = [r["n_invalid"] for r in results]
    pct_spurious = [r["pct_spurious"] for r in results]
    elapsed = [r["elapsed_s"] for r in results]
    file_sizes = [r["raw_file_size_mb"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Itemset counts at each filtering stage (log-scale y-axis)
    ax = axes[0, 0]
    ax.plot(sigmas, n_raw, "o-", label="Raw (LCM)", color="tab:red", linewidth=2)
    ax.plot(sigmas, n_valid, "s-", label="Valid trajectories", color="tab:blue", linewidth=2)
    ax.plot(sigmas, n_maximal, "^-", label="Maximal", color="tab:green", linewidth=2)
    ax.set_xlabel("Support (sigma)")
    ax.set_ylabel("# Itemsets")
    ax.set_title("Itemsets per pipeline stage")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Percentage of spurious (non-trajectory) itemsets per sigma
    ax = axes[0, 1]
    ax.bar(range(len(sigmas)), pct_spurious, color="tab:orange", alpha=0.8)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels(sigmas)
    ax.set_xlabel("Support (sigma)")
    ax.set_ylabel("% Spurious itemsets")
    ax.set_title("Percentage of spurious itemsets (non-trajectories)")
    ax.grid(True, alpha=0.3, axis="y")

    # (c) LCM execution time; red bars indicate runs that hit the timeout
    ax = axes[1, 0]
    colors = ["tab:red" if r["timed_out"] else "tab:blue" for r in results]
    ax.bar(range(len(sigmas)), elapsed, color=colors, alpha=0.8)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels(sigmas)
    ax.set_xlabel("Support (sigma)")
    ax.set_ylabel("Time (s)")
    ax.set_title("LCM execution time (red = timed out)")
    ax.grid(True, alpha=0.3, axis="y")

    # (d) Raw LCM output file size (proxy for search-space explosion)
    ax = axes[1, 1]
    ax.plot(sigmas, file_sizes, "D-", color="tab:purple", linewidth=2)
    ax.set_xlabel("Support (sigma)")
    ax.set_ylabel("File size (MB)")
    ax.set_title("Raw LCM output size")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = outdir / "spurious_analysis.png"
    plt.savefig(fig_path, dpi=150)
    print(f"[OK] Plot saved: {fig_path}")
    plt.close()

    # Second figure: valid vs. spurious breakdown by itemset size (# nodes)

    # Select up to 4 representative sigma values (skip timed-out runs)
    interesting = [r for r in results if not r["timed_out"]]
    if len(interesting) > 4:
        interesting = [interesting[0], interesting[len(interesting) // 3],
                       interesting[2 * len(interesting) // 3], interesting[-1]]

    n_panels = max(len(interesting), 1)
    ncols = min(n_panels, 2)
    nrows = (n_panels + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                                squeeze=False)

    width = 0.35
    for idx, r in enumerate(interesting):
        sizes_valid = r.get("size_dist_valid", {})
        sizes_invalid = r.get("size_dist_invalid", {})
        all_sizes = sorted(set(list(sizes_valid.keys()) + list(sizes_invalid.keys())))
        if not all_sizes:
            continue
        x_pos = np.arange(len(all_sizes))
        vals_v = [sizes_valid.get(s, 0) for s in all_sizes]
        vals_i = [sizes_invalid.get(s, 0) for s in all_sizes]

        ax_sub = axes2[idx // ncols, idx % ncols]
        ax_sub.bar(x_pos - width / 2, vals_v, width, label="Valid", color="tab:blue", alpha=0.7)
        ax_sub.bar(x_pos + width / 2, vals_i, width, label="Spurious", color="tab:red", alpha=0.7)
        ax_sub.set_xticks(x_pos)
        ax_sub.set_xticklabels([str(s) for s in all_sizes])
        ax_sub.set_xlabel("# Nodes")
        ax_sub.set_ylabel("# Itemsets")
        ax_sub.set_title(f"sigma = {r['sigma']}")
        ax_sub.legend(fontsize=8)
        ax_sub.grid(True, alpha=0.3, axis="y")

    # Hide unused subplots
    for idx in range(len(interesting), nrows * ncols):
        axes2[idx // ncols, idx % ncols].set_visible(False)

    fig2.tight_layout()
    fig2_path = outdir / "spurious_by_size.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"[OK] Plot saved: {fig2_path}")
    plt.close(fig2)


# =====================================================================
# Main entry point
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Quantify how many LCM frequent itemsets are spurious (non-trajectories).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--npy", required=True,
                    help="Path to the breastCancer.npy dataset file")
    ap.add_argument("--sigmas", default="2,3,4,5,6,8,10,15,20",
                    help="Comma-separated support thresholds to test")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for tree sampling (unweighted mode)")
    ap.add_argument("--mode", choices=["weighted", "unweighted"], default="weighted",
                    help="'weighted' (Alg 1, all trees with weights) or "
                         "'unweighted' (Alg 0, one sampled tree per patient)")
    ap.add_argument("--max_itemsets", type=int, default=None,
                    help="Cap on the number of itemsets LCM may output (-# flag). "
                         "None means no cap.")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Wall-clock timeout in seconds for each LCM invocation. "
                         "None means no timeout.")
    ap.add_argument("--lcmdir", default="./lcm53",
                    help="Path to the lcm53 binary directory")
    ap.add_argument("--outdir", default="spurious_analysis",
                    help="Output directory for results, plots, and intermediate files")
    ap.add_argument("--keep_gl", action="store_true",
                    help="Retain the germline (GL) node in transactions")
    ap.add_argument("--reuse_inputs", default=None,
                    help="Path to a previously prepared input directory (skips build_inputs)")
    ap.add_argument("--no_plot", action="store_true",
                    help="Disable plot generation")

    args = ap.parse_args()

    lcmdir = Path(args.lcmdir).resolve()
    npy_path = Path(args.npy).resolve()
    outdir = Path(args.outdir).resolve()
    drop_gl = not args.keep_gl

    ensure_dir(outdir)

    sigmas = [float(x.strip()) if "." in x.strip() else int(x.strip())
              for x in args.sigmas.split(",") if x.strip()]

    print("=" * 60)
    print("Spurious Itemset Analysis — MASTRO")
    print("=" * 60)
    print(f"  Dataset:       {npy_path}")
    print(f"  Mode:          {args.mode}")
    print(f"  Sigmas:        {sigmas}")
    print(f"  Max itemsets:  {args.max_itemsets or 'no limit'}")
    print(f"  Timeout:       {args.timeout or 'no timeout'}s")
    print(f"  Output:        {outdir}")
    print()

    # --- Input preparation: build or reuse transaction files ---
    if args.reuse_inputs:
        inputs_dir = Path(args.reuse_inputs).resolve()
        print(f"[INFO] Reusing pre-built inputs from: {inputs_dir}")
        graphs_all = inputs_dir / "graphs_all.txt"
        weights_uniform = inputs_dir / "weights_uniform.txt"
        graphs_sampled = inputs_dir / "graphs_sampled.txt"
    else:
        print("[STEP] Loading dataset and building transaction inputs...")
        data = load_npy(npy_path)
        patients_trees = list(data)

        inputs_dir = outdir / "inputs"
        ensure_dir(inputs_dir)
        # build_inputs writes graphs_all.txt (all trees), weights_uniform.txt,
        # and graphs_sampled.txt (one tree per patient) to the inputs directory.
        graphs_all, weights_uniform, _, graphs_sampled = build_inputs(
            patients_trees, inputs_dir, seed=args.seed, drop_gl=drop_gl,
        )
        print(f"  [OK] Inputs written to: {inputs_dir}")

    # In weighted mode, LCM receives all trees with per-tree weights;
    # in unweighted mode, only the sampled subset is used (no weights).
    if args.mode == "weighted":
        graphs_txt = graphs_all
        weights_txt = weights_uniform
    else:
        graphs_txt = graphs_sampled
        weights_txt = None

    # Convert human-readable edge labels to numeric IDs (done once;
    # the label-to-ID mapping is reused across all sigma values).
    print("[STEP] Converting labels to numeric IDs (transnum)...")
    table_file_ids = lcmdir / f"table-file-spurious_{graphs_txt.name}"
    file_graphs_ids = lcmdir / f"lcm-out-spurious_{graphs_txt.stem}_ids.txt"
    run_transnum(lcmdir, table_file_ids, graphs_txt, file_graphs_ids)
    print(f"  [OK] table-file: {table_file_ids}")
    print(f"  [OK] graphs_ids: {file_graphs_ids}")

    # --- Run analysis for each sigma (highest first — they are fastest) ---
    results = []
    for sigma in sorted(sigmas, reverse=True):
        print(f"\n{'—' * 50}")
        print(f"[STEP] Analysis for sigma = {sigma}")
        print(f"{'—' * 50}")

        workdir = outdir / f"run_sigma_{sigma}"

        r = run_analysis_for_sigma(
            sigma=sigma,
            mode=args.mode,
            lcmdir=lcmdir,
            workdir=workdir,
            file_graphs_ids=file_graphs_ids,
            table_file_ids=table_file_ids,
            weights_txt=weights_txt,
            max_itemsets=args.max_itemsets,
            timeout_sec=args.timeout,
        )
        if r is not None:
            results.append(r)

    # Sort results by ascending sigma for readability
    results.sort(key=lambda x: x["sigma"])

    # --- Write summary CSV ---
    csv_path = outdir / "spurious_summary.csv"
    fieldnames = [
        "sigma", "mode", "completed", "timed_out", "capped",
        "elapsed_s", "max_itemsets",
        "raw_file_lines", "raw_file_size_mb", "converted_file_size_mb",
        "n_raw", "n_valid", "n_invalid", "n_maximal",
        "pct_spurious", "pct_removed_by_maximality", "pct_final_vs_raw",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n[OK] Summary CSV: {csv_path}")

    # --- Write detailed JSON (includes size distributions and invalidity reasons) ---
    json_path = outdir / "spurious_details.json"
    # Convert frozenset/set fields to lists for JSON serialization
    serializable_results = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (set, frozenset)):
                sr[k] = list(v)
            else:
                sr[k] = v
        serializable_results.append(sr)
    json_path.write_text(json.dumps(serializable_results, indent=2))
    print(f"[OK] Detailed JSON: {json_path}")

    # --- Print summary table to stdout ---
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    header = f"{'sigma':>6} {'Time':>8} {'Raw':>10} {'Valid':>10} {'Spurious':>10} {'%Spur.':>8} {'Maximal':>10} {'%Final':>8} {'MB':>8}"
    print(header)
    print("-" * 90)
    for r in results:
        status = "T" if r["timed_out"] else ("C" if r["capped"] else " ")
        print(f"{r['sigma']:>5}{status} {r['elapsed_s']:>7.1f}s {r['n_raw']:>10,} "
              f"{r['n_valid']:>10,} {r['n_invalid']:>10,} {r['pct_spurious']:>7.1f}% "
              f"{r['n_maximal']:>10,} {r['pct_final_vs_raw']:>7.1f}% "
              f"{r['raw_file_size_mb']:>7.1f}")
    print("-" * 90)
    print("  T = timed out | C = capped (-# flag)")

    # --- Generate plots (unless --no_plot was specified) ---
    if not args.no_plot:
        print("\n[STEP] Generating plots...")
        plot_results(results, outdir)

    print(f"\n[DONE] All results saved to: {outdir}")


if __name__ == "__main__":
    main()

"""Time POTTR's trajectory extraction, stage by stage, for a single k.

This is the POTTR half of the timing comparison. It does not reimplement
POTTR: it imports run_POTTR and wraps the functions that module calls with
timers, then invokes run_POTTR.main() unchanged. The measured code path is
therefore POTTR's own, and the stage boundaries cannot drift out of sync with
it the way a copied pipeline would.

Two deliberate deviations from a plain run_POTTR invocation, both so that the
two sides of the comparison measure the same thing:

  * POTTR's significance test is stubbed out. run_POTTR runs it inline
    whenever the largest trajectory has fewer than 13 nodes, and there is no
    flag to disable it. The comparison is about extracting the frequent
    trajectories; neither side is timed with its significance test.
  * Nothing else is touched. The GEXF serialisation stays in and is counted,
    because it is part of what the tool does to produce its output.

Stages timed:

    read_dags     parse the DAG files into a dict of graphs per patient
    conflict      pairwise conflict graphs + their union   <- the O(T^2) stage
    ilp           Gurobi: maximum k-common trajectory
    support       recount support of the selected trajectories
    convert       write the MASTRO-format output

Usage:
    python3 pottr_timing_run.py --pottr_repo ../../POTTR \\
        --dags /abs/path/to/dags -k 5 -c 20 \\
        --output_path results/timing/cap2/pottr_k5 \\
        --out results/timing/cap2/pottr_k5.json
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pottr_repo", required=True, help="POTTR repo root (holds code/)")
    ap.add_argument("--dags", required=True, help="Directory of transitively closed DAG files")
    ap.add_argument("-k", type=int, required=True, help="Recurrence level")
    ap.add_argument("-c", "--cores", type=int, default=1, help="Threads for Gurobi and graph building")
    ap.add_argument("--parallel", action="store_true", help="Pass -parallel to POTTR")
    ap.add_argument("--pottr_verbose", action="store_true",
                    help="Pass -v to POTTR, which also turns on the Gurobi MIP log")
    ap.add_argument("--output_path", required=True, help="POTTR output directory for this k")
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--label", default="", help="Free-form tag copied into the JSON")
    args = ap.parse_args()

    code_dir = Path(args.pottr_repo).resolve() / "code"
    if not (code_dir / "run_POTTR.py").exists():
        raise SystemExit(f"[err] run_POTTR.py not found under {code_dir}")

    dags = Path(args.dags).resolve()
    outdir = Path(args.output_path).resolve()
    out_json = Path(args.out).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    n_dags = len(list(dags.glob("*.txt")))

    # POTTR resolves its imports relative to code/ and writes scratch files into
    # the cwd, so run from there exactly as run_pottr_breastcancer.sh does.
    record_base = {
        "method": "pottr",
        "label": args.label,
        "k": args.k,
        "n_dags": n_dags,
        "threads": args.cores,
        "parallel": args.parallel,
        "verbose": args.pottr_verbose,
    }

    sys.path.insert(0, str(code_dir))
    os.chdir(code_dir)

    import run_POTTR

    stages = []
    t_total = time.time()
    state = {"running": None, "started_at": None}

    def snapshot(status):
        """Write the record as it stands, so a kill still leaves evidence.

        POTTR can exceed any wall-clock budget on a large cohort. Without this
        the only thing a timeout leaves behind is the fact that it timed out,
        and the question the experiment exists to answer - whether the cost is
        the O(T^2) conflict stage or the ILP - stays unanswered.
        """
        partial = dict(record_base)
        partial["stages"] = list(stages)
        partial["status"] = status
        partial["completed"] = status == "ok"
        partial["total_s"] = round(time.time() - t_total, 3)
        if state["running"] is not None:
            partial["interrupted_in"] = state["running"]
            partial["interrupted_after_s"] = round(time.time() - state["started_at"], 3)
        tmp = out_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(partial, indent=2))
        os.replace(tmp, out_json)

    def on_signal(signum, _frame):
        snapshot("timeout" if signum == signal.SIGTERM else "interrupted")
        print(f"[partial] killed by signal {signum} during stage "
              f"{state['running']}; partial timings written to {out_json}")
        os._exit(124 if signum == signal.SIGTERM else 130)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def timed(label, fn):
        """Wrap fn so each call records its elapsed time and flushes a snapshot."""
        def wrapper(*a, **kw):
            state["running"] = label
            state["started_at"] = time.time()
            t0 = time.time()
            result = fn(*a, **kw)
            stages.append({"stage": label, "elapsed_s": round(time.time() - t0, 3)})
            state["running"] = None
            snapshot("running")
            return result
        return wrapper

    # The union conflict graph is built by two calls in the parallel branch and
    # one in the serial branch; both land under the same 'conflict' label, so
    # the totals are comparable across -parallel settings.
    run_POTTR.read_multiple_graphs_per_evolution = timed(
        "read_dags", run_POTTR.read_multiple_graphs_per_evolution)
    run_POTTR.get_conflict_graphs_parallel = timed(
        "conflict", run_POTTR.get_conflict_graphs_parallel)
    run_POTTR.get_conflict_graphs_single_thread = timed(
        "conflict", run_POTTR.get_conflict_graphs_single_thread)
    run_POTTR.get_union_conflict_graph = timed(
        "conflict", run_POTTR.get_union_conflict_graph)
    run_POTTR.POTTR.find_max_k_common_trajectory = timed(
        "ilp", run_POTTR.POTTR.find_max_k_common_trajectory)
    run_POTTR.compute_support.compute_support = timed(
        "support", run_POTTR.compute_support.compute_support)
    run_POTTR.convert_to_mastro_format.convert = timed(
        "convert", run_POTTR.convert_to_mastro_format.convert)

    # Stub the significance test: not part of what is being compared.
    def _skip_significance(*a, **kw):
        stages.append({"stage": "significance_SKIPPED", "elapsed_s": 0.0})
        return None
    run_POTTR.compute_significance.run_stat_significancce_test = _skip_significance

    argv = ["run_POTTR.py", "-o", str(outdir), "-d", str(dags),
            "-k", str(args.k), "-c", str(args.cores)]
    if args.parallel:
        argv.append("-parallel")
    if args.pottr_verbose:
        argv.append("-v")
    sys.argv = argv

    snapshot("running")
    trajectory_size = run_POTTR.main()
    total = round(time.time() - t_total, 3)

    converted = outdir / "converted_graphs.txt"
    n_traj = sum(1 for line in open(converted) if line.strip()) if converted.exists() else 0

    record = dict(record_base)
    record.update({
        "stages": stages,
        "total_s": total,
        "largest_trajectory_nodes": trajectory_size,
        "n_output_lines": n_traj,
        "status": "ok",
        "completed": True,
    })
    out_json.write_text(json.dumps(record, indent=2))
    print(f"[OK] {out_json}  total {total}s  largest trajectory {trajectory_size} nodes")
    for s in stages:
        print(f"      {s['stage']:<22} {s['elapsed_s']:>10.3f}s")


if __name__ == "__main__":
    main()

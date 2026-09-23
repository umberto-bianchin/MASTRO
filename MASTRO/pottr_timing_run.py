"""Time POTTR's trajectory extraction, stage by stage, for a single k.

This is the POTTR half of the timing comparison. It does not reimplement
POTTR: it imports run_POTTR and wraps the functions that module calls with
timers, then invokes run_POTTR.main() unchanged. The measured code path is
therefore POTTR's own, and the stage boundaries cannot drift out of sync with
it the way a copied pipeline would.

Two deliberate deviations from a plain run_POTTR invocation, both so that the
two sides of the comparison measure the same thing:

  * The solution pool is exposed as --pool. run_POTTR's CLI defaults it to 0,
    which makes Gurobi return a single optimal solution; the POTTR paper's own
    comparison against MASTRO instead sets PoolSearchMode=2 with
    PoolSolutions=50000 to collect the co-optimal trajectories. Those are two
    different experiments and the timings are not interchangeable: with the
    pool off, a run reports one arbitrary member of the optimal set, and
    repeated runs can report different ones.
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
    dedup         drop duplicate solutions (pairwise isomorphism, O(n^2))
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
import re
import resource
import signal
import threading
import sys
import time
from pathlib import Path


def tree_rss_mb():
    """Resident memory of this process and its descendants, in MB.

    Read from /proc so it covers live worker processes: getrusage only reports
    children that have already been reaped, which is useless while a pool is
    running. Returns None where /proc is not available.

    NOTE this SUMS per-process RSS, and forked workers share pages
    copy-on-write, so shared pages are counted once per process. The result is
    an upper bound on physical memory, not a measurement of it - useful for
    seeing a blow-up, not for quoting a footprint.
    """
    try:
        def rss(pid):
            with open(f"/proc/{pid}/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

        def kids(pid):
            out = []
            tdir = f"/proc/{pid}/task"
            for tid in os.listdir(tdir):
                with open(f"{tdir}/{tid}/children") as f:
                    out += f.read().split()
            return out

        total, stack = 0, [str(os.getpid())]
        while stack:
            pid = stack.pop()
            try:
                total += rss(pid)
                stack += kids(pid)
            except (OSError, ValueError):
                continue  # process exited between listing and reading
        return round(total / 1e6, 1)
    except Exception:
        return None


def total_cpu():
    """User+system CPU seconds of this process and its children.

    POTTR runs in-process, so Gurobi's threads land in RUSAGE_SELF, while the
    parallel conflict-graph stage forks workers that land in RUSAGE_CHILDREN.
    Both are needed to see how many cores a stage actually used.
    """
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def read_trajectories(path):
    """Summarise what POTTR actually returned.

    converted_graphs.txt alternates a trajectory line ending in a parenthesised
    support with a line listing the occurrences that carry it. Counting lines
    therefore double-counts; and because the occurrence ids are per TREE
    ("P<patient>-<tree>"), the support POTTR reports is a count of trees, not of
    patients. Both numbers are recorded so the difference is visible.
    """
    if not path.exists():
        return {"n_trajectories": 0, "trajectories": []}
    lines = [l.strip() for l in open(path) if l.strip()]
    out = []
    for i, line in enumerate(lines):
        m = re.search(r"\((\d+)\)$", line)
        if not m:
            continue
        occ = lines[i + 1].split() if i + 1 < len(lines) else []
        patients = {o.rsplit("-", 1)[0] for o in occ}
        out.append({
            "items": line[:m.start()].strip(),
            "support": int(m.group(1)),
            "n_occurrences": len(occ),
            "n_patients": len(patients),
        })
    return {"n_trajectories": len(out), "trajectories": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pottr_repo", required=True, help="POTTR repo root (holds code/)")
    ap.add_argument("--dags", required=True, help="Directory of transitively closed DAG files")
    ap.add_argument("-k", type=int, required=True, help="Recurrence level")
    ap.add_argument("-c", "--cores", type=int, default=1, help="Threads for Gurobi and graph building")
    ap.add_argument("--parallel", action="store_true", help="Pass -parallel to POTTR")
    ap.add_argument("--pool", type=int, default=0,
                    help="Gurobi solution pool size passed to POTTR (-pool). 0 keeps "
                         "run_POTTR's default of a single optimal solution; the POTTR "
                         "paper used 50000 for its comparison against MASTRO")
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
        "solution_pool": args.pool,
    }

    sys.path.insert(0, str(code_dir))
    os.chdir(code_dir)

    import run_POTTR

    stages = []
    t_total = time.time()
    cpu_total0 = total_cpu()
    state = {"running": None, "started_at": None, "error": None, "peak_rss_mb": 0}

    snapshot_lock = threading.Lock()

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
        partial["cpu_total_s"] = round(total_cpu() - cpu_total0, 3)
        if state["running"] is not None:
            partial["interrupted_in"] = state["running"]
            partial["interrupted_after_s"] = round(time.time() - state["started_at"], 3)
        if state.get("error"):
            partial["error"] = state["error"]
        if state["peak_rss_mb"]:
            partial["peak_rss_mb"] = state["peak_rss_mb"]
            partial["rss_now_mb"] = tree_rss_mb()
        # The sampler thread and the main thread both snapshot. Without the
        # lock they race on one shared temp path and whichever renames second
        # finds the file already gone, killing the sampler; the pid keeps the
        # temp names distinct in case anything else writes here too.
        tmp = out_json.with_suffix(f".json.tmp{os.getpid()}")
        with snapshot_lock:
            tmp.write_text(json.dumps(partial, indent=2))
            os.replace(tmp, out_json)

    def on_signal(signum, _frame):
        snapshot("timeout" if signum == signal.SIGTERM else "interrupted")
        print(f"[partial] killed by signal {signum} during stage "
              f"{state['running']}; partial timings written to {out_json}")
        os._exit(124 if signum == signal.SIGTERM else 130)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    # Sample memory and re-snapshot on a timer. A stage that dies to an
    # external SIGKILL - which is what an out-of-memory kill looks like, and is
    # neither catchable nor loggable - leaves no record of its own, so the last
    # periodic snapshot is the only evidence of how far it got and how much it
    # was holding when it went.
    def sampler():
        while not stop_sampling.is_set():
            try:
                rss = tree_rss_mb()
                if rss:
                    state["peak_rss_mb"] = max(state["peak_rss_mb"], rss)
                if state["running"]:
                    snapshot("running")
            except Exception as exc:
                # Never let the sampler die: losing it means losing the only
                # evidence of how far a stage got before an uncatchable kill.
                print(f"[sampler] {type(exc).__name__}: {exc}", file=sys.stderr)
            stop_sampling.wait(15)

    stop_sampling = threading.Event()
    threading.Thread(target=sampler, daemon=True).start()

    def timed(label, fn):
        """Wrap fn so each call records its elapsed time and flushes a snapshot."""
        def wrapper(*a, **kw):
            state["running"] = label
            state["started_at"] = time.time()
            c0, t0 = total_cpu(), time.time()
            result = fn(*a, **kw)
            wall = time.time() - t0
            cpu = total_cpu() - c0
            stages.append({"stage": label, "elapsed_s": round(wall, 3),
                           "cpu_s": round(cpu, 3),
                           "cpu_per_wall": round(cpu / wall, 2) if wall > 0.01 else None})
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
    # Deduplication is O(n^2) isomorphism tests over the returned solutions, so
    # with a large --pool it can dwarf the ILP. Untimed it would appear as an
    # unexplained gap between the ilp and support stages.
    run_POTTR.filter_duplicates = timed("dedup", run_POTTR.filter_duplicates)
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
            "-k", str(args.k), "-c", str(args.cores),
            "-pool", str(args.pool)]
    if args.parallel:
        argv.append("-parallel")
    if args.pottr_verbose:
        argv.append("-v")
    sys.argv = argv

    snapshot("running")
    try:
        trajectory_size = run_POTTR.main()
    except BaseException as exc:
        # A stage can die on its own - a worker pool killed for running out of
        # memory, a solver failure - and without this the record would stay at
        # "running" and the stage it died in would be lost with the process.
        state["error"] = f"{type(exc).__name__}: {exc}"
        snapshot("error")
        print(f"[ERROR] {out_json}  failed during stage {state['running']} "
              f"after {round(time.time() - (state['started_at'] or t_total), 1)}s: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    stop_sampling.set()
    total = round(time.time() - t_total, 3)
    found = read_trajectories(outdir / "converted_graphs.txt")

    record = dict(record_base)
    record.update({
        "stages": stages,
        "total_s": total,
        "cpu_total_s": round(total_cpu() - cpu_total0, 3),
        "cpu_per_wall": round((total_cpu() - cpu_total0) / max(total, 1e-9), 2),
        "peak_rss_mb": state["peak_rss_mb"] or None,
        "largest_trajectory_nodes": trajectory_size,
        "status": "ok",
        "completed": True,
        **found,
    })
    out_json.write_text(json.dumps(record, indent=2))
    print(f"[OK] {out_json}  total {total}s  "
          f"{found['n_trajectories']} trajectory(ies), largest {trajectory_size} nodes")
    for tr in found["trajectories"]:
        print(f"      {tr['items']}  support {tr['support']}  "
              f"distinct patients {tr['n_patients']}")
    for s in stages:
        ratio = "" if s.get("cpu_per_wall") is None else f"  {s['cpu_per_wall']:>6.2f}x"
        print(f"      {s['stage']:<22} {s['elapsed_s']:>10.3f}s{ratio}")


if __name__ == "__main__":
    main()

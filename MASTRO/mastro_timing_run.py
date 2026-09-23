"""Time the Multi-MASTRO mining stages on a prepared input directory.

This is the MASTRO half of the POTTR timing comparison. It mines the
expected-support family, timing every stage separately and writing a JSON
record.

Two things are deliberately left out, so that both sides are measured on the
same task - extracting the frequent trajectories, which is all POTTR does:

  * Significance. Neither side is timed with its statistical test.
  * The theta post-filters. POTTR has no analogue of them, and timing them
    correctly would mean re-mining the candidates at sigma_exp =
    floor(theta * sigma) rather than reusing the family mined at sigma, which
    is what run_pipeline.py does and what the published theta families are
    built from. Use run_pipeline.py for theta families; this script is for
    the expected-support comparison only.

The input directory must hold the files written by build_inputs() or by
breastcancer_to_pottr.py --ensemble_out:

    graphs_all.txt        one transaction per candidate tree
    weights_uniform.txt   w_t = 1 / M_i
    owner.txt             patient index of each transaction

Stages timed, in order:

    transnum    edge labels -> numeric item ids (perl)
    lcm         weighted frequent itemset mining (C)
    convert     numeric ids -> edge labels
    filter      keep relation-complete, maximal itemsets

Mining is single-threaded by construction: LCM is a serial C program and both
post-processing stages are serial Python. The script pins OMP_NUM_THREADS=1 and
records it, so the comparison against a multi-core POTTR run is explicit rather
than implied.

Usage:
    python3 mastro_timing_run.py --inputs results/timing/cap2/inputs \\
        --sigma 5 --out results/timing/cap2/mastro_sigma5.json
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from utils import SCRIPT_DIR, ensure_dir, run_transnum


def child_cpu():
    """User+system CPU seconds consumed so far by child processes."""
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def _time(label, fn, stages):
    """Run fn(), record wall and CPU time, return its result.

    Every stage here is a subprocess, so CPU is taken from RUSAGE_CHILDREN.
    Recording it turns the thread count from a claim into a measurement: a
    serial stage spends about one CPU second per wall second, and anything
    using several cores spends proportionally more. Without this, comparing
    against a method that is given N threads rests on an assumption about this
    side that nobody checked.
    """
    c0, t0 = child_cpu(), time.time()
    result = fn()
    wall = time.time() - t0
    cpu = child_cpu() - c0
    stages.append({"stage": label, "elapsed_s": round(wall, 3),
                   "cpu_s": round(cpu, 3),
                   "cpu_per_wall": round(cpu / wall, 2) if wall > 0.01 else None})
    return result


def count_patterns(path: Path) -> int:
    """Number of trajectories in a filtered/post-filtered mining output.

    The file alternates a pattern line with a line of occurrence ids, so the
    pattern count is the number of lines that are not occurrence lines. A
    pattern line always ends in a parenthesised support value.
    """
    n = 0
    with open(path) as f:
        for line in f:
            if line.rstrip().endswith(")"):
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="Directory with graphs_all.txt, weights_uniform.txt, owner.txt")
    ap.add_argument("--sigma", type=float, required=True, help="Support threshold")
    ap.add_argument("--lcmdir", default="./lcm53")
    ap.add_argument("--workdir", default=None,
                    help="Scratch directory for LCM intermediates (default: <inputs>/../mastro_work)")
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--label", default="", help="Free-form tag copied into the JSON")
    args = ap.parse_args()

    # Mining is serial; make that explicit so a threaded BLAS cannot muddy the
    # measurement, and record it alongside the timings.
    os.environ["OMP_NUM_THREADS"] = "1"

    inputs = Path(args.inputs)
    lcmdir = Path(args.lcmdir)
    workdir = Path(args.workdir) if args.workdir else inputs.parent / "mastro_work"
    ensure_dir(workdir)

    graphs = inputs / "graphs_all.txt"
    weights = inputs / "weights_uniform.txt"
    owner = inputs / "owner.txt"
    for p in (graphs, weights, owner):
        if not p.exists():
            raise SystemExit(f"[err] missing input: {p}")

    n_trees = sum(1 for line in open(graphs) if line.strip())
    n_patients = len({line.strip() for line in open(owner) if line.strip()})

    table_file = workdir / "table-file.txt"
    ids_file = workdir / "lcm-out-ids.txt"
    lcm_out = workdir / "lcm-out.txt"
    converted = workdir / "expected_uniform_convres.txt"
    filtered = workdir / "expected_uniform_filtered.txt"
    lcm_log = workdir / "lcm.log"

    stages = []
    t_total = time.time()
    cpu_total0 = child_cpu()

    _time("transnum", lambda: run_transnum(lcmdir, table_file, graphs, ids_file), stages)

    # LCM in weighted mode: the epsilon keeps a trajectory whose expected
    # support is exactly sigma, which a strict '>' comparison would drop.
    def _lcm():
        cmd = [str(lcmdir / "lcm"), "FfI", "-w", str(weights),
               str(ids_file), str(args.sigma - 1e-9), str(lcm_out)]
        with open(lcm_log, "w") as logf:
            subprocess.run(cmd, stdout=logf, stderr=logf, check=True)
    _time("lcm", _lcm, stages)

    # sys.executable, not "python3": the stages must run under the same
    # interpreter as this script, or a PATH python3 without numpy fails here.
    _time("convert", lambda: subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "convert_results.py"),
         "-m", str(table_file), "-i", str(lcm_out), "-o", str(converted)],
        check=True, stdout=subprocess.DEVNULL), stages)

    _time("filter", lambda: subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "filter_results.py"),
         "-i", str(converted), "-o", str(filtered)],
        check=True, stdout=subprocess.DEVNULL), stages)

    n_exp = count_patterns(filtered)

    record = {
        "method": "multi-mastro",
        "label": args.label,
        "sigma": args.sigma,
        "n_patients": n_patients,
        "n_trees": n_trees,
        "stages": stages,
        "total_s": round(time.time() - t_total, 3),
        "cpu_total_s": round(child_cpu() - cpu_total0, 3),
        "cpu_per_wall": round((child_cpu() - cpu_total0) / max(time.time() - t_total, 1e-9), 2),
        "threads_note": ("mining is serial by construction: LCM is compiled without "
                         "MULTI_CORE, convert/filter/postfilter are serial Python. "
                         "cpu_per_wall near 1.0 is the measurement that confirms it."),
        "n_trajectories": {"exp": n_exp},
        "status": "ok",
        "completed": True,
    }

    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text(json.dumps(record, indent=2))
    print(f"[OK] {out}  total {record['total_s']}s "
          f"(cpu {record['cpu_total_s']}s, {record['cpu_per_wall']}x wall)  "
          f"exp-family {n_exp} trajectories")
    for s in stages:
        ratio = "" if s["cpu_per_wall"] is None else f"  {s['cpu_per_wall']:>5.2f}x"
        print(f"      {s['stage']:<22} {s['elapsed_s']:>10.3f}s{ratio}")


if __name__ == "__main__":
    main()

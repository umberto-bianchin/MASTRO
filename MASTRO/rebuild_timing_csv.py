"""Rebuild timing.csv from the per-run JSON records.

The driver used to append a row every time it visited a run, so re-invoking it
duplicated every row it skipped. Rebuilding from the JSONs instead makes the
CSV a function of the results on disk: idempotent, and correct after a resume.

Usage:
    python3 rebuild_timing_csv.py --outdir results/timing_pottr_vs_mastro
"""

import argparse
import csv
import json

from pottr_timing_run import read_trajectories
from pathlib import Path

HEADER = ["cohort_cap", "n_patients", "n_trees", "method", "threads",
          "param", "value", "total_s", "cpu_total_s", "cpu_per_wall", "status",
          "n_trajectories", "pottr_support", "pottr_patients", "detail"]


def pottr_output(cohort_dir: Path, threads, k):
    """What POTTR returned for one run, read from its own output file.

    The parsing rule lives in pottr_timing_run.read_trajectories and is shared
    rather than copied: an early version of the timing script recorded the
    number of LINES in converted_graphs.txt, which double-counts, and having
    two copies of the rule is how that kind of bug survives a fix.

    Recomputing here rather than trusting the JSON means records written by
    that older version are corrected in place. The support POTTR reports counts
    TREES, since occurrence ids are "P<patient>-<tree>", so the distinct-patient
    count is reported next to it.
    """
    f = cohort_dir / f"pottr_t{threads}" / f"k{k}" / "converted_graphs.txt"
    found = read_trajectories(f)
    if not found["n_trajectories"]:
        return "", "", ""
    tr = found["trajectories"]
    return (found["n_trajectories"],
            ";".join(str(x["support"]) for x in tr),
            ";".join(str(x["n_patients"]) for x in tr))


def cohort_size(cohort_dir: Path):
    """(patients, trees) of a cohort.

    Prefers the transaction files, and falls back to manifest.csv, which
    records one row per patient with its tree count. The fallback matters when
    results are copied off a server: the transaction files are large and are
    usually left behind, while the manifest is small enough to always travel.
    """
    owner = cohort_dir / "inputs" / "owner.txt"
    graphs = cohort_dir / "inputs" / "graphs_all.txt"
    if owner.exists() and graphs.exists():
        pats = len({l.strip() for l in open(owner) if l.strip()})
        trees = sum(1 for l in open(graphs) if l.strip())
        return pats, trees

    manifest = cohort_dir / "inputs" / "manifest.csv"
    if manifest.exists():
        rows = list(csv.DictReader(open(manifest)))
        return len(rows), sum(int(r["n_trees"]) for r in rows)
    return "", ""


def row_from(rec, cap, pats, trees, cohort_dir):
    stages = rec.get("stages", [])
    detail = ";".join(
        f"{s['stage']}={s['elapsed_s']}"
        + (f"@{s['cpu_per_wall']}x" if s.get("cpu_per_wall") else "")
        for s in stages)
    if rec.get("interrupted_in"):
        detail += f";KILLED_IN={rec['interrupted_in']}@{rec.get('interrupted_after_s')}s"
    status = rec.get("status") or ("ok" if rec.get("completed") else "timeout")
    # Interrupted records carry cpu_total_s but no ratio: the success path is
    # the only place that computes it. Derive it so the column is never blank
    # when the numbers to fill it are present.
    cpw = rec.get("cpu_per_wall")
    if cpw in (None, "") and rec.get("cpu_total_s") and rec.get("total_s"):
        cpw = round(rec["cpu_total_s"] / rec["total_s"], 2)
    if rec["method"] == "pottr":
        n, sup, tpats = pottr_output(cohort_dir, rec.get("threads", ""), rec.get("k", ""))
        return [cap, pats, trees, "pottr", rec.get("threads", ""), "k",
                rec.get("k", ""), rec.get("total_s", ""),
                rec.get("cpu_total_s", ""), cpw if cpw is not None else "", status,
                n, sup, tpats, detail]
    fam = rec.get("n_trajectories", {})
    return [cap, pats, trees, "multi-mastro", rec.get("threads", 1), "sigma",
            rec.get("sigma", ""), rec.get("total_s", ""),
            rec.get("cpu_total_s", ""), cpw if cpw is not None else "", status,
            fam.get("exp", ""), "", "", detail]


def sort_key(r):
    # cohort cap 0 means "no cap", i.e. the largest cohort: sort it last.
    cap = int(r[0]) if str(r[0]).isdigit() else 0
    return (cap if cap else 10 ** 9, r[3], int(r[4] or 0), float(r[6] or 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/timing_pottr_vs_mastro")
    args = ap.parse_args()
    out = Path(args.outdir)

    rows = []
    for cdir in sorted(out.glob("cap*")):
        if not cdir.is_dir():
            continue
        cap = cdir.name[3:]
        pats, trees = cohort_size(cdir)
        for j in sorted(cdir.glob("*.json")):
            try:
                rec = json.load(open(j))
            except json.JSONDecodeError:
                print(f"[warn] unreadable, skipped: {j}")
                continue
            if "method" not in rec:
                continue
            rows.append(row_from(rec, cap, pats, trees, cdir))

    rows.sort(key=sort_key)
    csv_path = out / "timing.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"[OK] {csv_path}: {len(rows)} rows")
    for r in rows:
        print("  " + " | ".join(str(x) for x in r[:10]))


if __name__ == "__main__":
    main()

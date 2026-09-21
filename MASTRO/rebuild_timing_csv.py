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
from pathlib import Path

HEADER = ["cohort_cap", "n_patients", "n_trees", "method", "threads",
          "param", "value", "total_s", "status", "n_trajectories", "detail"]


def cohort_size(cohort_dir: Path):
    """(patients, trees) of a cohort, read from the inputs it was built from."""
    owner = cohort_dir / "inputs" / "owner.txt"
    graphs = cohort_dir / "inputs" / "graphs_all.txt"
    if not owner.exists():
        return "", ""
    pats = len({l.strip() for l in open(owner) if l.strip()})
    trees = sum(1 for l in open(graphs) if l.strip())
    return pats, trees


def row_from(rec, cap, pats, trees):
    stages = rec.get("stages", [])
    detail = ";".join(f"{s['stage']}={s['elapsed_s']}" for s in stages)
    if rec.get("interrupted_in"):
        detail += f";KILLED_IN={rec['interrupted_in']}@{rec.get('interrupted_after_s')}s"
    status = rec.get("status") or ("ok" if rec.get("completed") else "timeout")
    if rec["method"] == "pottr":
        return [cap, pats, trees, "pottr", rec.get("threads", ""), "k",
                rec.get("k", ""), rec.get("total_s", ""), status,
                rec.get("n_output_lines", ""), detail]
    fam = rec.get("n_trajectories", {})
    return [cap, pats, trees, "multi-mastro", rec.get("threads", 1), "sigma",
            rec.get("sigma", ""), rec.get("total_s", ""), status,
            fam.get("exp", ""), detail]


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
            rows.append(row_from(rec, cap, pats, trees))

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

"""
Summarize POTTR vs Multi-MASTRO on null (permuted) cohorts.

Every input is the pottr_significance.py table of one null cohort (one seed of
permute_pottr_cohort.py): the trajectories POTTR returned over its k-sweep, with
POTTR's own p-value and the Multi-MASTRO p-values of the same trajectories.
Nothing in a null cohort is conserved, so every p-value below alpha is a false
discovery.

Both tests are applied to the SAME trajectories, so the comparison isolates the
test from the miner. Neither column is a calibrated multiple-testing procedure:
POTTR's ILP picks the most frequent trajectory, and an uncorrected p-value of a
selected pattern is optimistic under any test. The calibrated reference for
Multi-MASTRO is its full pipeline with the Westfall-Young threshold, which the
calibration experiment (empirical_fwer_ensemble.py) measures.

With --wy_name, the WY thresholds of each null cohort
(<seed dir>/<wy_name>/wy_thresholds.txt, from scripts/run_pottr_null_wy.sh)
are read as well and the Multi-MASTRO trajectories below them are counted:
these are the corrected false discoveries. The expected-support threshold
covers every trajectory with expected support >= sigma; the theta threshold
covers the theta-maximal family only and is reported for reference.

Output: one row per seed, then the fraction of seeds with at least one
discovery at each alpha (a FWER-like rate) for each test.

Usage:
    python3 summarize_pottr_null.py --csv results/pottr_null/seed*/significance.csv \\
        --out results/pottr_null/summary.csv
"""

import argparse
import csv
import re
from pathlib import Path

ALPHAS = (0.05, 0.01, 0.001)
TESTS = {"pottr": "pottr_pval_perm", "mm_exp": "mastro_pval_exp",
         "mm_theta": "mastro_pval_theta"}


def as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_wy(path):
    """{alpha: (threshold_exp, threshold_theta)} from a wy_thresholds.txt."""
    out = {}
    for line in path.read_text().splitlines():
        kv = dict(tok.split("=") for tok in line.split())
        out[float(kv["alpha"])] = (float(kv["threshold_exp"]),
                                   float(kv["threshold_theta"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True,
                    help="pottr_significance.py tables, one per null seed")
    ap.add_argument("--wy_name", default=None,
                    help="WY output folder inside each seed dir "
                         "(e.g. wy_sigma2_theta1.0); adds the corrected counts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for path in sorted(args.csv):
        m = re.search(r"seed(\d+)", path)
        seed = int(m.group(1)) if m else path
        table = list(csv.DictReader(open(path)))
        r = dict(seed=seed, n_traj=len(table))
        for test, col in TESTS.items():
            pv = [p for p in (as_float(t[col]) for t in table) if p is not None]
            r[f"{test}_n_pval"] = len(pv)
            r[f"{test}_min_p"] = min(pv) if pv else ""
            for a in ALPHAS:
                r[f"{test}_n_below_{a}"] = sum(p <= a for p in pv)
        if args.wy_name:
            wy_path = Path(path).parent / args.wy_name / "wy_thresholds.txt"
            wy = read_wy(wy_path) if wy_path.exists() else {}
            for a in ALPHAS:
                if a not in wy:
                    continue
                for test, thr in zip(("mm_exp", "mm_theta"), wy[a]):
                    pv = [p for p in (as_float(t[TESTS[test]]) for t in table)
                          if p is not None]
                    r[f"{test}_wy_threshold_{a}"] = thr
                    r[f"{test}_n_wy_{a}"] = sum(p <= thr for p in pv)
        rows.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    print(f"[null] {len(rows)} null cohorts -> {out}")
    for a in ALPHAS:
        parts = []
        for test in TESTS:
            hit = sum(r[f"{test}_n_below_{a}"] > 0 for r in rows)
            n = sum(r[f"{test}_n_below_{a}"] for r in rows)
            parts.append(f"{test}: {hit}/{len(rows)} cohorts ({n} trajectories)")
        print(f"  alpha={a:<6} " + " | ".join(parts))
    for a in ALPHAS:
        parts = []
        for test in ("mm_exp", "mm_theta"):
            have = [r for r in rows if f"{test}_n_wy_{a}" in r]
            if have:
                hit = sum(r[f"{test}_n_wy_{a}"] > 0 for r in have)
                n = sum(r[f"{test}_n_wy_{a}"] for r in have)
                parts.append(f"{test}: {hit}/{len(have)} cohorts ({n} trajectories)")
        if parts:
            print(f"  WY alpha={a:<6} " + " | ".join(parts))


if __name__ == "__main__":
    main()

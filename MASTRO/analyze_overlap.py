"""
Overlap of the significant trajectory sets of the two tests.

For each (cohort, sigma, theta) the script reads the per-trajectory p-values
written by the discovery pipeline and the Westfall-Young thresholds written by
run_wy_correction_ensemble.py, declares a trajectory significant when its
p-value is at or below the threshold of its own family, and compares the two
resulting sets:

    both       significant under expected support and under theta-consensus
    only exp   significant under expected support only  (Prop. FT_exp not subset FT_theta)
    only theta significant under theta-consensus only   (Prop. FT_theta not subset FT_exp)

Each test is scored on its own family, so the two thresholds differ: the
expected-support threshold is calibrated on the expected-support family and the
theta-consensus threshold on the theta family. Because a single WY run emits
both thresholds, the expected-support threshold is taken from one designated
run (--exp-threshold-from, theta=1.0 by default) so that the same expected-
support threshold is used for every theta.

Expected layout under --results:

    <cohort>_sigma<sigma>/significance/alg1_expected_uniform_pvalues_exp.csv
    <cohort>_sigma<sigma>/significance/alg3_theta<theta>_pvalues_theta.csv
    <cohort>_wy_sigma<sigma>_theta<theta>/wy_thresholds.txt

Usage:
    python3 analyze_overlap.py
    python3 analyze_overlap.py --cohorts breastCancer --alpha 0.05 --list-diff
"""

import argparse
import csv
import re
from pathlib import Path

THRESHOLD_RE = re.compile(
    r"alpha=(?P<alpha>[\d.]+)\s+threshold_exp=(?P<exp>\S+)\s+threshold_theta=(?P<theta>\S+)")


def read_thresholds(wy_dir, alpha):
    """wy_thresholds.txt -> {'exp': float, 'theta': float} at the given alpha."""
    path = Path(wy_dir) / "wy_thresholds.txt"
    for line in open(path):
        m = THRESHOLD_RE.match(line.strip())
        if m and float(m.group("alpha")) == float(alpha):
            return {"exp": float(m.group("exp")), "theta": float(m.group("theta"))}
    raise SystemExit(f"no alpha={alpha} row in {path}")


def significant(csv_path, pval_column, threshold):
    """Patterns whose p-value is at or below the threshold (blanks are skipped)."""
    out = set()
    for row in csv.DictReader(open(csv_path)):
        value = row.get(pval_column, "")
        if value not in ("", None) and float(value) <= threshold:
            out.add(row["pattern"])
    return out


def compare(results, cohort, sigma, theta, alpha, exp_threshold_from):
    base = Path(results) / f"{cohort}_sigma{sigma}"
    t_theta = read_thresholds(
        Path(results) / f"{cohort}_wy_sigma{sigma}_theta{theta}", alpha)["theta"]
    t_exp = read_thresholds(
        Path(results) / f"{cohort}_wy_sigma{sigma}_theta{exp_threshold_from}", alpha)["exp"]

    exp = significant(base / "significance/alg1_expected_uniform_pvalues_exp.csv",
                      "pval_exp", t_exp)
    con = significant(base / f"significance/alg3_theta{theta}_pvalues_theta.csv",
                      "pval_theta", t_theta)

    return {
        "cohort": cohort, "sigma": sigma, "theta": theta,
        "threshold_exp": t_exp, "threshold_theta": t_theta,
        "exp": len(exp), "theta_cons": len(con), "both": len(exp & con),
        "only_exp": sorted(exp - con), "only_theta": sorted(con - exp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/discovery",
                    help="Dir holding <cohort>_sigma<s> and <cohort>_wy_sigma<s>_theta<t>")
    ap.add_argument("--cohorts", nargs="+", default=["breastCancer", "tracerx"])
    ap.add_argument("--sigmas", nargs="+", type=int, default=[2, 5])
    ap.add_argument("--thetas", nargs="+", default=["0.5", "1.0"])
    ap.add_argument("--alpha", default="0.05")
    ap.add_argument("--exp-threshold-from", default="1.0",
                    help="Which WY run supplies the expected-support threshold")
    ap.add_argument("--list-diff", action="store_true",
                    help="Also print the trajectories found by only one test")
    ap.add_argument("--latex", action="store_true", help="Emit the table body as LaTeX rows")
    args = ap.parse_args()

    rows = []
    for cohort in args.cohorts:
        for sigma in args.sigmas:
            for theta in args.thetas:
                try:
                    rows.append(compare(args.results, cohort, sigma, theta,
                                        args.alpha, args.exp_threshold_from))
                except (FileNotFoundError, SystemExit) as e:
                    print(f"[skip] {cohort} sigma={sigma} theta={theta}: {e}")

    header = f"{'cohort':<14}{'sigma':>6}{'theta':>7}{'exp':>6}{'theta':>7}{'both':>6}{'onlyE':>7}{'onlyT':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['cohort']:<14}{r['sigma']:>6}{r['theta']:>7}{r['exp']:>6}"
              f"{r['theta_cons']:>7}{r['both']:>6}{len(r['only_exp']):>7}{len(r['only_theta']):>7}")
        if args.list_diff:
            for p in r["only_exp"]:
                print(f"    only exp   : {p}")
            for p in r["only_theta"]:
                print(f"    only theta : {p}   (thr exp={r['threshold_exp']:.6g}, "
                      f"thr theta={r['threshold_theta']:.6g})")

    if args.latex:
        print("\n% Table 7.6 body")
        for r in rows:
            name = "Breast" if r["cohort"].lower().startswith("breast") else "TRACERx"
            print(f"{name} & ${r['sigma']}$ & ${r['theta']}$ & ${r['exp']}$ & "
                  f"${r['theta_cons']}$ & ${r['both']}$ & ${len(r['only_exp'])}$ & "
                  f"${len(r['only_theta'])}$ \\\\")


if __name__ == "__main__":
    main()

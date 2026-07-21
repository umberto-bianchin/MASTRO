#!/usr/bin/env python3
"""Aggregate the single-tree baseline seed sweep across random seeds.

For each sigma, across all seeds under <root>/sigma<S>/seed<K>, report:
  - mean +/- sd of the number of FREQUENT trajectories mined by Alg 0;
  - mean +/- sd of the number SIGNIFICANT at FWER <= alpha;
  - the appear/disappear picture: how many trajectories are significant in
    EVERY seed (a stable "core") versus in only SOME (seed-dependent), with a
    few concrete examples of the volatile ones.

Reads, per seed:
  sigma<S>/seed<K>/significance/alg0_mastro_random_pvalues_exp.csv  (the family)
  sigma<S>/seed<K>_wy/wy_thresholds.txt                              (WY thresh)
"""
import argparse
import csv
import glob
import os
import re
import statistics as st


def read_threshold(wy_dir, alpha):
    path = os.path.join(wy_dir, "wy_thresholds.txt")
    if not os.path.exists(path):
        return None
    want = f"{alpha:.2f}"
    for line in open(path):
        fields = dict(tok.split("=") for tok in line.split())
        if fields.get("alpha") == want:
            return float(fields["threshold_exp"])
    return None


def read_family(sig_csv):
    """Return list of (pattern, pval_exp) for the Alg 0 family."""
    rows = []
    with open(sig_csv) as f:
        for r in csv.DictReader(f):
            rows.append((r["pattern"], float(r["pval_exp"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sigma_list", default="2 5")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    for sigma in args.sigma_list.split():
        emit(f"===== sigma = {sigma} (FWER <= {args.alpha}) =====")
        seed_dirs = sorted(
            glob.glob(os.path.join(args.root, f"sigma{sigma}", "seed*")),
            key=lambda p: int(re.search(r"seed(\d+)", p).group(1)),
        )
        seed_dirs = [d for d in seed_dirs if not d.endswith("_wy")]

        n_frequent = []
        n_significant = []
        sig_sets = {}   # seed -> set(patterns)
        all_seeds = []

        for d in seed_dirs:
            seed = int(re.search(r"seed(\d+)", d).group(1))
            sig_csv = os.path.join(d, "significance",
                                   "alg0_mastro_random_pvalues_exp.csv")
            wy_dir = os.path.join(args.root, f"sigma{sigma}", f"seed{seed}_wy")
            if not os.path.exists(sig_csv):
                emit(f"  [warn] seed {seed}: missing {sig_csv}, skipped")
                continue
            thr = read_threshold(wy_dir, args.alpha)
            if thr is None:
                emit(f"  [warn] seed {seed}: missing WY threshold, skipped")
                continue

            fam = read_family(sig_csv)
            sig = {pat for pat, p in fam if p <= thr}
            n_frequent.append(len(fam))
            n_significant.append(len(sig))
            sig_sets[seed] = sig
            all_seeds.append(seed)

        if not all_seeds:
            emit("  no usable seeds found.\n")
            continue

        def ms(xs):
            m = st.mean(xs)
            s = st.pstdev(xs) if len(xs) > 1 else 0.0
            return m, s

        fm, fs = ms(n_frequent)
        sm, ss = ms(n_significant)
        emit(f"  seeds used         : {len(all_seeds)}  {all_seeds}")
        emit(f"  frequent (Alg 0)   : mean {fm:.1f} +/- {fs:.1f}   "
             f"min {min(n_frequent)}  max {max(n_frequent)}")
        emit(f"  significant @FWER  : mean {sm:.2f} +/- {ss:.2f}   "
             f"min {min(n_significant)}  max {max(n_significant)}")

        # appear/disappear across seeds
        n = len(all_seeds)
        freq_in = {}
        for s in all_seeds:
            for pat in sig_sets[s]:
                freq_in[pat] = freq_in.get(pat, 0) + 1
        core = [p for p, c in freq_in.items() if c == n]
        volatile = sorted([(c, p) for p, c in freq_in.items() if c < n],
                          reverse=True)
        emit(f"  significant in ALL seeds (stable core) : {len(core)}")
        emit(f"  significant in SOME but not all seeds  : {len(volatile)}")
        if volatile:
            emit("  examples of seed-dependent trajectories "
                 "(#seeds significant / total):")
            for c, pat in volatile[:8]:
                emit(f"      {c}/{n}   {pat}")
        emit("")

    summary_path = os.path.join(args.root, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"[OK] summary written to {summary_path}")


if __name__ == "__main__":
    main()

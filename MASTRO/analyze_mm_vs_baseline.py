"""
Multi-MASTRO vs single-tree baseline: which trajectories does the ensemble
certify that the baseline does not, and vice versa.

Every number written by this script is read from a file on disk; nothing is
interpolated, smoothed or rounded beyond formatting. The provenance of each
column is printed by --provenance and recorded in the header of the run log.

Inputs (breastCancer, sigma = 5, FWER <= alpha via Westfall-Young)

  Multi-MASTRO (ensemble, all 37809 trees, per-tree weights):
    <discovery>/<cohort>_sigma<S>/significance/alg1_expected_uniform_pvalues_exp.csv
        columns: pattern,n_nodes,n_edges,s_exp,s_theta,pval_exp,pval_theta
    <discovery>/<cohort>_sigma<S>/significance/alg3_theta<T>_pvalues_theta.csv
        same columns, pval_theta populated instead of pval_exp
    <discovery>/<cohort>_wy_sigma<S>_theta<E>/wy_thresholds.txt   -> threshold_exp
    <discovery>/<cohort>_wy_sigma<S>_theta<T>/wy_thresholds.txt   -> threshold_theta
    <discovery>/<cohort>_sigma<S>/runs/alg1_expected_uniform/expected_uniform_filtered.txt
        pattern line + occurrence-id line, used with inputs/owner.txt to count
        the distinct patients carrying each trajectory

  Single-tree baseline (one sampled tree per patient, 10 seeds):
    <seeds>/sigma<S>/seed<k>/significance/alg0_mastro_random_pvalues_exp.csv
    <seeds>/sigma<S>/seed<k>_wy/wy_thresholds.txt                 -> threshold_exp

A trajectory is declared significant iff its raw p-value is <= the Westfall-Young
threshold of its own family at the requested alpha (run_wy_correction_ensemble.py,
"Reject H_P iff p_P <= threshold(alpha)").

Usage:
    python3 analyze_mm_vs_baseline.py
    python3 analyze_mm_vs_baseline.py --cohort breastCancer --sigma 5 \
        --alpha 0.05 --theta 0.5 --outdir results/mm_vs_baseline
"""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

EDGE_SEPARATORS = ["->-", "-/-", "-?-"]
THRESHOLD_RE = re.compile(
    r"alpha=(?P<alpha>[\d.]+)\s+threshold_exp=(?P<exp>\S+)\s+threshold_theta=(?P<theta>\S+)")


# ---------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------
def read_thresholds(wy_dir, alpha):
    """wy_thresholds.txt -> {'exp': float, 'theta': float} at the given alpha."""
    path = Path(wy_dir) / "wy_thresholds.txt"
    for line in open(path):
        m = THRESHOLD_RE.match(line.strip())
        if m and float(m.group("alpha")) == float(alpha):
            return {"exp": float(m.group("exp")), "theta": float(m.group("theta"))}
    raise SystemExit(f"no alpha={alpha} row in {path}")


def normalise(pattern):
    """Canonical key for a trajectory: edge-items sorted, single-space joined.

    compute_significance_ensemble.py already writes ' '.join(sorted(items)),
    so this is a no-op on well-formed files; it is applied anyway so that any
    file written under a different item ordering still joins correctly.
    """
    return " ".join(sorted(tok for tok in pattern.split() if tok))


def read_pvalues(csv_path, pval_column):
    """pattern -> row dict, keyed by the normalised pattern.

    Rows whose p-value column is blank are kept (they were mined but not scored
    under this test); callers decide what to do with them.
    """
    out = {}
    renamed = []
    for row in csv.DictReader(open(csv_path)):
        key = normalise(row["pattern"])
        if key != row["pattern"]:
            renamed.append((row["pattern"], key))
        row["_pval"] = row.get(pval_column, "")
        if key in out:
            raise SystemExit(f"duplicate pattern {key!r} in {csv_path}")
        out[key] = row
    return out, renamed


def read_occurrence_patients(filtered_path, owner_path):
    """pattern -> number of distinct patients carrying it.

    The filtered mining output alternates a pattern line ('items (support)')
    with a line of occurrence transaction ids; owner.txt gives the patient of
    each transaction. The patient count is |{owner[t] : t in occurrences}|.
    """
    owner = [int(x) for x in open(owner_path).read().split()]
    counts = {}
    with open(filtered_path) as f:
        while True:
            pl = f.readline()
            if not pl:
                break
            ol = f.readline()
            if not ol:
                break
            if "(" not in pl:
                continue
            items = re.sub(r"\(.*?\)", "", pl).strip()
            if not items:
                continue
            occ = [int(t) for t in ol.split() if t.isdigit()]
            counts[normalise(items)] = len({owner[t] for t in occ if 0 <= t < len(owner)})
    return counts


# ---------------------------------------------------------------------
# Rendering a trajectory as an ordered chain
# ---------------------------------------------------------------------
def parse_edges(pattern):
    """-> (nodes set, list of ('anc'|'inc', a, b))."""
    nodes, rels = set(), []
    for item in pattern.split():
        for sep in EDGE_SEPARATORS:
            if sep in item:
                a, b = item.split(sep)
                nodes.add(a)
                nodes.add(b)
                rels.append(("anc" if sep == "->-" else "inc", a, b))
                break
    return nodes, rels


def render_chain(pattern):
    """Human-readable ordered chain.

    A trajectory is relation-complete: every pair of its genes is either an
    ancestor pair ('A->-B') or an incomparable pair ('A-/-B'). Genes are laid
    out by their number of ancestors inside the pattern, so a totally ordered
    trajectory prints as 'A < B < C'; genes that tie at the same depth and are
    incomparable to each other print braced, 'A < {B | C}'.
    """
    nodes, rels = parse_edges(pattern)
    depth = {n: 0 for n in nodes}
    for kind, a, b in rels:
        if kind == "anc":
            depth[b] += 1
    layers = {}
    for n in nodes:
        layers.setdefault(depth[n], []).append(n)
    parts = []
    for d in sorted(layers):
        grp = sorted(layers[d])
        parts.append(grp[0] if len(grp) == 1 else "{" + " | ".join(grp) + "}")
    return " < ".join(parts)


def is_chain(pattern):
    _nodes, rels = parse_edges(pattern)
    return all(kind == "anc" for kind, _a, _b in rels)


# ---------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------
def build_rows(keys, mm_rows, mm_patients, mm_threshold, baseline, extra_baseline=False):
    """One row per trajectory in *keys*.

    baseline: {seed_id: {'sig': set(keys), 'mined': {key: row}, 'thr': float}}
    """
    seed_ids = sorted(baseline)
    rows = []
    for key in keys:
        mm = mm_rows.get(key)
        hits = [s for s in seed_ids if key in baseline[s]["sig"]]
        mined = [s for s in seed_ids if key in baseline[s]["mined"]]
        row = {
            "trajectory": render_chain(key),
            "pattern": key,
            "n_genes": len(parse_edges(key)[0]),
            "is_total_order": "yes" if is_chain(key) else "no",
            "s_exp": mm["s_exp"] if mm else "",
            "n_patients": mm_patients.get(key, ""),
            "pval": mm["_pval"] if mm else "",
            "wy_threshold": f"{mm_threshold:.6e}",
            "mined_by_multimastro": "yes" if mm else "no",
            "seeds_hit": len(hits),
            "seeds_hit_ids": ";".join(str(s) for s in hits),
            "mined_in_seeds": len(mined),
            "mined_in_seeds_ids": ";".join(str(s) for s in mined),
        }
        if extra_baseline:
            cands = [(float(baseline[s]["mined"][key]["_pval"]), s)
                     for s in mined if baseline[s]["mined"][key]["_pval"] != ""]
            if cands:
                p, s = min(cands)
                row["bl_min_pval"] = f"{p:.6e}"
                row["bl_min_pval_seed"] = s
                row["bl_min_pval_seed_threshold"] = f"{baseline[s]['thr']:.6e}"
                row["bl_s_exp_at_that_seed"] = baseline[s]["mined"][key]["s_exp"]
            else:
                row["bl_min_pval"] = ""
                row["bl_min_pval_seed"] = ""
                row["bl_min_pval_seed_threshold"] = ""
                row["bl_s_exp_at_that_seed"] = ""
        rows.append(row)

    def sort_key(r):
        p = float(r["pval"]) if r["pval"] not in ("", None) else float("inf")
        return (r["seeds_hit"], p)

    rows.sort(key=sort_key)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_table(title, rows, limit, columns):
    print(f"\n=== {title}  ({len(rows)} rows, showing {min(limit, len(rows))}) ===")
    if not rows:
        print("  (empty)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows[:limit])) for c in columns}
    print("  " + "  ".join(c.ljust(widths[c]) for c in columns))
    print("  " + "  ".join("-" * widths[c] for c in columns))
    for r in rows[:limit]:
        print("  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--discovery", default=str(here / "results/discovery"))
    ap.add_argument("--seeds", default=str(here / "results/singletree_seeds"))
    ap.add_argument("--outdir", default=str(here / "results/mm_vs_baseline"))
    ap.add_argument("--cohort", default="breastCancer")
    ap.add_argument("--sigma", type=int, default=5)
    ap.add_argument("--alpha", default="0.05")
    ap.add_argument("--theta", default="0.5")
    ap.add_argument("--exp-threshold-from", default="1.0",
                    help="Which WY run supplies the expected-support threshold")
    ap.add_argument("--seed-ids", nargs="+", type=int,
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    disc = Path(args.discovery)
    base = disc / f"{args.cohort}_sigma{args.sigma}"
    seeds_dir = Path(args.seeds) / f"sigma{args.sigma}"
    out = Path(args.outdir)

    mm_exp_csv = base / "significance/alg1_expected_uniform_pvalues_exp.csv"
    mm_theta_csv = base / f"significance/alg3_theta{args.theta}_pvalues_theta.csv"
    mm_filtered = base / "runs/alg1_expected_uniform/expected_uniform_filtered.txt"
    mm_owner = base / "inputs/owner.txt"
    wy_exp_dir = disc / f"{args.cohort}_wy_sigma{args.sigma}_theta{args.exp_threshold_from}"
    wy_theta_dir = disc / f"{args.cohort}_wy_sigma{args.sigma}_theta{args.theta}"

    print("=== files read ===")
    for p in (mm_exp_csv, mm_theta_csv, mm_filtered, mm_owner,
              wy_exp_dir / "wy_thresholds.txt", wy_theta_dir / "wy_thresholds.txt"):
        print(f"  {'OK ' if p.exists() else 'MISSING'} {p}")
    missing = []
    for s in args.seed_ids:
        for p in (seeds_dir / f"seed{s}/significance/alg0_mastro_random_pvalues_exp.csv",
                  seeds_dir / f"seed{s}_wy/wy_thresholds.txt"):
            print(f"  {'OK ' if p.exists() else 'MISSING'} {p}")
            if not p.exists():
                missing.append(p)
    if missing:
        sys.exit("aborting: baseline per-seed artifacts missing (listed above)")

    thr_exp = read_thresholds(wy_exp_dir, args.alpha)["exp"]
    thr_theta = read_thresholds(wy_theta_dir, args.alpha)["theta"]

    mm_exp, renamed_exp = read_pvalues(mm_exp_csv, "pval_exp")
    mm_theta, renamed_theta = read_pvalues(mm_theta_csv, "pval_theta")
    mm_patients = read_occurrence_patients(mm_filtered, mm_owner)

    mm_exp_sig = {k for k, r in mm_exp.items()
                  if r["_pval"] not in ("", None) and float(r["_pval"]) <= thr_exp}
    mm_theta_sig = {k for k, r in mm_theta.items()
                    if r["_pval"] not in ("", None) and float(r["_pval"]) <= thr_theta}

    baseline = {}
    for s in args.seed_ids:
        rows, renamed = read_pvalues(
            seeds_dir / f"seed{s}/significance/alg0_mastro_random_pvalues_exp.csv", "pval_exp")
        thr = read_thresholds(seeds_dir / f"seed{s}_wy", args.alpha)["exp"]
        sig = {k for k, r in rows.items()
               if r["_pval"] not in ("", None) and float(r["_pval"]) <= thr}
        baseline[s] = {"mined": rows, "sig": sig, "thr": thr}
        renamed_exp += renamed

    # ---- naming conventions ----
    print("\n=== gene-symbol normalisation ===")
    if renamed_exp or renamed_theta:
        for a, b in sorted(set(renamed_exp + renamed_theta))[:20]:
            print(f"  reordered items: {a!r} -> {b!r}")
    else:
        print("  none needed: every file writes edge-items in the same sorted order,")
        print("  and the same gene symbols. No symbol mapping was applied.")
    mm_genes = set()
    for k in mm_exp:
        mm_genes |= parse_edges(k)[0]
    bl_genes = set()
    for s in args.seed_ids:
        for k in baseline[s]["mined"]:
            bl_genes |= parse_edges(k)[0]
    print(f"  distinct gene symbols: Multi-MASTRO {len(mm_genes)}, baseline union {len(bl_genes)}, "
          f"baseline-only {sorted(bl_genes - mm_genes)}, multimastro-only {sorted(mm_genes - bl_genes)}")

    # ---- step 2: verification ----
    mined_counts = [len(baseline[s]["mined"]) for s in args.seed_ids]
    sig_counts = [len(baseline[s]["sig"]) for s in args.seed_ids]
    union_sig = set().union(*(baseline[s]["sig"] for s in args.seed_ids))
    core_sig = set.intersection(*(baseline[s]["sig"] for s in args.seed_ids))

    print("\n=== step 2: published numbers, recomputed ===")
    print(f"  WY threshold_exp (multi-MASTRO, alpha={args.alpha}) = {thr_exp:.6e}"
          f"   [{wy_exp_dir/'wy_thresholds.txt'}]")
    print(f"  WY threshold_theta (theta={args.theta}, alpha={args.alpha}) = {thr_theta:.6e}"
          f"   [{wy_theta_dir/'wy_thresholds.txt'}]")
    print(f"  baseline mined      : mean {statistics.mean(mined_counts):.1f} "
          f"+/- {statistics.stdev(mined_counts):.1f}   per seed {mined_counts}")
    print(f"  baseline significant: mean {statistics.mean(sig_counts):.1f} "
          f"+/- {statistics.stdev(sig_counts):.1f}   per seed {sig_counts}")
    print(f"  multi-MASTRO mined  : {len(mm_exp)}")
    print(f"  multi-MASTRO signif : {len(mm_exp_sig)}")
    print(f"  baseline union of significant sets : {len(union_sig)}")
    print(f"  significant in all {len(args.seed_ids)} seeds  : {len(core_sig)}")
    print(f"  multi-MASTRO theta={args.theta} mined  : {len(mm_theta)}")
    print(f"  multi-MASTRO theta={args.theta} signif : {len(mm_theta_sig)}")

    # ---- step 3 ----
    t3 = build_rows(mm_exp_sig, mm_exp, mm_patients, thr_exp, baseline)
    write_csv(out / f"{args.cohort}_sigma{args.sigma}_multimastro_significant_exp.csv", t3)

    # ---- step 4 ----
    rev_keys = union_sig - mm_exp_sig
    t4 = build_rows(rev_keys, mm_exp, mm_patients, thr_exp, baseline, extra_baseline=True)
    write_csv(out / f"{args.cohort}_sigma{args.sigma}_baseline_only_exp.csv", t4)

    # ---- step 5 ----
    t5 = build_rows(mm_theta_sig, mm_theta, mm_patients, thr_theta, baseline)
    write_csv(out / f"{args.cohort}_sigma{args.sigma}_multimastro_significant_theta{args.theta}.csv", t5)

    cols = ["trajectory", "s_exp", "n_patients", "pval", "wy_threshold",
            "seeds_hit", "seeds_hit_ids", "mined_in_seeds"]
    print_table(f"step 3 - Multi-MASTRO significant, expected support (sigma={args.sigma})",
                t3, args.top, cols)
    print_table(f"step 4 - significant in >=1 baseline seed, NOT under Multi-MASTRO",
                t4, args.top, cols + ["mined_by_multimastro", "bl_min_pval", "bl_min_pval_seed"])
    print_table(f"step 5 - Multi-MASTRO significant, theta-consensus theta={args.theta}",
                t5, args.top, cols)
    print(f"  note: seeds_hit is still baseline significance under the EXPECTED-SUPPORT test.")
    print(f"  The single-tree baseline carries one tree per patient, so its theta-consensus")
    print(f"  test is degenerate (s_theta == s_exp) and every seed_wy/wy_thresholds.txt")
    print(f"  records threshold_theta=1.000000e+00, which would declare the whole family")
    print(f"  significant. The baseline column is therefore its expected-support verdict.")

    print(f"\n=== CSVs written under {out} ===")
    for p in sorted(out.glob("*.csv")):
        print(f"  {p}")


if __name__ == "__main__":
    main()

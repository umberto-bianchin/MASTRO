"""
Toy experiment: POTTR counts candidate trees as independent trials.

Every patient has M IDENTICAL candidate trees, so the trees of one patient carry
exactly the same evidence as a single tree: a correct patient-level test cannot
depend on M. Two cohort designs (--design):

  ab_ba  2 alterations per tree. n_ab patients A->B, n_other patients B->A.
         Per-tree null p = 1/4 (indep), 1/2 (perm). Under perm the observed
         support equals its expectation, so this design only discriminates
         under the indep null.

  perm   3 alterations per tree: one ordered pair plus a leaf sibling, so every
         tree has exactly ONE comparable pair and per-tree p_perm = 1/6.
         n_ab carriers of A->B (C a leaf); the non-carriers cycle through the
         other 5 orders (B->A, A->C, C->A, B->C, C->B), so no competing
         trajectory is shared by >= n_ab patients and the ILP at k = n_ab
         returns A->B deterministically. Discriminates under the perm null,
         the one used in the thesis (tab:pottr).

  all_orders  K alterations, and every patient carries ALL K! linear orders as
         chains, each with weight 1/K!. The cohort therefore holds no evidence
         on the order at all. Whichever chain the ILP returns, every patient
         supports it with exactly one of its K! trees, so counting trees gives
         support = n_patients out of n_patients*K! trials and POTTR's own test
         is accidentally right. Counting PATIENTS instead, which is the obvious
         repair, gives n_patients out of n_patients trials at the same per-tree
         probability and reports a false positive.

What we expect for pattern A->B:
  * POTTR's null (compute_prob) gives every tree the same p and tests
    Bin(n_trees, p) >= support_trees, with n_trees = n_patients*M and
    support_trees = n_ab*M both growing with M.
  * The ensemble test draws ONE placement per patient, shared by all its trees,
    so it tests  Bin(n_patients, p) >= n_ab, independent of M.

For every M the script builds the cohort twice (POTTR dags + matched ensemble
inputs on the same trees), runs the FULL POTTR pipeline (ILP + support +
POTTR's own significance test), then re-tests the trajectory POTTR returned
with the ensemble test under both null models via pottr_significance.py.

Usage (MASTRO env; POTTR runs in its own env through --pottr_python):
    python3 pottr_toy_experiment.py --design perm --M_list 1,2,3,5,10 \\
        --pottr_python ~/miniforge3/envs/pottr_env/bin/python \\
        --out results/pottr_toy
"""

import argparse
import csv
import itertools
import math
import re
import subprocess
import sys
from pathlib import Path

from utils import SCRIPT_DIR


def default_pottr_code():
    """POTTR's code/ dir: local layout (../../POTTR/code) or server (../POTTR/code)."""
    for cand in (SCRIPT_DIR.parent.parent / "POTTR" / "code",
                 SCRIPT_DIR.parent / "POTTR" / "code"):
        if (cand / "run_POTTR.py").exists():
            return cand
    return SCRIPT_DIR.parent.parent / "POTTR" / "code"


# One tree per order, as transaction items (a->-b ancestor, min-/-max incomparable)
TREES_AB_BA = {"AB": "A->-B", "BA": "B->-A"}
TREES_PERM = {
    "AB": "A->-B A-/-C B-/-C", "BA": "B->-A A-/-C B-/-C",
    "AC": "A->-C A-/-B B-/-C", "CA": "C->-A A-/-B B-/-C",
    "BC": "B->-C A-/-B A-/-C", "CB": "C->-B A-/-B A-/-C",
}


def chain_items(order):
    """Transitively closed chain over *order*, as transaction items."""
    return " ".join(f"{a}->-{b}"
                    for a, b in itertools.combinations(order, 2))


def all_orders_trees(K):
    """Every linear order over K alterations, as one chain per order."""
    labels = [chr(ord("A") + i) for i in range(K)]
    return [chain_items(o) for o in itertools.permutations(labels)]


def patient_trees(design, n_ab, n_other, K=3):
    """Candidate trees of every patient, n_ab carriers of A->B first.

    An entry is either one item line (the patient carries that tree M_i times)
    or a list of item lines (the patient carries exactly those trees once each).
    """
    if design == "all_orders":
        # Every patient carries all K! chains, so the cohort holds no evidence
        # at all on the order
        return [all_orders_trees(K)] * (n_ab + n_other)
    if design == "ab_ba":
        return [TREES_AB_BA["AB"]] * n_ab + [TREES_AB_BA["BA"]] * n_other
    others = ["BA", "AC", "CA", "BC", "CB"]
    return ([TREES_PERM["AB"]] * n_ab
            + [TREES_PERM[others[i % len(others)]] for i in range(n_other)])


def build_cohort(root, trees, n_trees):
    """Write POTTR dags (one file per tree) and matched ensemble inputs.

    n_trees[i] = identical candidate trees of patient i; weights are 1/M_i."""
    dags = root / "dags"
    ens = root / "inputs"
    dags.mkdir(parents=True, exist_ok=True)
    ens.mkdir(parents=True, exist_ok=True)
    for f in dags.glob("*.txt"):
        f.unlink()

    with (ens / "graphs_all.txt").open("w") as fg, \
            (ens / "weights_uniform.txt").open("w") as fw, \
            (ens / "owner.txt").open("w") as fo:
        for i, (entry, Mi) in enumerate(zip(trees, n_trees)):
            items = [entry] * Mi if isinstance(entry, str) else list(entry)
            for j, item in enumerate(items):
                (dags / f"P{i}-{j}_toy.txt").write_text(item + "\n")
                fg.write(item + "\n")
                fw.write(f"{1.0 / Mi}\n")
                fo.write(f"{i}\n")
    return dags, ens


def run_pottr(pottr_python, pottr_code, dags, out_k, k):
    out_k.mkdir(parents=True, exist_ok=True)
    log = out_k / "pottr.log"
    with log.open("w") as fl:
        subprocess.run([str(pottr_python), "run_POTTR.py",
                        "-o", str(out_k.resolve()), "-d", str(dags.resolve()),
                        "-k", str(k), "-c", "1"],
                       cwd=pottr_code, stdout=fl, stderr=subprocess.STDOUT,
                       check=True)


def read_pottr(out_k):
    """POTTR's trajectory, tree support and its own p-values (indep, perm)."""
    conv = (out_k / "converted_graphs.txt").read_text().splitlines()
    m = re.match(r"(.*)\((\d+)\)\s*$", conv[0])
    assert m, f"unexpected converted_graphs.txt line: {conv[0]!r}"
    traj = " ".join(e for e in m.group(1).split() if not e.startswith("0"))
    supp_trees = int(m.group(2))
    supporting = conv[1].split()
    with (out_k / "significance_output.txt").open() as f:
        header = f.readline().rstrip("\n").split(";")
        row = f.readline().rstrip("\n").split(";")
    rec = dict(zip(header, row))
    return dict(traj=traj, supp_trees=supp_trees,
                supp_patients=len({s.split("-")[0] for s in supporting}),
                n_trees_null=round(supp_trees / float(rec["traj_freq"])),
                pottr_exp_ind=float(rec["traj_exp_ind"]) * round(supp_trees / float(rec["traj_freq"])),
                pottr_pval_ind=float(rec["pval_ind"]),
                pottr_pval_perm=float(rec["pval_perm"]),
                # Mean per-tree null probability of the trajectory, taken from
                # POTTR's own output: traj_exp_* is a frequency, not a count
                p_tree_ind=float(rec["traj_exp_ind"]),
                p_tree_perm=float(rec["traj_exp_perm"]))


def binom_sf(n, s, p):
    """P(Bin(n, p) >= s), computed exactly from the terms."""
    if s <= 0:
        return 1.0
    if s > n:
        return 0.0
    return sum(math.comb(n, j) * p ** j * (1.0 - p) ** (n - j)
               for j in range(s, n + 1))


def run_ensemble(root, ens, null, theta=1.0, tag=""):
    out = root / f"ensemble_{null}{tag}.csv"
    subprocess.run([sys.executable, str(SCRIPT_DIR / "pottr_significance.py"),
                    "--pottr_dir", str(root / "pottr"), "--k_range", "1,100",
                    "--graphs_all", str(ens / "graphs_all.txt"),
                    "-w", str(ens / "weights_uniform.txt"),
                    "--owner", str(ens / "owner.txt"),
                    "--theta", repr(theta), "--null", null,
                    # placements are at most n_alt^k = 9: always enumerate exactly
                    "--mc_cutoff", "1000000",
                    "--out", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    row = next(csv.DictReader(out.open()))
    return dict(s_exp=float(row["s_exp"]), s_theta=int(row["s_theta"]),
                pval_exp=float(row["mastro_pval_exp"]),
                pval_theta=float(row["mastro_pval_theta"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M_list", default="1,2,3,5,10",
                    help="Identical candidate trees per patient to sweep")
    ap.add_argument("--design", choices=["ab_ba", "perm", "all_orders"],
                    default="perm")
    ap.add_argument("--K", type=int, default=3,
                    help="Alterations per patient, all_orders only: every "
                         "patient carries all K! chains, so M = K!")
    ap.add_argument("--n_ab", type=int, default=None,
                    help="Carriers of A->B (default: 5 for ab_ba, 4 for perm)")
    ap.add_argument("--n_other", type=int, default=None,
                    help="Non-carriers (default: 5 for ab_ba, 6 for perm)")
    ap.add_argument("--M_other", type=int, default=None,
                    help="Trees per NON-carrier (default: same M as the "
                         "carriers). Unequal M exposes the missing 1/M_i weights")
    ap.add_argument("--k", type=int, default=None,
                    help="POTTR k (default: n_ab, the carriers)")
    ap.add_argument("--pottr_python", required=True,
                    help="Python of the env with gurobipy/networkx/pandas/filelock")
    ap.add_argument("--pottr_code", default=str(default_pottr_code()))
    ap.add_argument("--out", default="results/pottr_toy")
    args = ap.parse_args()

    if args.design == "all_orders":
        # Every patient is identical, so n_ab is simply the cohort size
        n_ab_default, M_list = 10, [math.factorial(args.K)]
    else:
        n_ab_default = 5 if args.design == "ab_ba" else 4
        M_list = [int(x) for x in args.M_list.split(",")]
    n_ab = args.n_ab if args.n_ab is not None else n_ab_default
    n_other = args.n_other if args.n_other is not None else 10 - n_ab
    k = args.k if args.k is not None else n_ab
    trees = patient_trees(args.design, n_ab, n_other, args.K)

    out = Path(args.out) / args.design
    rows = []
    for M in M_list:
        root = out / f"M{M}"
        M_other = args.M_other if args.M_other is not None else M
        n_trees = [M] * n_ab + [M_other] * n_other
        dags, ens = build_cohort(root, trees, n_trees)
        out_k = root / "pottr" / f"k{k}"
        run_pottr(Path(args.pottr_python).expanduser(), Path(args.pottr_code),
                  dags, out_k, k)
        r = dict(design=args.design, M=M, M_other=M_other,
                 n_patients=len(trees), n_carriers=n_ab,
                 **read_pottr(out_k))
        for null in ("indep", "perm"):
            e = run_ensemble(root, ens, null)
            r.update({f"mm_{key}_{null}": v for key, v in e.items()})
            # theta = 1/M is the consensus test at its weakest: a patient counts
            # as soon as ONE of its trees carries the trajectory.
            e1 = run_ensemble(root, ens, null, theta=1.0 / M, tag="_theta_min")
            r.update({f"mm_{key}_min_{null}": v for key, v in e1.items()})
        # Naive repair of POTTR's test: keep its per-tree null probability but
        # count patients instead of trees
        for null, key in (("ind", "p_tree_ind"), ("perm", "p_tree_perm")):
            r[f"pottr_naivefix_pval_{null}"] = binom_sf(
                r["n_patients"], r["supp_patients"], r[key])
        rows.append(r)
        print(f"[M={M:>2}] POTTR {r['traj']}  supp={r['supp_trees']} trees "
              f"({r['supp_patients']} pts) / {r['n_trees_null']} trees | "
              f"p_ind={r['pottr_pval_ind']:.3e} p_perm={r['pottr_pval_perm']:.3e} || "
              f"MM s_exp={r['mm_s_exp_indep']:g} "
              f"p_exp_ind={r['mm_pval_exp_indep']:.3e} "
              f"p_exp_perm={r['mm_pval_exp_perm']:.3e} "
              f"p_theta_min_perm={r['mm_pval_theta_min_perm']:.3e} || "
              f"naive fix p_perm={r['pottr_naivefix_pval_perm']:.3e}",
              flush=True)

    summary = out / "summary.csv"
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[toy] written -> {summary}")


if __name__ == "__main__":
    main()

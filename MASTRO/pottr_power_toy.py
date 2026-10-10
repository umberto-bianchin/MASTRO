"""
Toy experiment: treating trees as independent trials can also cost power.

When the trees of a patient are alike, their indicators are positively
correlated and a test that treats them as independent trials underestimates the
variance (anti-conservative). When they disagree, the correlation is negative
and the same error overestimates the variance, so the test loses power.

The typical disagreement is the one phylogenetic uncertainty produces: a patient
whose two candidate trees differ by the order of two alterations. Under any
null that relabels the patient once, coherently across its trees, exactly one of
{A->B, B->A} carries any given order of two alterations, so the patient's
contribution is constant. POTTR instead counts two independent trials.

Cohort:
  * n_ab carriers, one tree A->B each;
  * n_unc uncertain patients, two trees {A->B, B->A} each (weights 1/2).

Under the perm null (per-tree p = 1/2):
  * Multi-MASTRO: the uncertain patients add a constant 1/2 each, so
        p = P(Bin(n_ab, 1/2) >= n_ab) = 2^-n_ab, for every n_unc;
  * POTTR: n_ab + 2 n_unc trials, n_ab + n_unc successes,
        p = P(Bin(n_ab + 2 n_unc, 1/2) >= n_ab + n_unc), which grows with n_unc.
At n_unc = 0 the two coincide. The ILP at k = n_ab + n_unc must take the A->B
tree of every patient, so POTTR returns A->B deterministically.

Usage (MASTRO env; POTTR runs in its own env through --pottr_python):
    python3 pottr_power_toy.py --n_ab 6 --unc_list 0,5,10,20,40 \\
        --pottr_python ~/miniforge3/envs/pottr_env/bin/python \\
        --out results/pottr_toy
"""

import argparse
import csv
from pathlib import Path

from pottr_toy_experiment import (
    binom_sf,
    build_cohort,
    default_pottr_code,
    read_pottr,
    run_ensemble,
    run_pottr,
)

TREE_AB = "A->-B"
TREE_BA = "B->-A"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_ab", type=int, default=6,
                    help="Carriers, one A->B tree each")
    ap.add_argument("--unc_list", default="0,5,10,20,40",
                    help="Uncertain patients {A->B, B->A} to sweep")
    ap.add_argument("--pottr_python", required=True,
                    help="Python of the env with gurobipy/networkx/pandas/filelock")
    ap.add_argument("--pottr_code", default=str(default_pottr_code()))
    ap.add_argument("--out", default="results/pottr_toy")
    args = ap.parse_args()

    c = args.n_ab
    out = Path(args.out) / f"power_{c}"
    rows = []
    for h in [int(x) for x in args.unc_list.split(",")]:
        root = out / f"unc{h}"
        trees = [TREE_AB] * c + [[TREE_AB, TREE_BA]] * h
        dags, ens = build_cohort(root, trees, [1] * c + [2] * h)
        k = c + h
        out_k = root / "pottr" / f"k{k}"
        run_pottr(Path(args.pottr_python).expanduser(), Path(args.pottr_code),
                  dags, out_k, k)
        r = dict(n_ab=c, n_unc=h, k=k, **read_pottr(out_k),
                 theory_mm_pval_perm=2.0 ** -c,
                 theory_pottr_pval_perm=binom_sf(c + 2 * h, c + h, 0.5))
        for null in ("perm", "indep"):
            e = run_ensemble(root, ens, null)
            r.update({f"mm_{key}_{null}": v for key, v in e.items()})
        rows.append(r)
        print(f"[n_unc={h:>3}] POTTR {r['traj']} supp={r['supp_trees']} trees "
              f"/ {r['n_trees_null']} | p_perm={r['pottr_pval_perm']:.3e} "
              f"(theory {r['theory_pottr_pval_perm']:.3e}) || "
              f"MM s_exp={r['mm_s_exp_perm']:g} "
              f"p_exp_perm={r['mm_pval_exp_perm']:.3e} "
              f"(theory {r['theory_mm_pval_perm']:.3e})", flush=True)

    summary = out / "summary.csv"
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[toy] written -> {summary}")


if __name__ == "__main__":
    main()

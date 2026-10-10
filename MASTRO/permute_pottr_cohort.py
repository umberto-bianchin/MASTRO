"""
Build a NULL copy of a matched POTTR + Multi-MASTRO cohort.

Input is the output of breastcancer_to_pottr.py: the POTTR dags
(P<i>-<j>_bc.txt, one tree per file) and the matched ensemble inputs
(graphs_all.txt, weights_uniform.txt, owner.txt, manifest.csv) built from the
identical trees. Every patient gets one random bijection of its own alterations,
drawn independently across patients and applied to ALL of its candidate trees
(permute_transactions, the same 'perm' null the calibration experiment uses).

What survives: patients, number of trees per patient, every tree topology, and
the agreement/disagreement between the trees of a patient. What is destroyed:
any recurrence of a labelled pattern across patients. Nothing in the output is
genuinely conserved, so every significant trajectory is a false discovery.

Permuting each tree independently instead would make POTTR's own null true by
construction, and would turn the alternative reconstructions of one patient
into unrelated trees; it is deliberately not offered.

Usage:
    python3 permute_pottr_cohort.py \\
        --dags_in DAGS --ens_in INPUTS --dags_out DAGS_NULL --ens_out INPUTS_NULL \\
        --seed 1
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from run_wy_correction_ensemble import permute_transactions
from utils import load_weights_and_owner


def read_manifest(path):
    """Rows (orig_patient_idx, n_trees) in ensemble owner order."""
    rows = []
    with path.open() as f:
        next(f)
        for line in f:
            _, orig, n = line.strip().split(",")
            rows.append((int(orig), int(n)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dags_in", required=True, help="POTTR dags of the real cohort")
    ap.add_argument("--ens_in", required=True,
                    help="Matched ensemble inputs (graphs_all, weights, owner, manifest)")
    ap.add_argument("--dags_out", required=True)
    ap.add_argument("--ens_out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    dags_in, ens_in = Path(args.dags_in), Path(args.ens_in)
    dags_out, ens_out = Path(args.dags_out), Path(args.ens_out)

    lines = (ens_in / "graphs_all.txt").read_text().splitlines()
    _, owner0, n_patients, _ = load_weights_and_owner(
        ens_in / "weights_uniform.txt", ens_in / "owner.txt")
    manifest = read_manifest(ens_in / "manifest.csv")
    assert len(manifest) == n_patients, "manifest and owner.txt disagree"

    # graphs_all line t is tree j of the owner0[t]-th selected patient, which
    # breastcancer_to_pottr.py wrote to P<orig>-<j>_bc.txt: check it, so the two
    # sides of the null cohort describe the same trees
    files = []
    t = 0
    for orig, n_trees in manifest:
        for j in range(n_trees):
            f = dags_in / f"P{orig}-{j}_bc.txt"
            assert f.read_text().strip() == lines[t].strip(), \
                f"{f.name} does not match graphs_all line {t}"
            files.append(f.name)
            t += 1
    assert t == len(lines), "graphs_all has trees the manifest does not list"
    n_dags = len(list(dags_in.glob("*_bc.txt")))
    assert n_dags == len(lines), f"{n_dags} dag files for {len(lines)} trees"

    rng = np.random.default_rng(args.seed)
    permuted = permute_transactions(lines, owner0, n_patients, rng)

    dags_out.mkdir(parents=True, exist_ok=True)
    for f in dags_out.glob("*_bc.txt"):
        f.unlink()
    for name, line in zip(files, permuted):
        (dags_out / name).write_text(line + "\n")

    ens_out.mkdir(parents=True, exist_ok=True)
    (ens_out / "graphs_all.txt").write_text("\n".join(permuted) + "\n")
    for name in ("weights_uniform.txt", "owner.txt", "manifest.csv"):
        shutil.copyfile(ens_in / name, ens_out / name)

    n_changed = sum(a != b for a, b in zip(lines, permuted))
    print(f"[null cohort] seed={args.seed}: {n_patients} patients, "
          f"{len(lines)} trees ({n_changed} relabelled) -> {dags_out}, {ens_out}")


if __name__ == "__main__":
    main()

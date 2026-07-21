"""Print and save summary statistics for a .npy tree-ensemble dataset.

Loads a dataset (object array indexed data[patient][tree][edge] = [parent,
child]), computes the per-cohort statistics via utils.compute_dataset_stats
(number of patients, trees-per-patient distribution, mutation counts, fraction
of patients whose trees all share the same mutation set), prints a short summary
including the trees-per-patient histogram, and writes the full statistics to a
JSON file.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from utils import (
    ensure_dir,
    load_npy,
    compute_dataset_stats,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True, help="Path to breastCancer.npy")
    ap.add_argument("--outdir", default=None, help="Output directory (default: results_<timestamp>)")
    ap.add_argument("--keep_gl", action="store_true", help="Keep GL in mutation sets")
    ap.add_argument("--tree_dist", action="store_true",
                    help="Also compute and print the distribution of #trees per patient")
    args = ap.parse_args()

    npy_path = Path(args.npy).resolve()
    drop_gl = not args.keep_gl

    if args.outdir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(f"results_{ts}").resolve()
    else:
        outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)

    data = load_npy(npy_path)
    patients_trees = list(data)

    stats = compute_dataset_stats(patients_trees, drop_gl=drop_gl)

    if args.tree_dist:
        from collections import Counter
        dist = Counter(len(tlist) for tlist in patients_trees)
        stats["trees_per_patient_distribution"] = {
            str(k): v for k, v in sorted(dist.items())
        }
        print("\n--- Distribution: #trees -> #patients ---")
        for n_trees, n_patients in sorted(dist.items()):
            bar = "#" * min(n_patients, 60)
            print(f"  {n_trees:4d} trees: {n_patients:5d} patients  {bar}")

    (outdir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print("[OK] Dataset stats written to", outdir / "dataset_stats.json")

    print("\n[DONE] Outputs are under:", outdir)

if __name__ == "__main__":
    main()
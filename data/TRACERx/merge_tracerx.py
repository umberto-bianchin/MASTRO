"""Merge individual TRACERx tree files into graphs.txt + owner.txt.

Usage:
    python3 merge_tracerx.py --indir data_tracerx --outdir .
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

PAT_RE = re.compile(r"^(CRUK\d+(?:_Tumour\d+)?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data_tracerx", help="Directory with per-tree .txt files")
    ap.add_argument("--outdir", default=".", help="Where to write graphs.txt and owner.txt")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Group files by patient
    patient_files = defaultdict(list)
    for f in sorted(indir.glob("CRUK*_tracerx_tree_*.txt")):
        m = PAT_RE.match(f.name)
        if m:
            patient_files[m.group(1)].append(f)

    graphs_path = outdir / "graphs_tracerx.txt"
    owner_path = outdir / "owner_tracerx.txt"

    n_transactions = 0
    with graphs_path.open("w") as fg, owner_path.open("w") as fo:
        for patient_id in sorted(patient_files):
            for tree_file in patient_files[patient_id]:
                line = tree_file.read_text().strip()
                if line:
                    fg.write(line + "\n")
                    fo.write(patient_id + "\n")
                    n_transactions += 1

    print(f"[OK] {len(patient_files)} patients, {n_transactions} transactions")
    print(f"     {graphs_path}")
    print(f"     {owner_path}")


if __name__ == "__main__":
    main()

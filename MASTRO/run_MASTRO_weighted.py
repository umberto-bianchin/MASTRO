# =============================================================================
# MASTRO pipeline  (Algorithms 1, 2 & 3)
#
# Stage 1  – Weighted frequent itemset mining (via LCM)
#   Uses transaction weights w_t = 1/M_i so that weighted support equals
#   the expected-support metric.  The support threshold -s is therefore a
#   float (expected # of patients).
#   This stage alone corresponds to Algorithm 1 (expected-support mining).
#   Omit -owner (and skip -theta / -st) to stop here.
#
# Stage 2  – Post-filter for robustness (requires -w and -owner)
#   Algorithm 2 (default):  keep patterns P whose theta-support
#                           s^(θ)(P) ≥ σ_θ  (see postfilter_theta.py)
#   Algorithm 3 (-alg3)  :  keep only theta-MAXIMAL patterns within each
#                           s^(θ) bucket  (see postfilter_theta.py --maximal)
#
# Example usage:
#   # Algorithm 1 only (expected support):
#   python3 run_MASTRO_weighted.py -g test.txt -s 2 -w weights.txt
#
#   # Algorithm 2 (theta-frequent):
#   python3 run_MASTRO_weighted.py -g test.txt -s 2 -w weights.txt \
#       -owner owner.txt -theta 0.8 -st 2
#
#   # Algorithm 3 (theta-maximal):
#   python3 run_MASTRO_weighted.py -g test.txt -s 1.2 -w weights.txt \
#       -owner owner.txt -theta 0.6 -st 2 -alg3
# =============================================================================

import argparse
import os
from utils import count_lines

parser = argparse.ArgumentParser(
    description="Two-stage robust trajectory mining (Algorithms 2/3)"
)
parser.add_argument("-g", help="input file with graphs (one transaction per line)")
parser.add_argument("-p", type=int, help="permutation type: 0=independent, 1=permutation, 2=ind. random topology (def=0)", default=0)
parser.add_argument("-s", type=float, help="minimum (expected) support threshold (float, def=2)", default=2)
parser.add_argument("-minp", help="path to append minimum p-value (optional)", default="minptest.csv")

parser.add_argument("-w", help="weight file: one weight per transaction (w_t = 1/M_i)", default=None)

parser.add_argument("-theta", type=float, help="theta threshold in (0,1] (def=1.0)", default=1.0)
parser.add_argument("-st", type=int, help="sigma_theta  minimum theta-support (def=2)", default=2)
parser.add_argument("-owner", help="owner file: one patient-id per transaction line", default=None)

parser.add_argument("-alg3", action="store_true",
                    help="if set, run Algorithm 3 (theta-maximal) instead of Algorithm 2 (theta-frequent)")

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Derive file paths used by the pipeline stages
# ---------------------------------------------------------------------------
temp_files_out = args.g.replace(".txt", "")

# transnum.pl writes the edge→ID mapping here
table_file_ids = "./lcm53/table-file-" + args.g
# Numeric-ID version of the graph file (input to LCM)
file_graphs_ids = "./lcm53/lcm-out-" + temp_files_out + "_ids.txt"
# LCM weight flag: pass "-w <file>" when weighted, else empty
weights = f" -w {args.w} " if args.w is not None else " "
# Raw LCM output (pattern + occurrence lines)
output_lcm = "./lcm53/lcm-out-" + args.g
# Intermediate results after ID→label conversion and maximal filtering
results_converted = temp_files_out + "_convres.txt"
results_filtered  = temp_files_out + "_filtered.txt"
results_significance = temp_files_out + "_final.txt"

# Stage-2 post-filter output files
results_theta     = temp_files_out + f"_theta{args.theta}_st{args.st}.txt"
results_theta_max = temp_files_out + f"_theta{args.theta}_st{args.st}_thetamax.txt"

# Step 1: convert edge labels → numeric IDs using transnum.pl
cmd = "./lcm53/transnum.pl " + table_file_ids + " < " + args.g + " > " + file_graphs_ids
print(cmd)
os.system(cmd)


# Sanity check: the number of transactions must equal the number of weights
# and the number of owner labels (when provided).
# NOTE: n_tr is always computed so it is available for the owner check below.
n_tr = count_lines(file_graphs_ids)

if args.w is not None:
    n_w = count_lines(args.w)
    if n_tr != n_w:
        raise ValueError(f"Mismatch: {n_tr} transactions vs {n_w} weights")

if args.owner is not None:
    n_o = count_lines(args.owner)
    if n_tr != n_o:
        raise ValueError(f"Mismatch: {n_tr} transactions vs {n_o} owner lines")

# Step 2: Run LCM (frequent itemset miner).
#   "FfI" flags:  F = frequent itemsets, f = output occurrences, I = input in item-list format.
#   When weights are provided (via -w), LCM uses weighted support ≥ σ.
cmd = ("./lcm53/lcm FfI" + weights + file_graphs_ids + " "
       + str(args.s) + " " + output_lcm + " > out_lcm_" + args.g + ".txt 2>&1")
print(cmd)
os.system(cmd)

# Step 3: Convert numeric IDs back to human-readable edge labels.
cmd = "python3 convert_results.py -m " + table_file_ids + " -i " + output_lcm + " -o " + results_converted
print(cmd)
os.system(cmd)

# Step 4: Keep only one representative per occurrence set (maximal pattern
# by containment).  Identical occurrence lists → keep the largest itemset.
cmd = "python3 filter_results.py -i " + results_converted + " -o " + results_filtered
print(cmd)
os.system(cmd)

# Step 5  –  Stage-2 post-filter (robustness via θ-support)
# Requires both weights (-w) and owner (-owner) to know per-patient support.
if args.w is not None and args.owner is not None:
    out_file = results_theta_max if args.alg3 else results_theta
    cmd = (
        "python3 postfilter_theta.py"
        " -i " + results_filtered +
        " -o " + out_file +
        " -w " + args.w +
        " -owner " + args.owner +
        " -theta " + str(args.theta) +
        " -st " + str(args.st) +
        (" --maximal" if args.alg3 else "")
    )
    print(cmd)
    os.system(cmd)
else:
    print("Skipping postfilter (need -w and -owner).")
    print("Output candidates (expected-maximal):", results_filtered)

# compute significance of results
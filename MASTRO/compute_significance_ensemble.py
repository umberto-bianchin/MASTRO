# =============================================================================
# Ensemble-based significance testing for MASTRO
#
# Compute p-values for mined trajectories when each patient has an ensemble of
# weighted candidate trees. The script supports:
#
#   - Expected-support test
#   - theta-consensus test
#
# Both tests support:
#   - 'perm'  null model: a single random permutation sigma_i of A_i, applied
#             coherently to all candidate trees of patient i
#   - 'indep' null model: each alteration of P is placed i.i.d. uniformly
#             on the alteration set (matches MASTRO's baseline denominator
#             |A_i|^k)
#   - Exact computation when M_i <= --mc_cutoff and the total number of
#     placements fits --max_exact_placements
#   - Monte-Carlo fallback otherwise (--mc_samples draws per patient)
#
# The third null model ("random topology") is not
# implemented in this first version
#
# Main flow:
#   1. Convert each patient ensemble to a pairwise-relation tensor
#   2. Build one patient-level null distribution Phi_i per trajectory
#   3. Aggregate Phi_i across patients and compute the requested p-value(s)
#
# Input files usually come from run_pipeline.py:
#   -i / --input    : a filtered result file (_filtered.txt / _alg2_*.txt /
#                     _alg3_*.txt) - alternating pattern / occurrence lines
#   -w / --weights  : weights.txt used by the weighted pipeline
#   --owner         : owner.txt  (one patient id per transaction)
#   --npy           : the original .npy dataset, needed to reconstruct each
#                     tree's pairwise-relation graph for the null calculation
#
# Output: a CSV with one row per valid trajectory, containing the observed
# supports and the computed p-values
# =============================================================================

import argparse
import csv
import itertools
import math
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

from utils import (
    ensure_dir,
    load_npy,
    load_weights_and_owner,
    read_result_pairs,
    parse_items_from_pattern_line,
    parse_pattern_nodes_and_edges,
    parse_occ_ids,
    is_valid_trajectory,
    compute_theta_support,
    extract_edges_from_tree,
    tree_edges_to_transaction_items,
)


# =====================================================================
# Pairwise-relation encoding
# =====================================================================
# Encode each pairwise relation as a small integer for fast tensor matching
REL_NONE = 0  # disagreement across trees / no item / diagonal
REL_ANC_FWD = 1  # a is ancestor of b
REL_ANC_REV = 2  # b is ancestor of a
REL_INC = 3  # a and b incomparable (different branches)
REL_UNC = 4  # order unknown (a and b in the same node)

REL_FLIP = {
    REL_NONE: REL_NONE,
    REL_ANC_FWD: REL_ANC_REV,
    REL_ANC_REV: REL_ANC_FWD,
    REL_INC: REL_INC,
    REL_UNC: REL_UNC,
}
# Flip relation direction when filling the symmetric tensor entry


def parse_edge_item_code(item):
    """Convert one edge item string to a canonical pair and relation code"""
    if "->-" in item:
        x, y = item.split("->-", 1)
        if x < y:
            return (x, y), REL_ANC_FWD
        return (y, x), REL_ANC_REV
    if "-/-" in item:
        x, y = item.split("-/-", 1)
        if x > y:
            x, y = y, x
        return (x, y), REL_INC
    if "-?-" in item:
        x, y = item.split("-?-", 1)
        if x > y:
            x, y = y, x
        return (x, y), REL_UNC
    return None


def items_to_rel_dict(items):
    """Convert an iterable of edge-items to a {(min,max): code} dict"""
    out = {}
    for item in items:
        parsed = parse_edge_item_code(item)
        if parsed is not None:
            out[parsed[0]] = parsed[1]
    return out


def build_rel_tensor(trees_rel_list, alterations):
    """Transform a patient's candidate trees into an (M, n_alt, n_alt) tensor"""
    n = len(alterations)
    M = len(trees_rel_list)
    alt_idx = {a: i for i, a in enumerate(alterations)}
    rel = np.zeros((M, n, n), dtype=np.int8)
    for j, rd in enumerate(trees_rel_list):
        for (x, y), code in rd.items():
            if x not in alt_idx or y not in alt_idx:
                # Skip alterations that do not appear in this patient's set
                # (can happen when using the --graphs_all path)
                continue
            i1 = alt_idx[x]
            i2 = alt_idx[y]
            # Fill both directions for direct lookup during matching
            rel[j, i1, i2] = code
            rel[j, i2, i1] = REL_FLIP[code]
    return rel


def pattern_expected_matrix(pattern_items):
    """Transform a trajectory pattern into its expected relation matrix"""
    nodes = set()
    edges = []
    for item in pattern_items:
        parsed = parse_edge_item_code(item)
        if parsed is None:
            continue
        (x, y), code = parsed
        nodes.add(x)
        nodes.add(y)
        edges.append(((x, y), code))
    pnodes = sorted(nodes)
    k = len(pnodes)
    pidx = {u: i for i, u in enumerate(pnodes)}
    expected = np.zeros((k, k), dtype=np.int8)
    for (x, y), code in edges:
        ix = pidx[x]
        iy = pidx[y]
        expected[ix, iy] = code
        expected[iy, ix] = REL_FLIP[code]
    return pnodes, expected


# =====================================================================
# Placement enumeration / sampling
# =====================================================================
# Generate null placements of trajectory nodes onto patient alterations
def build_placements_exact(n_alt, k, null_model, max_placements):
    """Build the full placement matrix, or return None if it is too large"""
    if k > n_alt:
        return np.zeros((0, k), dtype=np.int32)
    if null_model == "perm":
        # Injective placements: pattern nodes map to distinct alterations
        total = math.perm(n_alt, k)
        if total > max_placements:
            return None
        return np.array(list(itertools.permutations(range(n_alt), k)),
                        dtype=np.int32)
    # Independent placements: repeated target alterations are allowed
    total = n_alt ** k
    if total > max_placements:
        return None
    return np.array(list(itertools.product(range(n_alt), repeat=k)),
                    dtype=np.int32)


def build_placements_mc(n_alt, k, B, null_model, rng):
    """Sample a placement matrix from the selected null model"""
    if k > n_alt:
        return np.zeros((0, k), dtype=np.int32)
    if null_model == "perm":
        # Sample without replacement to mimic a random permutation
        placements = np.empty((B, k), dtype=np.int32)
        for b in range(B):
            placements[b] = rng.choice(n_alt, size=k, replace=False)
        return placements
    # Sample independently with replacement
    return rng.integers(0, n_alt, size=(B, k), dtype=np.int32)


def compute_match_sets(placements, rel_tensor, expected):
    """Map every placement h to the candidate-tree set J(h) it matches

    Example:
      placements[b] = [0, 2, 3] means:
        pattern node 0 -> patient alteration 0
        pattern node 1 -> patient alteration 2
        pattern node 2 -> patient alteration 3

      If this placement satisfies the pattern relations in candidate trees
      1 and 4, the output entry is frozenset({1, 4})
    """
    B = placements.shape[0]
    M = rel_tensor.shape[0]
    if B == 0:
        return []
    k = expected.shape[0]

    # match[b, j] says whether placement b still matches candidate tree j
    match = np.ones((B, M), dtype=bool)

    # Check every required pairwise relation in the pattern
    for p in range(k):
        for q in range(p + 1, k):
            exp_code = int(expected[p, q])
            if exp_code == REL_NONE:
                continue

            # For each placement b, read the patient alterations assigned to
            # pattern nodes p and q
            left = placements[:, p]
            right = placements[:, q]

            # Compare one required pattern relation across all placements/trees
            # actual has shape (M, B): candidate trees x placements
            actual = rel_tensor[:, left, right]

            # A placement/tree pair survives only if this relation also matches
            # Transpose to (B, M) so it aligns with match[placement, tree]
            match &= (actual == exp_code).T

    # Convert each placement row into the set of matched candidate trees
    row_sums = match.sum(axis=1)
    if row_sums.max(initial=0) == 0:
        return [frozenset()] * B
    out = []
    for b in range(B):
        if row_sums[b] == 0:
            out.append(frozenset())
        else:
            out.append(frozenset(np.nonzero(match[b])[0].tolist()))
    return out


# =====================================================================
# Patient-level Phi_i
# =====================================================================
def compute_phi_i(rel_tensor, weights_i, pattern_items, alterations,
                  null_model, mc_cutoff, mc_samples, max_exact_placements,
                  L, rng):
    """Compute patient-level phi_i by histogramming null placement match sets

    Returns (phi, used_mc) where *used_mc* is True when the patient-level
    distribution was estimated by Monte-Carlo sampling rather than exact
    enumeration. Callers use this flag to apply the add-one Monte-Carlo
    correction (B*p + 1)/(B + 1) to the aggregated p-value, which keeps an
    MC-estimated tail valid (never exactly 0) and slightly conservative.
    """
    M, n_alt, _ = rel_tensor.shape
    if M == 0 or n_alt == 0:
        return {0: 1.0}, False

    # Transform trajectory in k x k matrix
    pnodes, expected = pattern_expected_matrix(pattern_items)
    k = len(pnodes)
    if k == 0 or k > n_alt:
        return {0: 1.0}, False

    # Patient cannot generate this pattern under the null
    alt_set = set(alterations)
    if not all(node in alt_set for node in pnodes):
        return {0: 1.0}, False

    # Build the null realizations h: pattern nodes -> patient alterations
    placements = None
    used_mc = False
    if M <= mc_cutoff:
        placements = build_placements_exact(n_alt, k, null_model, max_exact_placements)
    if placements is None:
        placements = build_placements_mc(n_alt, k, mc_samples, null_model, rng)
        used_mc = True

    # Number of exact placements or Monte-Carlo samples
    B_total = placements.shape[0]
    if B_total == 0:
        return {0: 1.0}, False

    # For each placement h, compute J(h): matched candidate tree indices
    match_sets = compute_match_sets(placements, rel_tensor, expected)

    # Estimate r_i(J) by counting how often each match set appears
    counter = Counter(match_sets)
    phi = defaultdict(float)
    for J_set, cnt in counter.items():
        rj = cnt / float(B_total)
        if not J_set:
            v = 0.0
        else:
            # Convert J into its weighted patient support v_i(J)
            v = sum(weights_i[j] for j in J_set)

        # Store phi_i on the L-scaled integer support grid
        exp = int(round(L * v))
        phi[exp] += rj
    return dict(phi), used_mc


# =====================================================================
# Aggregation
# =====================================================================
# Aggregate patient-level null distributions
def _phi_to_array(phi):
    """Turn a {exponent: probability} dict into a dense numpy array"""
    if not phi:
        return np.array([1.0])
    max_exp = max(phi.keys())
    arr = np.zeros(max_exp + 1, dtype=np.float64)
    for e, p in phi.items():
        arr[e] = p
    return arr

def convolve_phi(phis):
    """Convolve all patient phi_i distributions into the cohort null.

    Under the null the patients are independent, so the distribution of the
    cohort support (the sum of the per-patient supports) is exactly the
    convolution of the individual per-patient laws phi_i.
    """
    F = np.array([1.0])
    for phi in phis:
        if not phi:
            continue
        arr = _phi_to_array(phi)
        F = np.convolve(F, arr)
    return F


def tail_sum(F, threshold):
    """Return sum_{t >= threshold} F[t] for array-valued F"""
    threshold = max(0, int(threshold))
    if threshold >= len(F):
        return 0.0
    return float(F[threshold:].sum())


def compute_pvalue_pb(probs, k_obs):
    """Compute Poisson-binomial upper tail Pr[sum Bernoulli(probs) >= k_obs]"""
    probs = [p for p in probs if p > 0.0]
    n = len(probs)
    if k_obs <= 0:
        return 1.0
    if n < k_obs:
        return 0.0
    dp = np.zeros(n + 1, dtype=np.float64)
    dp[0] = 1.0
    for i, p in enumerate(probs):
        new = np.zeros(n + 1, dtype=np.float64)
        new[0] = dp[0] * (1.0 - p)
        for j in range(1, i + 2):
            new[j] = dp[j] * (1.0 - p) + dp[j - 1] * p
        dp = new
    return float(dp[k_obs:].sum())


# =====================================================================
# Integer rescaling factor L
# =====================================================================
def compute_L(weights, cap):
    """Compute the integer scaling factor used for weighted support bins.

    A weighted support is a sum of transaction weights, that is a sum of
    fractions. To represent the null as a convolution on an integer grid every
    weighted support must fall on an integer bin, so we scale by L = least common
    multiple of the weight denominators; then L * (weighted support) is always an
    integer. If the natural LCM would exceed *cap*, L is capped and the weights
    are effectively rounded to 1/cap precision.
    """
    L = 1
    for w in weights:
        if w <= 0:
            continue
        f = Fraction(w).limit_denominator(1_000_000)
        L = math.lcm(L, f.denominator)
        if L > cap:
            print(f"[WARN] natural LCM > {cap}; capping L at {cap} "
                  f"(weights rounded to {1.0/cap:.2e} precision)")
            return cap
    return L


# =====================================================================
# Dataset preprocessing
# =====================================================================
def preprocess_patient(tlist, drop_gl):
    """Transform raw patient trees into (alterations, relation tensor)"""
    trees_rel = []
    alt_set = set()
    for tr in tlist:
        edges = extract_edges_from_tree(tr)
        items = tree_edges_to_transaction_items(edges, drop_gl=drop_gl)
        rd = items_to_rel_dict(items)
        trees_rel.append(rd)
        for (x, y) in rd.keys():
            alt_set.add(x)
            alt_set.add(y)
    alterations = sorted(alt_set)
    rel_tensor = build_rel_tensor(trees_rel, alterations)
    return alterations, rel_tensor


def preprocess_patient_from_items(item_lines):
    """Transform pre-computed transaction strings into a relation tensor"""
    trees_rel = []
    alt_set = set()
    for line in item_lines:
        items = line.split()
        rd = items_to_rel_dict(items)
        trees_rel.append(rd)
        for (x, y) in rd.keys():
            alt_set.add(x)
            alt_set.add(y)
    alterations = sorted(alt_set)
    rel_tensor = build_rel_tensor(trees_rel, alterations)
    return alterations, rel_tensor


# =====================================================================
# Per-trajectory processing (parallelised across trajectories)
# =====================================================================
# Shared read-only state, populated once in main() before the worker pool
# is created. On Linux the pool is forked, so workers inherit this dict by
# copy-on-write and none of the (large) relation tensors are ever pickled.
_G = {}


def _process_trajectory(task):
    """Compute the p-value row for a single mined trajectory.

    Reads the shared cohort state from the module-level _G dict (inherited
    via fork). Returns a result row dict, or None if the pattern is not a
    valid trajectory.
    """
    idx, pl, ol = task
    G = _G

    pattern_items = parse_items_from_pattern_line(pl)
    if not pattern_items:
        return None
    pattern_items = sorted(pattern_items)
    pnodes_set, numedges = parse_pattern_nodes_and_edges(pattern_items)
    if not _G["allow_incomplete"] and not is_valid_trajectory(pnodes_set, numedges):
        return None

    occ_ids = parse_occ_ids(ol)
    # Observed supports. y_obs is quantised on the SAME L-grid as the null
    # statistic, aggregating weights per patient and rounding once per patient.
    pi_obs = [0.0] * G["n_patients"]
    for t in occ_ids:
        if 0 <= t < G["K"]:
            pi_obs[G["owner0"][t]] += G["weights"][t]
    y_obs = sum(int(round(G["L"] * v)) for v in pi_obs)
    s_exp = sum(pi_obs)
    s_theta, _ = compute_theta_support(
        occ_ids, G["weights"], G["owner0"], G["n_patients"], G["K"], G["theta"]
    )

    # Independent RNG stream per trajectory (stable regardless of ordering).
    seed_base = G["seed"] + (idx + 1) * 1_000_003

    phis = []
    used_mc = False
    for i in range(G["n_patients"]):
        info = G["patient_info"][i]
        if info is None:
            continue
        rng = np.random.default_rng(seed_base + i)
        phi_i, mc_i = compute_phi_i(
            rel_tensor=info["rel_tensor"],
            weights_i=G["patient_weights"][i],
            pattern_items=pattern_items,
            alterations=info["alterations"],
            null_model=G["null"],
            mc_cutoff=G["mc_cutoff"],
            mc_samples=G["mc_samples"],
            max_exact_placements=G["max_exact_placements"],
            L=G["L"],
            rng=rng,
        )
        phis.append(phi_i)
        used_mc = used_mc or mc_i

    # Monte-Carlo p-value correction (add-one numerator). When any patient used
    # the MC fallback the aggregated null tail is estimated from ~B samples, so
    # we report the unbiased estimator (B*p + 1)/(B + 1) rather than the raw
    # tail. Adding 1 to the numerator keeps the p-value valid (never exactly 0)
    # and slightly conservative, and it reduces to the exact tail as B -> inf;
    # when p == 0 it collapses to 1/(B + 1), matching the previous floor. Exact
    # (fully enumerated) trajectories keep their true, possibly tiny, p-value.
    B = G["mc_samples"]

    def _mc_correct(p):
        if not used_mc:
            return p
        return (B * p + 1.0) / (B + 1.0)

    pval_exp = ""
    pval_theta = ""

    if G["want_exp"]:
        F = convolve_phi(phis)
        pval_exp = f"{_mc_correct(tail_sum(F, y_obs)):.6e}"

    if G["want_theta"]:
        r_is = []
        for phi_i in phis:
            r_is.append(sum(p for e, p in phi_i.items() if e >= G["theta_threshold"]))
        pval_theta = f"{_mc_correct(compute_pvalue_pb(r_is, int(s_theta))):.6e}"

    return {
        "pattern": " ".join(pattern_items),
        "n_nodes": len(pnodes_set),
        "n_edges": numedges,
        "s_exp": f"{s_exp:.6f}",
        "s_theta": int(s_theta),
        "pval_exp": pval_exp,
        "pval_theta": pval_theta,
    }

# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Ensemble-based significance testing for MASTRO output"
    )
    ap.add_argument("-i", "--input", required=True,
                    help="Filtered result file (alternating pattern / occurrence lines)")
    ap.add_argument("-o", "--output", required=True,
                    help="Output CSV path")
    npy_grp = ap.add_mutually_exclusive_group(required=True)
    npy_grp.add_argument("--npy",
                    help="Original .npy dataset (list of trees per patient)")
    npy_grp.add_argument("--graphs_all",
                    help="Pre-computed transactions file (one line per tree, "
                         "used together with --owner and --weights)")
    ap.add_argument("-w", "--weights", default=None,
                    help="weights.txt (optional; default: all 1.0, single-tree)")
    ap.add_argument("--owner", default=None,
                    help="owner.txt (optional; default: identity 0..K-1, single-tree)")
    ap.add_argument("--null", choices=["indep", "perm"], default="perm",
                    help="Null model for alteration placement (default: perm)")
    ap.add_argument("--test", choices=["exp", "theta", "both"], default="both",
                    help="Which significance test to run (default: both)")
    ap.add_argument("--theta", type=float, default=1.0,
                    help="Theta threshold for the theta-consensus test")
    ap.add_argument("--mc_cutoff", type=int, default=8,
                    help="Use Monte-Carlo when M_i > this value (default: 8)")
    ap.add_argument("--mc_samples", type=int, default=10000,
                    help="Number of MC samples per patient in the fallback "
                         "(default: 10000)")
    ap.add_argument("--max_exact_placements", type=int, default=2_000_000,
                    help="Fall back to MC if exact enumeration exceeds this")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the Monte-Carlo RNG")
    ap.add_argument("--L_cap", type=int, default=1000,
                    help="Cap on the integer rescaling factor L; larger gives "
                         "more accuracy but bloats the exp-test convolution")
    ap.add_argument("--keep_gl", action="store_true",
                    help="Do not drop the germline GL node")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Parallel workers over trajectories (default: 1). "
                         "On Linux the pool is forked, so per-patient tensors "
                         "are shared with no pickling overhead.")
    ap.add_argument("--allow-incomplete", dest="allow_incomplete",
                    action="store_true",
                    help="Do not require the completeness constraint "
                         "numedges==C(n,2). Use for testing incomplete posets "
                         "(e.g. POTTR trajectories): only the pairwise relations "
                         "actually present in the pattern constrain the null.")
    args = ap.parse_args()

    drop_gl = not args.keep_gl
    input_path = Path(args.input)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    # Load weights/owners and build per-patient relation tensors
    if args.weights and args.owner:
        weights, owner0, n_patients, K = load_weights_and_owner(
            Path(args.weights), Path(args.owner)
        )
    else:
        # Single-tree fallback: derive K from --graphs_all (line count) or
        # --npy (sum of trees per patient). owner = identity, weights = 1.0
        if args.graphs_all:
            K = sum(1 for _ in open(args.graphs_all))
        else:
            data = load_npy(Path(args.npy))
            K = sum(len(list(tlist)) for tlist in data)
        owner0 = list(range(K))
        weights = [1.0] * K
        n_patients = K
        print(f"[INFO] no --weights/--owner given: assuming single-tree "
              f"(K={K}, w=1.0, owner=identity)")

    patient_info: list = [None] * n_patients

    if args.npy:
        data = load_npy(Path(args.npy))
        patients_trees = list(data)
        for i, tlist in enumerate(patients_trees):
            if i >= n_patients:
                break
            tlist = list(tlist)
            if len(tlist) == 0:
                continue
            alt, rt = preprocess_patient(tlist, drop_gl=drop_gl)
            patient_info[i] = {"alterations": alt, "rel_tensor": rt}
    else:
        # Group pre-computed transaction lines by patient
        graphs_lines = [l.rstrip("\n") for l in open(args.graphs_all)]
        assert len(graphs_lines) == K, (
            f"graphs_all has {len(graphs_lines)} lines but owner has {K}")
        patient_item_lines = [[] for _ in range(n_patients)]
        for t in range(K):
            patient_item_lines[owner0[t]].append(graphs_lines[t])
        for i in range(n_patients):
            if not patient_item_lines[i]:
                continue
            alt, rt = preprocess_patient_from_items(patient_item_lines[i])
            patient_info[i] = {"alterations": alt, "rel_tensor": rt}

    # Group transaction weights by patient
    patient_weights = [[] for _ in range(n_patients)]
    for t in range(K):
        patient_weights[owner0[t]].append(weights[t])

    # Check tensor/weight alignment per patient
    for i in range(n_patients):
        info = patient_info[i]
        wts = patient_weights[i]
        if info is None:
            if wts:
                raise RuntimeError(
                    f"Patient {i}: empty in dataset but has {len(wts)} weights"
                )
            continue
        if info["rel_tensor"].shape[0] != len(wts):
            raise RuntimeError(
                f"Patient {i}: rel_tensor has {info['rel_tensor'].shape[0]} "
                f"trees but {len(wts)} weights"
            )

    # Compute rescaling factor L
    L = compute_L([w for pw in patient_weights for w in pw], cap=args.L_cap)
    
    non_empty = sum(1 for p in patient_info if p is not None)

    print(f"[INFO] input       = {input_path}")
    print(f"[INFO] test        = {args.test}   null = {args.null}")
    print(f"[INFO] theta       = {args.theta}  (theta-test only)")
    print(f"[INFO] n_patients  = {n_patients}   transactions = {K}")
    print(f"[INFO] non-empty   = {non_empty}   empty slots = {n_patients - non_empty}")
    print(f"[INFO] L           = {L}")

    # Integer threshold for the theta-test
    # The small epsilon avoids floating-point issues when L time theta is exact (e.g. theta=1)
    theta_threshold = int(math.ceil(L * args.theta - 1e-12))

    want_exp = args.test in ("exp", "both")
    want_theta = args.test in ("theta", "both")

    # Publish shared read-only cohort state for the trajectory workers.
    # y_obs is quantised on the SAME L-grid as the null statistic
    # Y(P) = sum_i round(L * v_i(J_i)); see _process_trajectory for how the
    # observed statistic is aggregated on that grid.
    _G.update(
        n_patients=n_patients, K=K, L=L, owner0=owner0, weights=weights,
        patient_info=patient_info, patient_weights=patient_weights,
        null=args.null, mc_cutoff=args.mc_cutoff, mc_samples=args.mc_samples,
        max_exact_placements=args.max_exact_placements, seed=args.seed,
        theta=args.theta, theta_threshold=theta_threshold,
        want_exp=want_exp, want_theta=want_theta,
        allow_incomplete=args.allow_incomplete,
    )

    tasks = [(idx, pl, ol) for idx, (pl, ol) in enumerate(read_result_pairs(input_path))]
    print(f"[INFO] n_jobs      = {args.n_jobs}   candidate lines = {len(tasks)}")

    # Parallelise across trajectories. On Linux we fork so the workers inherit
    # _G (and its relation tensors) by copy-on-write, avoiding any pickling of
    # the cohort state; only the small (idx, pattern, occ) task is sent.
    total = len(tasks)
    step = max(1, total // 20)  # ~5% progress ticks
    results = []
    if args.n_jobs and args.n_jobs > 1 and tasks:
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = None  # non-fork platform: falls back to default (spawn re-imports module)
        with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=ctx) as ex:
            for i, r in enumerate(ex.map(_process_trajectory, tasks, chunksize=1), 1):
                results.append(r)
                if i % step == 0 or i == total:
                    print(f"[sig] {i}/{total} trajectories", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            results.append(_process_trajectory(t))
            if i % step == 0 or i == total:
                print(f"[sig] {i}/{total} trajectories", flush=True)

    rows = [r for r in results if r is not None]

    print(f"[INFO] Writing {len(rows)} rows to {output_path}")
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pattern", "n_nodes", "n_edges", "s_exp", "s_theta",
                        "pval_exp", "pval_theta"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if rows:
        if want_exp:
            vals = [float(r["pval_exp"]) for r in rows if r["pval_exp"] != ""]
            if vals:
                print(f"[DONE] min pval_exp   = {min(vals):.3e}")
        if want_theta:
            vals = [float(r["pval_theta"]) for r in rows if r["pval_theta"] != ""]
            if vals:
                print(f"[DONE] min pval_theta = {min(vals):.3e}")


if __name__ == "__main__":
    main()

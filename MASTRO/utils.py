# =============================================================================
# MASTRO Shared Utilities
#
# Common functions and constants used across multiple MASTRO scripts:
#   - breastCancer_experiments.py
#   - analyze_spurious_itemsets.py
#   - patient_mutset.py
#   - postfilter_theta.py
#   - postfilter_theta_maximal.py
# =============================================================================

import re
import subprocess
import time
from pathlib import Path

import numpy as np


# =====================================================================
# Constants
# =====================================================================
GERMLINE_LABEL = "GL"
"""Label of the germline root node in each phylogenetic tree."""

EDGE_SEPARATORS = ["->-", "-/-", "-?-"]
"""Separators used in MASTRO edge-item encoding:
  ->-  ancestor relation (a is ancestor of b)
  -/-  incomparable relation (different branches)
  -?-  uncertain relation
"""

SCRIPT_DIR = Path(__file__).resolve().parent
"""Directory containing the MASTRO scripts (for locating convert_results.py etc.)."""


# =====================================================================
# General helpers
# =====================================================================
def ensure_dir(p: Path):
    """Create directory and parents if they don't exist."""
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, cwd=None):
    """Execute a shell command, printing it first for logging."""
    print("[CMD]", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)


def load_npy(path: Path):
    """Load a .npy dataset (allow_pickle for object arrays)."""
    return np.load(path, allow_pickle=True)


# =====================================================================
# File I/O helpers
# =====================================================================
def read_floats(path):
    """Read one float per line from a text file (blank lines are skipped)."""
    with open(path, "r") as f:
        return [float(line.strip()) for line in f if line.strip()]


def read_ints(path):
    """Read one integer per line from a text file (blank lines are skipped)."""
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


# =====================================================================
# Tree / dataset helpers
# =====================================================================
def extract_edges_from_tree(tree):
    """Convert a tree array to a list of directed edges.

    Dataset format:
        data[patient_id][tree_id][edge_id] = [parent_label, child_label]

    Returns a list of (parent, child) string tuples.
    """
    return [(str(u).strip(), str(v).strip()) for u, v in tree]


def mutation_set_from_edges(edges, drop_gl=True):
    """Collect the set of unique mutation labels appearing as edge endpoints.

    If *drop_gl* is True, the germline label 'GL' is excluded.
    """
    muts = set()
    for u, v in edges:
        if not (drop_gl and u == GERMLINE_LABEL):
            muts.add(u)
        if not (drop_gl and v == GERMLINE_LABEL):
            muts.add(v)
    return muts


def mutation_set_from_tree(tree, drop_gl=False):
    """Return the set of mutation labels appearing in a tree.

    Convenience wrapper around extract_edges_from_tree + mutation_set_from_edges.
    """
    edges = extract_edges_from_tree(tree)
    return mutation_set_from_edges(edges, drop_gl=drop_gl)


def tree_edges_to_transaction_items(edges, drop_gl=True):
    """Encode the complete set of pairwise mutation relationships as items.

    Given the directed edges (u -> v) of a phylogenetic tree, we produce
    one item for each pair of mutations:
      - Ancestor relation  :  "a->-b"               (a is an ancestor of b)
      - Incomparable       :  "min(a,b)-/-max(a,b)"  (neither is ancestor)

    The lexicographic ordering for incomparable pairs ensures a canonical
    representation.  If *drop_gl* is True, the germline node GL is excluded.
    """
    # Build adjacency structures
    children = {}
    parent = {}

    for u, v in edges:
        children.setdefault(u, []).append(v)
        parent[v] = u

    all_nodes = set()
    for u, v in edges:
        all_nodes.add(u)
        all_nodes.add(v)

    if drop_gl:
        nodes = [x for x in all_nodes if x != GERMLINE_LABEL]
    else:
        nodes = list(all_nodes)

    # Pre-compute the ancestor set for every node by walking
    # parent pointers up to the root.
    ancestors = {x: set() for x in all_nodes}
    for x in all_nodes:
        cur = x
        seen = set()
        while cur in parent:
            cur = parent[cur]
            if cur in seen:
                break
            seen.add(cur)
            ancestors[x].add(cur)

    def lab(z):
        """Sanitize label: replace spaces with underscores."""
        return str(z).replace(" ", "_")

    # Enumerate all distinct pairs of (non-GL) mutations
    items = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if a in ancestors[b]:
                items.append(f"{lab(a)}->-{lab(b)}")
            elif b in ancestors[a]:
                items.append(f"{lab(b)}->-{lab(a)}")
            else:
                x, y = sorted([lab(a), lab(b)])
                items.append(f"{x}-/-{y}")

    return items


# =====================================================================
# MASTRO input file builders
# =====================================================================
def build_inputs(patients_trees, outdir: Path, seed: int, drop_gl=True):
    """Create all input files needed by the MASTRO mining pipeline.

    Outputs:
      - graphs_all.txt      : one transaction per tree (ALL trees for all patients)
      - weights_uniform.txt : w_t = 1/M_i for each transaction (for weighted FIM)
      - owner.txt           : patient index for each transaction (0-based)
      - graphs_sampled.txt  : one randomly-sampled tree per patient (for Alg 0)

    Returns (graphs_all, weights_uniform, owner, graphs_sampled) as Paths.
    """
    rng = np.random.default_rng(seed)

    graphs_all = outdir / "graphs_all.txt"
    weights_uniform = outdir / "weights_uniform.txt"
    owner = outdir / "owner.txt"
    graphs_sampled = outdir / "graphs_sampled.txt"

    # All trees: for Algorithms 1, 2, 3
    with graphs_all.open("w") as fg, \
            weights_uniform.open("w") as fw, \
            owner.open("w") as fo:
        for i, tlist in enumerate(patients_trees):
            Mi = len(tlist)
            if Mi == 0:
                continue
            w = 1.0 / Mi  # each tree contributes 1/M_i so total patient weight = 1
            for tr in tlist:
                edges = extract_edges_from_tree(tr)
                items = tree_edges_to_transaction_items(edges, drop_gl=drop_gl)
                fg.write(" ".join(items) + "\n")
                fw.write(f"{w}\n")
                fo.write(f"{i}\n")

    # One random tree per patient: for Algorithm 0
    with graphs_sampled.open("w") as fs:
        for i, tlist in enumerate(patients_trees):
            if not tlist:
                continue
            j = int(rng.integers(0, len(tlist)))
            tr = tlist[j]
            edges = extract_edges_from_tree(tr)
            items = tree_edges_to_transaction_items(edges, drop_gl=drop_gl)
            fs.write(" ".join(items) + "\n")

    return graphs_all, weights_uniform, owner, graphs_sampled


# =====================================================================
# LCM invocation helpers
# =====================================================================
def run_transnum(lcmdir: Path, table_file_ids: Path, graphs_txt: Path,
                 file_graphs_ids: Path):
    """Run transnum.pl to convert edge labels -> numeric IDs.

    transnum.pl reads transactions from stdin and writes the numeric
    version to stdout, while accumulating the label->ID mapping in
    *table_file_ids*.
    """
    print("[CMD] perl", str(lcmdir / "transnum.pl"), str(table_file_ids),
          "<", str(graphs_txt), ">", str(file_graphs_ids))
    with open(graphs_txt, "r") as fin, open(file_graphs_ids, "w") as fout:
        subprocess.run(
            ["perl", str(lcmdir / "transnum.pl"), str(table_file_ids)],
            stdin=fin, stdout=fout, check=True,
        )


def run_lcm(lcmdir: Path, file_graphs_ids: Path, sigma, output_lcm: Path,
            log_path: Path, weights_txt: Path | None = None):
    """Invoke LCM for (weighted) frequent itemset mining.

    Flags: F = frequent itemsets, f = output frequencies, I = output transaction IDs.
    When *weights_txt* is provided, LCM uses "-w <file>" for weighted
    support mode; *sigma* can then be a float.
    """
    cmd = [str(lcmdir / "lcm"), "FfI"]
    if weights_txt is not None:
        cmd += ["-w", str(weights_txt)]
        # LCM accumulates 1/M_i weights via sequential addition in C
        # A 1e-9 tolerance is negligible relative to any real support difference.
        effective_sigma = float(sigma) - 1e-9
    else:
        effective_sigma = sigma
    cmd += [str(file_graphs_ids), str(effective_sigma), str(output_lcm)]

    print("[CMD]", " ".join(cmd), ">", str(log_path), "2>&1")
    with open(log_path, "w") as logf:
        subprocess.run(cmd, stdout=logf, stderr=logf, check=True)


def run_lcm_limited(lcmdir: Path, file_graphs_ids: Path, sigma,
                    output_lcm: Path, log_path: Path,
                    weights_txt: Path | None = None,
                    max_itemsets: int | None = None,
                    timeout_sec: int | None = None):
    """Invoke LCM with optional solution cap (-# flag) and timeout.

    Returns a dict:
      - completed  : True if LCM finished normally
      - timed_out  : True if killed by timeout
      - capped     : True if the -# flag was used
      - elapsed_s  : wall-clock seconds
      - max_itemsets : the cap value (or None)
    """
    cmd = [str(lcmdir / "lcm"), "FfI"]
    if weights_txt is not None:
        cmd += ["-w", str(weights_txt)]
        effective_sigma = float(sigma) - 1e-9
    else:
        effective_sigma = sigma
    if max_itemsets is not None:
        cmd += ["-#", str(max_itemsets)]
    cmd += [str(file_graphs_ids), str(effective_sigma), str(output_lcm)]

    print("[CMD]", " ".join(cmd), ">", str(log_path), "2>&1")

    t0 = time.time()
    timed_out = False

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=logf)
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            print(f"  [TIMEOUT] LCM killed after {timeout_sec}s")
            proc.kill()
            proc.wait()

    elapsed = time.time() - t0

    return {
        "completed": proc.returncode == 0 and not timed_out,
        "timed_out": timed_out,
        "capped": max_itemsets is not None,
        "elapsed_s": round(elapsed, 2),
        "max_itemsets": max_itemsets,
    }


# =====================================================================
# Result parsing helpers
# =====================================================================
def parse_items_from_pattern_line(pattern_line: str):
    """Extract edge-items from a result pattern line.

    Pattern lines look like:  'A->-B A-/-C (42)'
    The '(...)' annotation (support count) is stripped, and the remaining
    tokens are returned as a frozenset.
    """
    s = re.sub(r"\(.*?\)", "", pattern_line).strip()
    if not s:
        return frozenset()
    return frozenset(tok for tok in s.split() if tok.strip())


def parse_occ_ids(line):
    """Parse space-separated transaction IDs from a single line.

    Returns a list of ints.
    """
    return [int(tok) for tok in line.strip().split() if tok.strip().isdigit()]


def read_result_pairs(result_path: Path):
    """Read all (pattern_line, occ_line) pairs from a result file.

    The file alternates pattern lines (items + support) with occurrence-list
    lines.  Yields (pattern_line, occ_line) tuples.
    """
    if not result_path.exists():
        return

    with result_path.open("r") as f:
        while True:
            pl = f.readline()
            if not pl:
                break
            ol = f.readline()
            if not ol:
                break
            if "(" not in pl:
                continue
            yield pl, ol


def read_result_itemsets(result_path: Path):
    """Read all mined patterns from a result file.

    Returns a list of frozensets (one per pattern).
    """
    itemsets = []
    for pl, _ol in read_result_pairs(result_path):
        items = parse_items_from_pattern_line(pl)
        if items:
            itemsets.append(items)
    return itemsets


def parse_pattern_nodes_and_edges(items_raw, separators=None):
    """Extract mutation nodes and edge count from a list of edge-item strings.

    Returns (nodes: set, numedges: int).
    """
    if separators is None:
        separators = EDGE_SEPARATORS

    nodes = set()
    numedges = 0
    for item in items_raw:
        for sep in separators:
            if sep in item:
                parts = item.split(sep)
                nodes.add(parts[0])
                nodes.add(parts[1])
                numedges += 1
                break
    return nodes, numedges


def is_valid_trajectory(nodes, numedges):
    """Check whether an itemset corresponds to a valid trajectory.

    An itemset is valid if the number of edges equals C(n,2) = n*(n-1)/2,
    i.e. every pair of mutations has a defined relationship.
    """
    n = len(nodes)
    return numedges > 0 and numedges == n * (n - 1) / 2


# =====================================================================
# Weights / owner loading (for postfilter scripts)
# =====================================================================
def load_weights_and_owner(weights_path, owner_path):
    """Load and validate weights and owner vectors.

    Returns (weights, owner0, n_patients, K) where:
      - weights : list[float] of transaction weights
      - owner0  : list[int] of 0-based patient IDs per transaction
      - n_patients : number of distinct patients
      - K       : total number of transactions
    """
    weights = read_floats(weights_path)
    owner = read_ints(owner_path)

    K = len(weights)
    if len(owner) != K:
        raise ValueError(f"Mismatch: weights lines={K} vs owner lines={len(owner)}")
    if K == 0:
        raise ValueError("Empty weights/owner files")

    # Normalize to 0-based
    shift = 1 if min(owner) == 1 else 0
    owner0 = [x - shift for x in owner]
    if min(owner0) != 0:
        raise ValueError(
            "Owner ids must be 0-based or 1-based contiguous starting at 0/1"
        )

    n_patients = max(owner0) + 1
    return weights, owner0, n_patients, K


def count_lines(path) -> int:
    """Return the number of lines in a text file."""
    with open(path, "r") as f:
        return sum(1 for _ in f)


def compute_dataset_stats(patients_trees, drop_gl=True):
    """Compute summary statistics about the dataset.

    Reports: number of patients, trees per patient (min/median/max/mean),
    fraction of patients where all trees share the same mutation set,
    mutation counts per tree, and global unique mutation count.
    """
    n = len(patients_trees)
    Mi = [len(tlist) for tlist in patients_trees]
    Mi_arr = np.array(Mi)

    same_mutset_per_patient = 0
    patient_mut_counts = []
    tree_mut_counts = []
    global_mut_union = set()

    for tlist in patients_trees:
        mutsets = []
        for tr in tlist:
            edges = extract_edges_from_tree(tr)
            muts = mutation_set_from_edges(edges, drop_gl=drop_gl)
            mutsets.append(muts)
            tree_mut_counts.append(len(muts))
            global_mut_union |= muts

        if mutsets:
            inter = set.intersection(*mutsets)
            uni = set.union(*mutsets)
            patient_mut_counts.append((len(inter), len(uni)))
            if inter == uni:
                same_mutset_per_patient += 1

    return {
        "n_patients": int(n),
        "drop_gl": bool(drop_gl),
        "trees_per_patient": {
            "min": int(Mi_arr.min()),
            "median": float(np.median(Mi_arr)),
            "max": int(Mi_arr.max()),
            "mean": float(Mi_arr.mean()),
            "p25": float(np.percentile(Mi_arr, 25)),
            "p75": float(np.percentile(Mi_arr, 75)),
        },
        "fraction_patients_same_mutation_set_across_trees": float(same_mutset_per_patient / max(1, n)),
        "global_unique_mutations": int(len(global_mut_union)),
        "tree_mutations": {
            "min": int(np.min(tree_mut_counts)) if tree_mut_counts else 0,
            "median": float(np.median(tree_mut_counts)) if tree_mut_counts else 0,
            "max": int(np.max(tree_mut_counts)) if tree_mut_counts else 0,
        },
        "patient_mutations_intersection_union_summary": {
            "min_intersection": int(min(x[0] for x in patient_mut_counts)) if patient_mut_counts else 0,
            "min_union": int(min(x[1] for x in patient_mut_counts)) if patient_mut_counts else 0,
            "max_union": int(max(x[1] for x in patient_mut_counts)) if patient_mut_counts else 0,
        }
    }


def compute_theta_support(occ_ids, weights, owner0, n_patients, K, theta):
    """Compute theta-support for a pattern given its occurrence IDs.

    Returns (s_theta, pi) where:
      - s_theta : number of patients with weighted presence >= theta
      - pi      : list[float] per-patient weighted presence
    """
    pi = [0.0] * n_patients
    for t in occ_ids:
        if 0 <= t < K:
            pi[owner0[t]] += weights[t]
    # 1e-9 tolerance absorbs floating-point accumulation error
    s_theta = sum(1 for val in pi if val >= theta - 1e-9)
    return s_theta, pi

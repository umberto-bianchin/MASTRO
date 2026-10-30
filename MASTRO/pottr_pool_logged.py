"""
Run POTTR's run_POTTR.py unchanged, and log its Gurobi solution pool before
POTTR discards part of it.

With -pool, find_max_k_common_trajectory (POTTR/code/POTTR.py) keeps only the
pool solutions of maximum size and, among those, the first one per *set of
alterations* (`if solution_nodes not in node_selection`): two optima on the
same alterations in different orders (A->-B vs B->-A) collapse to whichever
Gurobi lists first. This wrapper reads the whole pool after that call returns
and writes, for one k, how many solutions Gurobi returned, whether the pool is
full, and every distinct (alterations, order) trajectory in it, marking the
ones POTTR keeps. POTTR's own outputs are not touched.

Run from POTTR/code, with the arguments of run_POTTR.py after --pool_log:
    cd ../../POTTR/code && python3 <repo>/MASTRO/pottr_pool_logged.py \\
        --pool_log <out>/k5/pool_log.json -o <out>/k5 -d <dags> -k 5 -pool 50000 -c 20
"""

import json
import os
import runpy
import sys

ROOT_LABEL = "0"


class _ModelRecorder:
    """Stands in for the gurobipy module inside POTTR.py: forwards everything
    and keeps a handle on each Model it creates."""

    def __init__(self, gp):
        self._gp = gp
        self.models = []

    def __getattr__(self, name):
        return getattr(self._gp, name)

    def Model(self, *args, **kwargs):
        m = self._gp.Model(*args, **kwargs)
        self.models.append(m)
        return m


def read_pool(m, union_conflict_graph, input_graphs, pool_size):
    graphs = {g.name: (patient, g) for patient in input_graphs for g in input_graphs[patient]}
    node_vars = [v for v in m.getVars() if v.VarName in union_conflict_graph.nodes]
    graph_vars = [v for v in m.getVars() if v.VarName in graphs]
    max_size = int(m.ObjVal)

    trajs = {}           # (nodes, edges) -> entry
    kept_nodes = []      # POTTR's node_selection, in pool order
    n_optimal = 0
    for i in range(m.SolCount):
        m.setParam("SolutionNumber", i)
        if int(m.PoolObjVal) != max_size:
            continue
        n_optimal += 1
        nodes = sorted(v.VarName for v, x in zip(node_vars, m.getAttr("Xn", node_vars)) if x > 0.5)
        sel = [v.VarName for v, x in zip(graph_vars, m.getAttr("Xn", graph_vars)) if x > 0.5]
        # the order POTTR would build: union of the selected trees' edges among the nodes
        edges = sorted({f"{u}->-{w}" for name in sel for u, w in graphs[name][1].edges
                        if u in nodes and w in nodes and ROOT_LABEL not in (u, w)})
        key = (tuple(nodes), tuple(edges))
        e = trajs.setdefault(key, {"nodes": nodes, "edges": edges, "n_solutions": 0,
                                   "trees": set(), "kept_by_pottr": False})
        e["n_solutions"] += 1
        e["trees"].update(sel)
        if nodes not in kept_nodes:
            kept_nodes.append(nodes)
            e["kept_by_pottr"] = True

    out = []
    for e in trajs.values():
        trees = sorted(e.pop("trees"))
        e["n_trees_seen"] = len(trees)
        e["n_patients_seen"] = len({graphs[t][0] for t in trees})
        out.append(e)
    return {
        "max_size": max_size,
        "pool_size": pool_size,
        "sol_count": m.SolCount,
        "n_optimal": n_optimal,
        "pool_full": pool_size > 0 and m.SolCount >= pool_size,
        "n_node_sets": len(kept_nodes),
        "n_trajectories": len(out),
        "trajectories": sorted(out, key=lambda e: -e["n_solutions"]),
    }


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "--pool_log":
        sys.exit(__doc__)
    log_path = sys.argv[2]
    argv = sys.argv[3:]

    # POTTR's modules come from the current directory (POTTR/code), not from here
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != here]
    sys.path.insert(0, os.getcwd())
    import POTTR

    rec = _ModelRecorder(POTTR.gp)
    POTTR.gp = rec
    orig = POTTR.find_max_k_common_trajectory
    logs = []

    def logged(union_conflict_graph, input_graphs, k, cores, solution_pool_size=5000, verbose=False):
        res = orig(union_conflict_graph, input_graphs, k, cores,
                   solution_pool_size=solution_pool_size, verbose=verbose)
        entry = read_pool(rec.models[-1], union_conflict_graph, input_graphs, solution_pool_size)
        entry["k"] = k
        logs.append(entry)
        print(f"[POOL] k={k} max_size={entry['max_size']} solutions={entry['sol_count']}"
              f" (full={entry['pool_full']}) node_sets={entry['n_node_sets']}"
              f" trajectories={entry['n_trajectories']}", flush=True)
        return res

    POTTR.find_max_k_common_trajectory = logged
    sys.argv = ["run_POTTR.py"] + argv
    try:
        runpy.run_path(os.path.join(os.getcwd(), "run_POTTR.py"), run_name="__main__")
    finally:
        if logs:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(logs[0] if len(logs) == 1 else logs, f, indent=1)


if __name__ == "__main__":
    main()

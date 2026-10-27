"""
Significant breastCancer trajectories drawn as trees: one
figure per sigma, every trajectory single-tree MASTRO certifies in at least one
of its 10 seeds plus every trajectory MAESTRI certifies, ranked by p-value.
Crimson: significant for MAESTRI and in fewer than all MASTRO seeds (where
MAESTRI adds something); black: significant for MAESTRI and in every MASTRO
seed; grey: significant for MASTRO only.

Each entry reads
    rank)  p = MAESTRI expected-support p-value    s = expected support
           MASTRO k/10   s_exp theta=0.5 theta=1   (MAESTRI families, crimson
                                                    where significant)
A trajectory outside MAESTRI's mined family has no MAESTRI p-value; its header
shows instead the smallest p-value and the support over the MASTRO seeds that
certify it, in grey and marked with a dagger. Trajectories significant for
MAESTRI and in no MASTRO seed carry a "new" badge.

Everything is recomputed from the raw p-values and the Westfall-Young resample
minima with the tie-aware utils.wy_threshold, not read from the
analyze_mm_vs_baseline.py tables (several saved wy_thresholds.txt predate the
tie handling; on sigma=2 seed 6 that changes the significant set).

  MAESTRI:  <discovery>/breastCancer_sigma<S>/significance/
                alg1_expected_uniform_pvalues_exp.csv, alg3_theta<T>_pvalues_theta.csv
            <discovery>/breastCancer_wy_sigma<S>_theta<T>/min_pvalues.csv
  MASTRO:   <seeds>/sigma<S>/seed<k>/significance/alg0_mastro_random_pvalues_exp.csv
            <seeds>/sigma<S>/seed<k>_wy/min_pvalues.csv

Output, per sigma: a standalone TikZ file (LuaLaTeX, Carlito, same palette as
Figures/method.tex) and a CSV with one row per drawn trajectory.

    python3 plot_trajectory_trees.py --outdir results/trajectory_trees
"""

import argparse
import csv
import math
from pathlib import Path

import utils

THETAS = ("0.5", "1.0")
FAMILIES = ("exp",) + THETAS
FAMILY_TAG = {"exp": r"$s_{\exp}$", "0.5": r"$\theta{=}0.5$", "1.0": r"$\theta{=}1$"}

# layout, cm. UNITS x UNIT_W + gaps ~ 15.9 cm: ~0.77 scale on the LNCS text width
UNITS = 4
UNIT_W = 3.8
GAP_X = 0.2
ROW_GAP = 0.32
HEAD_H = 0.66        # two header lines
STEP_X = 1.2         # level to level
BOX_W = 0.98
ROOT_X = 0.16        # root circle centre, from the entry's left edge
FIRST_X = 0.42 + BOX_W / 2
SLOT_H = 0.56        # leaf to leaf
TAG_X = (1.45, 1.93, 2.65)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def parse_pattern(pattern):
    """'A->-B C-/-D' -> canonical frozenset of (a, rel, b); -/- pairs sorted."""
    items = []
    for tok in pattern.split():
        for sep in ("->-", "-/-", "-?-"):
            if sep in tok:
                a, b = tok.split(sep)
                if sep != "->-":
                    a, b = sorted((a, b))
                items.append((a, sep, b))
                break
        else:
            raise ValueError(f"unparsable item {tok!r} in {pattern!r}")
    return frozenset(items)


def read_pvalues(path, column):
    """{pattern: (p, s_exp)}"""
    with open(path) as fh:
        return {parse_pattern(r["pattern"]): (float(r[column]), float(r["s_exp"]))
                for r in csv.DictReader(fh) if r[column] not in ("", "nan")}


def read_minima(path, column):
    with open(path) as fh:
        return [float(r[column]) for r in csv.DictReader(fh)]


def significant(pvals, minima, alpha):
    t = utils.wy_threshold(minima, alpha)
    return {p for p, (v, _) in pvals.items() if v <= t}


def collect(discovery, seeds, sigma, alpha, n_seeds):
    """{pattern: {'seeds': {k: (p, s)}, 'maestri': {family...}, 'mm': (p, s) or None}}"""
    sig_dir = Path(discovery) / f"breastCancer_sigma{sigma}" / "significance"
    wy = lambda t: Path(discovery) / f"breastCancer_wy_sigma{sigma}_theta{t}" / "min_pvalues.csv"

    mm_exp = read_pvalues(sig_dir / "alg1_expected_uniform_pvalues_exp.csv", "pval_exp")
    maestri = {"exp": significant(mm_exp, read_minima(wy(THETAS[0]), "min_pval_exp"), alpha)}
    for t in THETAS:
        maestri[t] = significant(
            read_pvalues(sig_dir / f"alg3_theta{t}_pvalues_theta.csv", "pval_theta"),
            read_minima(wy(t), "min_pval_theta"), alpha)

    out = {}
    new = lambda: {"seeds": {}, "maestri": set()}
    for fam, pats in maestri.items():
        for p in pats:
            out.setdefault(p, new())["maestri"].add(fam)
    for k in range(n_seeds):
        d = Path(seeds) / f"sigma{sigma}"
        pv = read_pvalues(d / f"seed{k}" / "significance" / "alg0_mastro_random_pvalues_exp.csv",
                          "pval_exp")
        for p in significant(pv, read_minima(d / f"seed{k}_wy" / "min_pvalues.csv",
                                             "min_pval_exp"), alpha):
            out.setdefault(p, new())["seeds"][k] = pv[p]
    for p, e in out.items():
        e["mm"] = mm_exp.get(p)
    return out


# ---------------------------------------------------------------------
# Tree of a trajectory
# ---------------------------------------------------------------------
def build_tree(pattern):
    """Hasse diagram of the ->- order: {gene: parent or None}. Every pattern in
    the data is a forest under the germline (no gene has two incomparable
    immediate predecessors); anything else is an error, not drawn."""
    genes = sorted({g for a, _, b in pattern for g in (a, b)})
    anc = {g: {a for a, rel, b in pattern if rel == "->-" and b == g} for g in genes}
    parent = {}
    for g in genes:
        direct = [a for a in anc[g] if not any(a in anc[c] for c in anc[g])]
        if len(direct) > 1:
            raise ValueError(f"{g} has two immediate predecessors in {sorted(pattern)}")
        parent[g] = direct[0] if direct else None
    return parent


def children(parent):
    kids = {}
    for g, p in parent.items():
        kids.setdefault(p, []).append(g)
    for v in kids.values():
        v.sort()
    return kids


def label(parent):
    """Text form, e.g. 'CDH1 < {PIK3CA < GATA3 | TP53}' (for the CSV only)."""
    kids = children(parent)

    def sub(p):
        parts = [g + (" < " + sub(g) if g in kids else "") for g in kids[p]]
        return parts[0] if len(parts) == 1 else "{" + " | ".join(parts) + "}"
    return sub(None)


def layout(parent):
    """(level, slot) per gene, germline at level 0; leaves take consecutive
    slots top to bottom, an inner node sits level with the middle of its
    children. Returns (positions, number of leaves, depth)."""
    kids = children(parent)
    pos, nxt = {}, [0]

    def place(g, lvl):
        if g not in kids:
            y = nxt[0]
            nxt[0] += 1
        else:
            ys = [place(c, lvl + 1) for c in kids[g]]
            y = (ys[0] + ys[-1]) / 2
        pos[g] = (lvl, y)
        return y

    place(None, 0)
    return pos, nxt[0], max(lvl for lvl, _ in pos.values())


# ---------------------------------------------------------------------
# TikZ
# ---------------------------------------------------------------------
PREAMBLE = r"""% !TEX TS-program = LuaLaTeX
% Generated by MASTRO/plot_trajectory_trees.py - do not edit by hand.
\documentclass[tikz, border=4pt]{standalone}

\usepackage{tikz}
\usepackage{amsmath, amssymb}
\usepackage{fontspec}
\setmainfont{Carlito}
\setsansfont{Carlito}

\definecolor{crimson}{HTML}{9B0000}
\definecolor{ink}{HTML}{262626}
\definecolor{bodygray}{HTML}{5A5A5A}

\tikzset{
  root/.style={circle, draw=ink, line width=0.6pt, fill=white,
               inner sep=0pt, minimum size=2.6mm},
  gene/.style={rectangle, draw=ink, line width=0.6pt, fill=white,
               inner xsep=2.4pt, inner ysep=1pt, minimum width=9.8mm, minimum height=4.0mm,
               font=\sffamily\bfseries\scriptsize, text=ink},
  hlgene/.style={gene, draw=crimson, line width=1.0pt, text=crimson},
  ed/.style={line width=0.6pt, ink},
  hled/.style={line width=1.0pt, crimson},
  lowgene/.style={gene, draw=gray!55, text=gray!70!black},
  lowed/.style={line width=0.6pt, gray!55},
  head/.style={font=\scriptsize, text=ink, anchor=base west},
  greyhead/.style={head, text=gray!60!black},
  sub/.style={font=\sffamily\tiny, text=bodygray, anchor=base west},
  tagon/.style={font=\sffamily\tiny\bfseries, text=crimson, anchor=base west},
  tagoff/.style={font=\sffamily\tiny, text=gray!45, anchor=base west},
  newbadge/.style={fill=crimson, text=white, font=\sffamily\tiny\bfseries,
                   inner sep=1.2pt, anchor=base west},
}

\begin{document}
\begin{tikzpicture}
"""


def sci(p):
    """2.54e-06 -> 2.54 \\cdot 10^{-6}"""
    e = math.floor(math.log10(p))
    m = p / 10 ** e
    if round(m, 2) >= 10:
        m, e = m / 10, e + 1
    return rf"{m:.2f} \cdot 10^{{{e}}}"


def fmt_s(s):
    return f"{s:.0f}" if abs(s - round(s)) < 1e-9 else f"{s:.1f}"


def header(entry):
    """(p, s, from_maestri) shown in the header."""
    if entry["mm"] is not None:
        return entry["mm"] + (True,)
    p, s = min(entry["seeds"].values())
    return p, s, False


def entry_size(entry):
    _, leaves, depth = entry["layout"]
    width = FIRST_X + (depth - 1) * STEP_X + BOX_W / 2
    span = max(1, math.ceil((width - 0.05) / UNIT_W))
    return span, HEAD_H + leaves * SLOT_H


def draw_entry(x0, y0, rank, entry, n_seeds):
    """Entry with top-left corner (x0, y0)."""
    hl = bool(entry["maestri"]) and len(entry["seeds"]) < n_seeds
    p, s, from_mm = header(entry)
    L = []
    hs = "head" if from_mm else "greyhead"
    dag = "" if from_mm else r"$^\dagger$"
    L.append(rf"\node[{hs}] at ({x0:.3f},{y0 - 0.26:.3f}) "
             rf"{{{{\tiny {rank})}}\enspace$p = {sci(p)}${dag}\enspace\enspace$s = {fmt_s(s)}$}};")
    sy = y0 - 0.52
    if hl and not entry["seeds"]:
        L.append(rf"\node[newbadge] at ({x0 + 3.25:.3f},{sy:.3f}) {{new}};")
    L.append(rf"\node[sub] at ({x0 + 0.28:.3f},{sy:.3f}) "
             rf"{{MASTRO {len(entry['seeds'])}/{n_seeds}}};")
    for i, fam in enumerate(FAMILIES):
        st = "tagon" if fam in entry["maestri"] else "tagoff"
        L.append(rf"\node[{st}] at ({x0 + TAG_X[i]:.3f},{sy:.3f}) {{{FAMILY_TAG[fam]}}};")

    pos, _, _ = entry["layout"]
    ty = y0 - HEAD_H - SLOT_H / 2

    def xy(g):
        lvl, slot = pos[g]
        x = x0 + (ROOT_X if g is None else FIRST_X + (lvl - 1) * STEP_X)
        return x, ty - slot * SLOT_H

    edst, nst = (("hled", "hlgene") if hl else
                 ("ed", "gene") if entry["maestri"] else ("lowed", "lowgene"))
    name = {None: "r", **{g: f"g{i}" for i, g in enumerate(entry["tree"])}}
    L.append(r"\node[root] (r) at ({:.3f},{:.3f}) {{}};".format(*xy(None)))
    for g in entry["tree"]:
        L.append(r"\node[{}] ({}) at ({:.3f},{:.3f}) {{{}}};".format(nst, name[g], *xy(g), g))
    for g, par in entry["tree"].items():
        L.append(rf"\draw[{edst}] ({name[par]}.east) -- ({name[g]}.west);")
    return L


def figure(entries, n_seeds):
    """Flow layout: entries left to right, ranked, a row wraps at UNITS units."""
    L = [PREAMBLE]
    y, col, row_h = 0.0, 0, 0.0
    for rank, e in enumerate(entries, 1):
        span, h = entry_size(e)
        if col + span > UNITS:
            y -= row_h + ROW_GAP
            col, row_h = 0, 0.0
        L += draw_entry(col * (UNIT_W + GAP_X), y, rank, e, n_seeds)
        col += span
        row_h = max(row_h, h)
    L += [r"\end{tikzpicture}", r"\end{document}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--discovery", default="results/exact")
    ap.add_argument("--seeds", default="results/singletree_seeds")
    ap.add_argument("--sigmas", default="2,5")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--outdir", default="results/trajectory_trees")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for s in (int(v) for v in args.sigmas.split(",")):
        data = collect(args.discovery, args.seeds, s, args.alpha, args.n_seeds)
        for p, e in data.items():
            e["pattern"] = p
            e["tree"] = build_tree(p)
            e["label"] = label(e["tree"])
            e["layout"] = layout(e["tree"])
        # MASTRO's own order: by p-value; entries outside MAESTRI's family last
        entries = sorted(data.values(), key=lambda e: (not header(e)[2], header(e)[0]))

        tex = outdir / f"trajectory_trees_sigma{s}.tex"
        tex.write_text(figure(entries, args.n_seeds))
        with open(tex.with_suffix(".csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "trajectory", "pattern", "maestri_pval_exp", "maestri_s_exp",
                        "mastro_seeds", "mastro_seed_ids", "mastro_best_pval", "mastro_best_s"]
                       + [f"maestri_{f}" for f in FAMILIES])
            for rank, e in enumerate(entries, 1):
                best = min(e["seeds"].values()) if e["seeds"] else ("", "")
                mm = e["mm"] or ("", "")
                w.writerow([rank, e["label"],
                            " ".join(f"{a}{r}{b}" for a, r, b in sorted(e["pattern"])),
                            *mm, len(e["seeds"]), ";".join(map(str, sorted(e["seeds"]))), *best]
                           + [int(f in e["maestri"]) for f in FAMILIES])
        print(f"sigma={s}: {len(entries)} drawn, "
              f"MASTRO>=1 seed {sum(bool(e['seeds']) for e in entries)}, "
              + ", ".join(f"MAESTRI {f} {sum(f in e['maestri'] for e in entries)}"
                          for f in FAMILIES)
              + f", outside MAESTRI family {sum(e['mm'] is None for e in entries)} -> {tex}")


if __name__ == "__main__":
    main()

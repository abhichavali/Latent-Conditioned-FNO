"""Canonical equation graphs for the encoder: one expression tree per additive term of each field's RHS.

Canonical form: symbols renamed by position (fields f0.. in spec["fields"] order, statics s0.. in spec["static"]
order, so target names eta/u/v, b/s map onto the same slots as the data's u/v/w, a), sympy expand, linear operators
(dx dy lap) distributed over sums with numeric factors pulled out, then commutative args (add, mul) sorted.
Node features: type (field / static / const / op), op id, slot id, and for constants (sign, log10|c|), never raw values.
"""

import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import sympy as sp

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_collection"))
import pde  # noqa: E402

F_MAX, S_MAX = 3, 2
OPS = ["add", "mul", "pow", "dx", "dy", "lap", "sin", "cos", "exp", "tanh", "log", "abs", "other"]
TYPES = ["field", "static", "const", "op"]
NODE_DIM = len(TYPES) + len(OPS) + F_MAX + S_MAX + 2


def _lin(e):
    """Distribute dx/dy/lap over sums and pull numeric factors out: dx(2*u*v - b*u) -> 2*dx(u*v) - dx(b*u)."""
    if e.is_Atom or e.is_number:
        return e
    args = [_lin(a) for a in e.args]
    if type(e).__name__ in pde.OPS:
        out = 0
        for t in sp.Add.make_args(sp.expand(args[0])):
            c, m = t.as_coeff_Mul()
            if not m.is_number:  # derivative of a constant is 0
                out += c * type(e)(m)
        return out
    return e.func(*args)


def canonical(spec):
    """spec -> [(target_slot, sympy term)] with symbols f0.., s0.. ."""
    syms, rhs = pde.parse(spec)
    nf = len(spec["fields"])
    ren = {s: sp.Symbol(f"f{i}" if i < nf else f"s{i - nf}") for i, s in enumerate(syms)}
    out = []
    for k, r in enumerate(rhs):
        r = sp.expand(_lin(sp.expand(r.xreplace(ren))))
        out += [(k, t) for t in sp.Add.make_args(r) if t != 0]
    return out


def _tree(e, nodes, edges, parent=None, pos=0):
    i = len(nodes)
    if e.is_Symbol:
        n = str(e)
        nodes.append(("field" if n[0] == "f" else "static", int(n[1:])))
    elif e.is_number:
        nodes.append(("const", float(e)))
    else:
        op = type(e).__name__.lower()
        nodes.append(("op", op if op in OPS else "other"))
    if parent is not None:
        edges.append((i, parent, pos))
    if not (e.is_Atom or e.is_number):
        kids = sorted(e.args, key=sp.default_sort_key) if (e.is_Add or e.is_Mul) else e.args
        for p, a in enumerate(kids):
            _tree(a, nodes, edges, i, p if e.is_Pow else 0)


def graph(spec):
    """Canonical graph: {"terms": [{"slot": target field slot, "nodes": [(type, val)], "edges": [(child, parent, pos)]}],
    "text": [str(term)]}.  Pure python / picklable."""
    terms, text = [], []
    for k, t in canonical(spec):
        nodes, edges = [], []
        _tree(t, nodes, edges)
        terms.append({"slot": k, "nodes": nodes, "edges": edges})
        text.append(f"d{k}: {t}")
    return {"terms": terms, "text": text}


def node_features(node):
    typ, val = node
    x = np.zeros(NODE_DIM, np.float32)
    x[TYPES.index(typ)] = 1
    o = len(TYPES)
    if typ == "op":
        x[o + OPS.index(val)] = 1
    o += len(OPS)
    if typ == "field":
        x[o + val] = 1
    o += F_MAX
    if typ == "static":
        x[o + val] = 1
    o += S_MAX
    if typ == "const":
        x[o] = np.sign(val)
        x[o + 1] = np.log10(abs(val)) / 3 if val != 0 else -4.0
    return x


def _graph_of(args):
    gid, spec = args
    return gid, graph(spec)


def cached_graphs(items, path, workers=16):
    """items [(gid, spec)] -> {gid: graph}, computed once (sympy is slow) and cached in a pickle at path."""
    cache = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    key = lambda s: repr((s["fields"], s.get("static", []), s["rhs"]))  # recompute if the equation changed
    todo = [(g, s) for g, s in items if g not in cache or cache[g].get("key") != key(s)]
    if todo:
        with Pool(workers) as p:
            cache.update((g, {**gr, "key": key(s)}) for (g, gr), (_, s) in zip(p.map(_graph_of, todo, chunksize=8), todo))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        pickle.dump(cache, open(tmp, "wb"))
        os.replace(tmp, path)
    return {g: cache[g] for g, _ in items}


def pad(graphs, nmax=None, emax=None, tmax=None):
    """List of graphs -> padded numpy arrays for the encoder (one row per equation)."""
    nn = [sum(len(t["nodes"]) for t in g["terms"]) for g in graphs]
    tt = [len(g["terms"]) for g in graphs]
    nmax = nmax or max(nn)
    emax = emax or max(max(n - t, 1) for n, t in zip(nn, tt))
    tmax = tmax or max(tt)
    E = len(graphs)
    out = {"x": np.zeros((E, nmax, NODE_DIM), np.float32), "node_term": np.zeros((E, nmax), np.int32),
           "node_mask": np.zeros((E, nmax), np.float32), "edges": np.zeros((E, emax, 3), np.int32),
           "edge_mask": np.zeros((E, emax), np.float32), "term_slot": np.zeros((E, tmax), np.int32),
           "term_mask": np.zeros((E, tmax), np.float32)}
    for e, g in enumerate(graphs):
        n0 = ne = 0
        assert len(g["terms"]) <= tmax and sum(len(t["nodes"]) for t in g["terms"]) <= nmax, "graph larger than padding"
        for j, t in enumerate(g["terms"]):
            k = len(t["nodes"])
            out["x"][e, n0 : n0 + k] = [node_features(n) for n in t["nodes"]]
            out["node_term"][e, n0 : n0 + k] = j
            out["node_mask"][e, n0 : n0 + k] = 1
            for c, p, pos in t["edges"]:
                out["edges"][e, ne] = (n0 + c, n0 + p, pos)
                out["edge_mask"][e, ne] = 1
                ne += 1
            out["term_slot"][e, j] = t["slot"]
            out["term_mask"][e, j] = 1
            n0 += k
    return out


if __name__ == "__main__":  # self-check: renaming, linear-operator distribution and ordering are canonical
    a = {"fields": ["eta", "u"], "static": ["b"], "rhs": {"eta": "-dx((eta - b)*u) + 8*lap(eta)", "u": "-9.81*dx(eta)"}}
    b = {"fields": ["u", "v"], "static": ["a"], "rhs": {"u": "8*lap(u) + dx(a*v) - dx(v*u)", "v": "-9.81*dx(u)"}}
    ga, gb = graph(a), graph(b)
    assert sorted(ga["text"]) == sorted(gb["text"]), (ga["text"], gb["text"])
    assert len(ga["terms"]) == 4 and {t["slot"] for t in ga["terms"]} == {0, 1}
    P = pad([ga, gb])
    assert P["x"].shape[-1] == NODE_DIM and P["term_mask"].sum() == 8
    print(ga["text"])
    print("canon.py ok")

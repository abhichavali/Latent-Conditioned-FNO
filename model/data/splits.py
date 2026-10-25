"""Seeded, config-driven splits, identical logic for every dataset root:

  rehearsal  held out BY STRUCTURE: a handful of library-term pairs / triples (template slot names "kind:item", as in
             coverage.py); EVERY equation containing one of them is held out.  Each term of a chosen structure still
             appears in training on its own, so rehearsal tests unseen combinations of seen building blocks, like the
             SWE / AP targets do.  Terms that make up the SWE / AP pieces are never used, so target-relevant
             combinations stay in training.
  val_ic     ~10% of the IC groups (windows of one pool IC stay together) of each training equation.
  train      everything else.  The SWE / AP targets live outside the datasets and never enter training.
"""

import itertools
import os
import sys
from collections import Counter

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_collection"))
from template import LIB  # noqa: E402

# library slots that define the SWE / AP pieces of sample.pieces(): kept out of rehearsal structures
PIECE_TERMS = {"X:u*v", "Y:u*w", "P:v*{f}x", "P:w*{f}y", "X:u", "Y:u", "X:a*v", "Y:a*w",
               "P:u*u*u", "P:u*v", "P:R*v", "P:R*u*v", "P:R*v*v"}


def term_names(spec):
    """Library terms present (kind:item, coverage.py naming), read off the template coefficients."""
    t = spec["template"]
    lib, n = LIB[t["nf"]], len(spec["fields"])
    names = set()
    for i, k, c in t["coef"]:
        if i < n and c != 0:
            kind, item = lib[k]
            names.add(f"{kind}:{item if isinstance(item, str) else '|'.join(item)}")
    return names


def choose_structures(sets, frac, band, seed, max_size=3, min_train=10):
    """Greedy seeded pick of term pairs / triples until their equations cover ~frac of the set."""
    E = len(sets)
    count = Counter(x for s in sets for x in s)
    ok = {x for x, c in count.items() if c <= 0.25 * E and x not in PIECE_TERMS and not x.startswith("L:")}
    cands = []
    for k in range(2, max_size + 1):
        combos = Counter(c for s in sets for c in itertools.combinations(sorted(s & ok), k))
        cands += [c for c, n in combos.items() if band[0] * E <= n <= band[1] * E]
    r = np.random.default_rng(seed)
    r.shuffle(cands)
    held, chosen = set(), []
    for c in cands:
        hit = {i for i, s in enumerate(sets) if set(c) <= s}
        new = held | hit
        if len(new) > 1.3 * frac * E:
            continue
        # every term of the structure must still be trained on, outside the held-out equations
        if any(sum(1 for i, s in enumerate(sets) if x in s and i not in new) < min_train for x in c):
            continue
        held, chosen = new, chosen + [list(c)]
        if len(held) >= frac * E:
            break
    return chosen


def make(eqs, cfg):
    """eqs [Eq] -> {"structures", "train", "rehearsal", "val_ic": {gid: [sample idx]}, "train_windows": {...}}."""
    sets = [term_names(e.spec) for e in eqs]
    structs = cfg.get("rehearsal_structures") or choose_structures(
        sets, cfg["rehearsal_frac"], cfg["rehearsal_band"], cfg["seed"])
    reh = [i for i, s in enumerate(sets) if any(set(c) <= s for c in structs)]
    reh_set = set(reh)
    r = np.random.default_rng(cfg["seed"] + 1)
    out = {"structures": structs, "train": [], "rehearsal": [eqs[i].gid for i in reh], "val_ic": {}, "train_windows": {}}
    for i, e in enumerate(eqs):
        if i in reh_set:
            continue
        g = np.unique(e.groups)
        nv = int(round(cfg["val_ic_frac"] * len(g))) if len(g) >= cfg["val_ic_min_groups"] else 0
        vg = set(r.choice(g, nv, replace=False).tolist()) if nv else set()
        v = [int(w) for w, gg in zip(e.windows, e.groups) if gg in vg]
        out["train"].append(e.gid)
        out["val_ic"][e.gid] = v
        out["train_windows"][e.gid] = [int(w) for w, gg in zip(e.windows, e.groups) if gg not in vg]
    out["counts"] = {"equations": len(eqs), "train": len(out["train"]), "rehearsal": len(reh),
                     "train_windows": sum(map(len, out["train_windows"].values())),
                     "val_ic_windows": sum(map(len, out["val_ic"].values())),
                     "train_eqs_with_val_ic": sum(1 for v in out["val_ic"].values() if v),
                     "rehearsal_per_structure": [sum(1 for s in sets if set(c) <= s) for c in structs]}
    return out


if __name__ == "__main__":  # self-check on synthetic term sets: chosen structures cover ~frac, terms stay trained
    r = np.random.default_rng(0)
    pool = [f"P:t{i}" for i in range(30)]
    sets = [set(r.choice(pool, r.integers(2, 6), replace=False)) for _ in range(2000)]
    st = choose_structures(sets, 0.05, (0.004, 0.015), 0)
    held = {i for i, s in enumerate(sets) if any(set(c) <= s for c in st)}
    assert 0.05 * len(sets) <= len(held) <= 0.065 * len(sets), len(held)
    assert all(any(x in s for i, s in enumerate(sets) if i not in held) for c in st for x in c)
    print(st, len(held))
    print("splits.py ok")

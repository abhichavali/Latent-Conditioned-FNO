"""Building-block coverage of an equation set: how many equations contain each library term and with how many
different partner terms, whether the SWE / Aliev-Panfilov pieces appear alone and in every pair, and which terms
almost always come together (an encoder cannot separate their effects).

  python coverage.py --sample 4096                    # as the sampler draws them (before calibration filtering)
  python coverage.py --out /workspace/pdegen/out      # a generated dataset (also writes out/coverage.json)
"""

import argparse
import glob
import itertools
import json
from collections import Counter

import numpy as np

import sample
from template import LIB


def from_dataset(out):
    for f in sorted(glob.glob(f"{out}/eq_*/spec.json")):
        s = json.load(open(f))
        nf = s["template"]["nf"]
        C = np.zeros((nf, len(LIB[nf])))
        for i, k, c in s["template"]["coef"]:
            C[i, k] = c
        yield C, nf, len(s["fields"])


def from_sampler(n, seed=7):
    r = np.random.default_rng(seed)
    for _ in range(n):
        e = sample.equation(r)
        yield e["C"], e["nf"], e["n"]


def name(kind, item):
    return f"{kind}:{item if isinstance(item, str) else '|'.join(item)}"


def report(eqs):
    sets, swe, ap = [], Counter(), Counter()
    for C, nf, n in eqs:
        sets.append({name(*LIB[nf][k]) for k in range(len(LIB[nf])) if (C[:n, k] != 0).any()})
        ps = sample.pieces(C, nf, n)
        if n == 3:
            swe[frozenset(ps & set(sample.SWE_PIECES))] += 1
        if n >= 2:
            ap[frozenset(ps & set(sample.AP_PIECES))] += 1
    count = Counter(x for s in sets for x in s)
    partners = {k: set() for k in count}
    for s in sets:
        for x in s:
            partners[x] |= s - {x}
    pair = Counter(p for s in sets for p in itertools.combinations(sorted(s), 2))
    common = {k for k, c in count.items() if c > len(sets) / 2}  # e.g. diffusion: co-occurs with everything
    together = sorted(((pair[(a, b)] / min(count[a], count[b]), a, b) for a, b in pair
                       if not {a, b} & common and min(count[a], count[b]) >= 20
                       and pair[(a, b)] / min(count[a], count[b]) >= 0.9), reverse=True)

    def blocks(c, pieces):
        alone = {p: c[frozenset({p})] for p in pieces}
        pairs = {f"{p}+{q}": (c[frozenset({p, q})], sum(v for k, v in c.items() if {p, q} <= k))
                 for p, q in itertools.combinations(pieces, 2)}
        return {"none": c[frozenset()], "alone": alone, "pair exactly / any superset": pairs,
                "triples": {"+".join(sorted(k)): v for k, v in c.items() if len(k) == 3}}

    lib_all = {name(*x) for nf in LIB for x in LIB[nf]}
    counts = np.array(list(count.values()))
    return {"equations": len(sets), "terms used": f"{len(count)} / {len(lib_all)}",
            "never used": sorted(lib_all - set(count)),
            "per-term equations min / median": [int(counts.min()), int(np.median(counts))],
            "per-term partners min / median": [int(min(map(len, partners.values()))), int(np.median(list(map(len, partners.values()))))],
            "rarest terms (equations, partners)": {k: (count[k], len(partners[k])) for k in sorted(count, key=count.get)[:12]},
            "SWE blocks (3-field equations)": blocks(swe, sample.SWE_PIECES),
            "AP blocks (2+ field equations)": blocks(ap, sample.AP_PIECES),
            "nearly always together (>= 90%, excluding terms in > half the equations)": [f"{a} & {b}: {f:.0%}" for f, a, b in together[:20]]}


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--sample", type=int)
    ap_.add_argument("--out")
    a = ap_.parse_args()
    rep = report(from_dataset(a.out) if a.out else from_sampler(a.sample or 2000))
    print(json.dumps(rep, indent=1))
    if a.out:
        json.dump(rep, open(f"{a.out}/coverage.json", "w"), indent=1)

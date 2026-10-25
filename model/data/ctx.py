"""v3 data: context-feature cache and a per-equation group stream (train_ctx.py).

Cache: <ctx_cache>/<tag>/<eq>.npz per equation, every stored window: x (n, frames, nf, 32, 32) f16, st (n, ns, 32, 32)
f16, aux (n, AUX_DIM) = models/context.ctx_features of frames CTX_FRAMES.  ~6 GB for A u B, held in RAM by the stream.

  python data/ctx.py --config config/ctx.yaml        # build (resumable, skips finished equations)

Group sample (one equation): Q target windows from distinct IC groups, then k context windows from the other groups
(so a context window never shares an IC with a target).  Slots: K_max context slots (first k used), then the Q target
slots; cmask = context, ctmask = context + targets.  k = 0 with probability p0, else log-uniform on 1..K_max.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import adapter  # noqa: E402
from data.adapter import AUX_DIM  # noqa: E402
from data.canon import F_MAX, S_MAX  # noqa: E402
from models.context import CTX_FRAMES, CTX_N, ctx_features  # noqa: E402

NFR = len(CTX_FRAMES)


def _build_one(a):
    tag, d, out = a
    if os.path.exists(out):
        return 0
    e = adapter.Eq(tag, d)
    u = np.asarray(e.u()[:, list(CTX_FRAMES)], np.float32) * e.scale + e.shift
    f = [ctx_features(u[i], e.statics(i)) for i in range(u.shape[0])]
    x = np.stack([v["x"][:, : e.nf] for v in f])
    st = np.stack([v["st"][: e.ns] for v in f])
    aux = np.stack([v["aux"] for v in f])
    np.savez(out + ".tmp.npz", x=x, st=st, aux=aux)
    os.replace(out + ".tmp.npz", out)
    return 1


def build_cache(roots, root, workers=14):
    jobs = []
    for r in roots:
        os.makedirs(f"{root}/{r['tag']}", exist_ok=True)
        for d in sorted(os.listdir(r["path"])):
            if d.startswith("eq_") and os.path.exists(f"{r['path']}/{d}/u.npy"):
                jobs.append((r["tag"], f"{r['path']}/{d}", f"{root}/{r['tag']}/{d}.npz"))
    with Pool(workers) as p:
        n = sum(p.imap_unordered(_build_one, jobs, chunksize=4))
    print(f"ctx cache: {n} built, {len(jobs)} total", flush=True)


def load_cache(eqs, root, workers=16):
    def one(e):
        tag, name = e.gid.split("/")
        z = np.load(f"{root}/{tag}/{name}.npz")
        return e.gid, (z["x"], z["st"], z["aux"])

    with ThreadPoolExecutor(workers) as p:
        return dict(p.map(one, eqs))


def make_item(e, row, tw, cw, qs, cache, K):
    """One equation's group: target windows tw with query frames qs (Q, nq), context windows cw (<= K)."""
    X, ST, AUX = cache[e.gid]
    S = K + len(tw)
    cx = np.zeros((S, NFR, F_MAX, CTX_N, CTX_N), np.float16)
    cst = np.zeros((S, S_MAX, CTX_N, CTX_N), np.float16)
    caux = np.zeros((S, AUX_DIM), np.float32)
    for j, w in [(j, w) for j, w in enumerate(cw)] + [(K + j, w) for j, w in enumerate(tw)]:
        cx[j, :, : e.nf], cst[j, : e.ns], caux[j] = X[w], ST[w], AUX[w]
    cmask = np.arange(S) < len(cw)
    tg = [adapter.sample(e, row, int(w), q) for w, q in zip(tw, qs)]
    out = {k: np.stack([t[k] for t in tg]) for k in ("u0", "st", "pres", "aux", "y", "t")}
    out.update(logT=tg[0]["logT"], logL=tg[0]["logL"], eq=np.int32(row), k=np.int32(len(cw)), cx=cx, cst=cst, caux=caux,
               cpres=tg[0]["pres"], cmask=cmask, ctmask=cmask | (np.arange(S) >= K))
    return out


def _groups(e):
    return dict(zip(e.windows.tolist(), e.groups.tolist()))


def draw(r, e, tw_pool, cw_pool, gmap, Q, nq, K, k):
    """Targets from distinct groups of tw_pool, then k context windows of cw_pool outside the target groups (in random
    order; k = K gives an ordered list whose prefixes are nested contexts)."""
    tw_pool = np.asarray(tw_pool)
    g = np.array([gmap[int(w)] for w in tw_pool])
    ug = r.permutation(np.unique(g))[:Q]
    tw = [int(r.choice(tw_pool[g == x])) for x in ug]
    tw += [int(x) for x in r.choice(tw_pool, Q - len(tw))]  # fewer groups than Q: repeat
    cw_pool = np.asarray([w for w in cw_pool if gmap[int(w)] not in set(ug.tolist())])
    cw = [int(x) for x in r.permutation(cw_pool)[: min(k, len(cw_pool))]]
    qs = [np.sort(r.choice(np.arange(1, e.frames), nq, replace=False)) for _ in tw]
    return tw, cw, qs


class CtxStream:
    """Deterministic: batch(step) depends only on (seed, step).  eqs [(Eq, row)], windows: allowed (train) windows."""

    def __init__(self, eqs, windows, cache, E, Q, nq, K, p0, seed, weight_power=0.5, workers=12, prefetch=6):
        self.eqs, self.windows, self.cache = eqs, [np.asarray(w) for w in windows], cache
        self.E, self.Q, self.nq, self.K, self.p0, self.seed = E, Q, nq, K, p0, seed
        self.gmaps = [_groups(e) for e, _ in eqs]
        n = np.array([len(w) for w in windows], np.float64)
        p = np.where(n > 0, n**weight_power, 0)
        self.p = p / p.sum()
        self.pool, self.ahead, self.prefetch = ThreadPoolExecutor(workers), ThreadPoolExecutor(2), prefetch

    def batch(self, step):
        r = np.random.default_rng([self.seed, step])
        items = []
        for i in r.choice(len(self.eqs), self.E, p=self.p):
            e, row = self.eqs[i]
            k = 0 if r.random() < self.p0 else int(np.exp(r.uniform(0, np.log(self.K + 1))))
            tw, cw, qs = draw(r, e, self.windows[i], self.windows[i], self.gmaps[i], self.Q, self.nq, self.K, min(k, self.K))
            items.append((e, row, tw, cw, qs))
        out = list(self.pool.map(lambda a: make_item(*a, self.cache, self.K), items))
        return {k: np.stack([o[k] for o in out]) for k in out[0]}

    def iterate(self, start):
        from collections import deque

        futs = deque(self.ahead.submit(self.batch, s) for s in range(start, start + self.prefetch))
        s = start + self.prefetch
        while True:
            yield futs.popleft().result()
            futs.append(self.ahead.submit(self.batch, s))
            s += 1


def fixed_batches(eqs, tw_pools, cw_pools, cache, E, Q, nq, K, seed, n_batches):
    """Evaluation batches with K ordered context windows per equation (b["k"] = how many exist); evaluate k-shot by
    masking the first min(k, b["k"]) slots."""
    r = np.random.default_rng(seed)
    gm = [_groups(e) for e, _ in eqs]
    out = []
    for _ in range(n_batches):
        items = []
        for i in r.choice(len(eqs), E, replace=len(eqs) < E):
            e, row = eqs[i]
            items.append(make_item(e, row, *draw(r, e, tw_pools[i], cw_pools[i], gm[i], Q, nq, K, K), cache, K))
        out.append({k: np.stack([o[k] for o in items]) for k in items[0]})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    from train import load_config

    cfg = load_config(a.config)
    build_cache(cfg["data"]["roots"], cfg["ctx_cache"], a.workers)

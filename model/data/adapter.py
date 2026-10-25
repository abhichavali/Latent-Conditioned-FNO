"""Dataset adapter: equation table over one or more dataset roots, test-time-computable normalization, and a
background-prefetched batch stream of (equation, IC window, query frames) samples from memory-mapped u.npy.

Slots: field slot k = spec["fields"][k] (graph field node k), static slot k = spec["static"][k] (graph static node k),
padded to F_MAX = 3 / S_MAX = 2 with presence flags.

Normalization (per trajectory, per field, from the IC alone, so it needs no knowledge of the equation):
  m = spatial mean of the IC, s = spatial std of the IC floored at FLOOR; an all-constant IC (e.g. momentum at rest)
  falls back to s = FALLBACK.  Fields enter and leave the model as (u - m) / s, and the model also sees
  [tanh(m / s / 5), log10(s) / 3] per slot, so absolute levels are not lost.
  Statics: divided by their RMS (no mean subtraction), plus log10(rms) / 3 as a per-slot scalar.
"""

import glob
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np

from data.canon import F_MAX, S_MAX

FLOOR, FALLBACK, ZERO_TOL = 1e-3, 1.0, 1e-6
AUX_DIM = 2 * F_MAX + S_MAX  # per-slot normalization scalars
KEEP = ("id", "family", "fields", "static", "rhs", "L", "N", "dt", "substeps", "frames", "frame_dt", "norm",
        "samples", "template", "pieces", "perm", "tags")


class Eq:
    """One equation of a dataset root: trimmed spec, unique windows, lazily memory-mapped arrays."""

    def __init__(self, tag, d):
        s = json.load(open(f"{d}/spec.json"))
        self.spec = {k: s[k] for k in KEEP if k in s}
        self.gid, self.dir = f"{tag}/{s['id']}", d
        self.nf, self.ns = len(s["fields"]), len(s["static"])
        self.frames, self.L = s["frames"], float(s["L"])
        self.T = (s["frames"] - 1) * float(s["frame_dt"])
        first = {}  # duplicated windows (same pool IC and start frame) are kept once
        for i, x in enumerate(s["samples"]):
            first.setdefault((x["ic"], x["start_frame"]), i)
        self.windows = np.array(sorted(first.values()))
        self.groups = np.array([s["samples"][i]["ic"] for i in self.windows])  # windows of one IC stay together
        self.shift = np.array(s["norm"]["shift"], np.float32)[:, None, None]
        self.scale = np.array(s["norm"]["scale"], np.float32)[:, None, None]
        self._u = self._st = None
        self.bad = set()  # non-finite sample indices (drop_nonfinite)

    def u(self):
        if self._u is None:
            self._u = np.load(f"{self.dir}/u.npy", mmap_mode="r")
        return self._u

    def statics(self, w):  # (ns, N, N) raw float32
        if not self.ns:
            return None
        if self._st is None:
            self._st = np.load(f"{self.dir}/static.npy", mmap_mode="r")
        st = np.asarray(self._st[w], np.float32)
        return st[None] if st.ndim == 2 else st

    def frames_raw(self, w, idx):  # (len(idx), nf, N, N) de-normalized float32
        return np.asarray(self.u()[w, idx], np.float32) * self.scale + self.shift


def load_equations(roots):
    """roots [{"tag", "path"}] -> [Eq], ids globally unique as "tag/eq_XXXX".  Only complete eq_* dirs."""
    eqs = []
    for r in roots:
        for d in sorted(glob.glob(os.path.join(r["path"], "eq_*"))):
            if os.path.exists(f"{d}/spec.json") and os.path.exists(f"{d}/u.npy"):
                eqs.append(Eq(r["tag"], d))
    assert len({e.gid for e in eqs}) == len(eqs), "duplicate equation ids across roots: give each root its own tag"
    return eqs


def _nonfinite(d):
    u = np.load(f"{d}/u.npy", mmap_mode="r")
    return [i for i in range(u.shape[0]) if not np.isfinite(np.asarray(u[i], np.float32)).all()]


def drop_nonfinite(eqs, path, workers=32):
    """Remove windows containing non-finite values (some exist in A).  One full read per equation, cached by id."""
    cache = json.load(open(path)) if os.path.exists(path) else {}
    todo = [e for e in eqs if e.gid not in cache]
    if todo:
        with Pool(workers) as p:
            cache.update(zip([e.gid for e in todo], p.map(_nonfinite, [e.dir for e in todo], chunksize=4)))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump(cache, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
    for e in eqs:
        e.bad = set(cache[e.gid])
        keep = np.array([w not in e.bad for w in e.windows], bool)
        e.windows, e.groups = e.windows[keep], e.groups[keep]
    return {e.gid: sorted(e.bad) for e in eqs if e.bad}


def normalize(ic, frames=None, statics=None):
    """ic (nf, N, N) raw, frames (Q, nf, N, N) raw or None, statics (ns, N, N) raw or None ->
    padded model inputs.  Pure numpy; the same function serves training data and new target ICs."""
    nf, N = ic.shape[0], ic.shape[-1]
    m = ic.mean((-2, -1))
    sd = ic.std((-2, -1))
    s = np.where(sd < ZERO_TOL, FALLBACK, np.maximum(sd, FLOOR)).astype(np.float32)
    u0 = np.zeros((F_MAX, N, N), np.float32)
    u0[:nf] = (ic - m[:, None, None]) / s[:, None, None]
    st = np.zeros((S_MAX, N, N), np.float32)
    pres = np.zeros(F_MAX + S_MAX, np.float32)
    pres[:nf] = 1
    aux = np.zeros(AUX_DIM, np.float32)
    aux[:nf] = np.tanh(m / s / 5)
    aux[F_MAX : F_MAX + nf] = np.log10(s) / 3
    if statics is not None:
        ns = statics.shape[0]
        rms = np.maximum(np.sqrt((statics**2).mean((-2, -1))), FLOOR)
        st[:ns] = statics / rms[:, None, None]
        pres[F_MAX : F_MAX + ns] = 1
        aux[2 * F_MAX : 2 * F_MAX + ns] = np.log10(rms) / 3
    out = {"u0": u0, "st": st, "pres": pres, "aux": aux, "m": m, "s": s}
    if frames is not None:
        y = np.zeros((frames.shape[0], F_MAX, N, N), np.float32)
        y[:, :nf] = (frames - m[:, None, None]) / s[:, None, None]
        out["y"] = y
    return out


def denormalize(y, m, s):
    """Model output (..., F_MAX, N, N) -> raw fields (..., nf, N, N)."""
    nf = len(m)
    return y[..., :nf, :, :] * s[:, None, None] + m[:, None, None]


def scalars(T, L):
    return np.float32(np.log(T)), np.float32(np.log(L))


def sample(e, row, w, q):
    """One training sample: equation e (latent row `row`), sample index w, query frames q (Q,)."""
    raw = e.frames_raw(w, np.concatenate([[0], q]))
    n = normalize(raw[0], raw[1:], e.statics(w))
    lt, ll = scalars(e.T, e.L)
    return {"u0": n["u0"], "st": n["st"], "pres": n["pres"], "aux": n["aux"], "y": n["y"],
            "t": (q / (e.frames - 1)).astype(np.float32), "logT": lt, "logL": ll, "eq": np.int32(row)}


def collate(items):
    return {k: np.stack([it[k] for it in items]) for k in items[0]}


class Stream:
    """Deterministic batch stream: batch(step) depends only on (seed, step), so a resumed run sees the same data.

    eqs: list of (Eq, row) ; windows: per equation the allowed sample indices."""

    def __init__(self, eqs, windows, batch, q, seed, weight_power=0.5, workers=8, prefetch=8):
        self.eqs, self.windows, self.B, self.Q, self.seed = eqs, windows, batch, q, seed
        n = np.array([len(w) for w in windows], np.float64)
        p = np.where(n > 0, n**weight_power, 0)
        self.p = p / p.sum()
        self.pool = ThreadPoolExecutor(workers)  # per-sample loads
        self.ahead = ThreadPoolExecutor(2)  # whole batches; separate pool, or batch tasks starve their own samples
        self.prefetch = prefetch

    def batch(self, step):
        r = np.random.default_rng([self.seed, step])
        idx = r.choice(len(self.eqs), self.B, p=self.p)
        items = []
        for i in idx:
            e, row = self.eqs[i]
            w = int(r.choice(self.windows[i]))
            q = np.sort(r.choice(np.arange(1, e.frames), self.Q, replace=False))
            items.append((e, row, w, q))
        return collate(list(self.pool.map(lambda a: sample(*a), items)))

    def iterate(self, start):
        futs = deque(self.ahead.submit(self.batch, s) for s in range(start, start + self.prefetch))
        s = start + self.prefetch
        while True:
            yield futs.popleft().result()
            futs.append(self.ahead.submit(self.batch, s))
            s += 1


def fixed_batches(eqs, windows, batch, q, seed, n_batches):
    """A fixed evaluation set: n_batches batches of (equation, window, frames) drawn once with its own seed."""
    st = Stream(eqs, windows, batch, q, seed, weight_power=1.0, workers=8)
    out = [st.batch(i) for i in range(n_batches)]
    st.pool.shutdown()
    return out

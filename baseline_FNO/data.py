"""Prior-free raw data for the FNO baseline: random initial conditions for the two held-out demo equations
(shallow water, Aliev-Panfilov), solved with the generic spectral solver; 20 evenly spaced frames per rollout.

  python data.py --out /root/baseline_data          -> {swe,ap}_{train,test}.npy (n, 20, fields, 128, 128) f32,
                                                       swe_static.npy (2, 128, 128) [bathymetry, sponge]
"""

import argparse
import os
import sys
import time

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_collection"))
import demos  # noqa: E402
import pde  # noqa: E402

FRAMES = 20
# frame spacing ~4x the viz demos' (2 s, 1 t.u.): fronts move a few cells per frame
EQS = {
    "swe": {**demos.SWE, "frames": FRAMES, "substeps": 32},  # dt 0.25 s -> 8 s per frame, 152 s per rollout
    "ap": {**demos.AP, "frames": FRAMES, "substeps": 125},  # dt 0.02 -> 2.5 t.u. (~32 ms) per frame
}


def wrap(d):
    return (d + 0.5) % 1 - 0.5


def grid(n):
    x = np.arange(n) / n
    return np.meshgrid(x, x)


def swe_ic(r, n, static):
    """1-2 offshore thrust-fault sources (elongated uplift, often with a subsidence lobe: a leading-depression
    N-wave), fluid at rest, demo coast geometry (static = [bathymetry b, sponge s])."""
    X, Y = grid(n)
    eta = np.zeros_like(X)
    for _ in range(r.integers(1, 3)):
        cx, cy, th = r.uniform(0.08, 0.36), r.uniform(0.2, 0.8), r.uniform(-0.6, 0.6)
        w, l, A = r.uniform(0.03, 0.07), r.uniform(0.08, 0.22), r.uniform(0.5, 2)
        x, y = wrap(X - cx), wrap(Y - cy)
        a, c = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
        eta += A * np.exp(-((a / w) ** 2) - (c / l) ** 2)
        if r.random() < 0.6:
            off = r.choice([-1, 1]) * r.uniform(1.5, 2.2) * w
            eta -= A * r.uniform(0.3, 0.7) * np.exp(-(((a - off) / w) ** 2) - (c / l) ** 2)
    z = np.zeros_like(eta)
    return np.concatenate([np.stack([eta, z, z]), static])


def ap_ic(r, n):
    """1-2 broken wavefronts (excited band segment with a refractory wake behind it, random position, angle,
    size: the open ends curl into re-entrant spirals) plus 0-2 extra stimuli.  Fully random patches mostly
    annihilate on the torus and leave tissue at rest, which is a poor baseline."""
    X, Y = grid(n)
    st = lambda d: 0.5 * (1 + np.tanh(d / 0.012))
    u, v = np.zeros_like(X), np.zeros_like(X)
    for _ in range(r.integers(1, 3)):
        x, y, th = wrap(X - r.random()), wrap(Y - r.random()), r.uniform(0, 2 * np.pi)
        a, c = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
        w, half, wake = r.uniform(0.02, 0.04), r.uniform(0.12, 0.3), r.uniform(0.08, 0.2)
        along = st(half - np.abs(c))
        u = np.maximum(u, st(w - np.abs(a)) * along)
        v = np.maximum(v, r.uniform(1, 2) * st(a + w + wake) * st(-w - a) * along)
    for _ in range(r.integers(0, 3)):  # extra point stimuli (target waves)
        u = np.maximum(u, st(r.uniform(0.03, 0.08) - np.hypot(wrap(X - r.random()), wrap(Y - r.random()))))
    return np.stack([u, v * (1 - u)])


def simulate(spec, ics, batch=64):
    key, out = pde.solver_key(spec), []
    for i in range(0, len(ics), batch):
        tr = pde.rollout(key, jnp.asarray(ics[i : i + batch], jnp.float32), spec["dt"], spec["substeps"], spec["frames"])
        out.append(np.asarray(tr[:, :, : len(spec["fields"])]))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/baseline_data")
    ap.add_argument("--train", type=int, default=1024)
    ap.add_argument("--test", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    b, sp, b_vis, _ = demos.coast(EQS["swe"])
    static = np.stack([b, sp]).astype(np.float32)
    np.save(f"{a.out}/swe_static.npy", static)  # (2, N, N): bathymetry, sponge rate
    np.save(f"{a.out}/swe_bvis.npy", b_vis.astype(np.float32))  # visual bed (land) for the tsunami page
    for k, (name, spec) in enumerate(EQS.items()):
        for split, n, seed in (("train", a.train, 0), ("test", a.test, 1)):
            r = np.random.default_rng([seed, k])
            ics = np.stack([swe_ic(r, spec["N"], static) if name == "swe" else ap_ic(r, spec["N"]) for _ in range(n)])
            t = time.time()
            tr = simulate(spec, ics)
            ok = np.isfinite(tr).all((1, 2, 3, 4))
            np.save(f"{a.out}/{name}_{split}.npy", tr[ok].astype(np.float32))
            act = (np.abs(tr[ok][:, -1, 0]).max((-2, -1)) > (0.5 if name == "ap" else 0.05)).mean()
            print(f"{name} {split}: {ok.sum()}/{n} finite, still active at the last frame {act:.0%}, {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()

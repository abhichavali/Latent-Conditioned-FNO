"""The two targets (never seen in pretraining), in the pretraining conventions: pde.rollout, periodic, 128^2, 15 frames.

  swe  coastal tsunami of data_collection/demos.py: fields eta, u, v; statics b (bathymetry), s (sponge rate).
       frame_dt 10 s (dt 0.25 s x 40 substeps), T = 140 s, L = 4000 m.  Statics are the fixed demo coast.
       In mixed units (see SWE_MIX): x in km, t in ks, eta / b in m, u / v in km/ks = m/s, sponge s in 1/ks.
  ap   Aliev-Panfilov with a scar: diffusion dx(D dx u) + dy(D dy u), D = 1 except a smooth low-conductivity patch;
       otherwise demos.AP (k 8, a 0.15, eps0 0.01, mu1 0.2, mu2 0.3, 0.05 lap v).  frame_dt 2.5 (dt 0.02 x 125),
       T = 35, L = 64.  D is resampled per IC (its knobs are part of the parameter vector).

Physical knobs (fixed-length vectors; optional parts carry an on-flag and are ignored when off):
  sample(name, n, rng) -> params (n, P), ics (n, nf, N, N), statics (n, S, N, N);  KNOBS[name] = names;
  features(name, params) -> knob features in [0, 1] for k-means;  simulate(name, ics, statics) -> trajectories.

  python targets.py --out /workspace/targets       # fixed 100-IC test sets in the pretraining layout
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", "data_collection"))
import demos  # noqa: E402
import pde  # noqa: E402

from data import canon  # noqa: E402

# SWE in mixed units: horizontal length in km, time in ks, vertical length (eta, b) in m.  demos.SWE in SI has L = 4000
# (log L 8.3 vs pretraining's 1.6-5.3), and pure km would push the eta IC std (1.4e-4 km) under adapter.normalize's
# absolute floor 1e-3.  With x = KM x', t = KS t', u' = u KS / KM (km/ks = m/s, so numerically u' = u):
#   d_t' eta = -d_x'((eta - b) u') - ... + nu KS / KM^2 lap' eta - s KS eta            nu' = 8 * 1e3 / 1e6 = 8e-3
#   d_t' u'  = -u' d_x' u' - ... - g KS^2 / KM^2 d_x' eta + nu' lap' u' - s KS u'     g' = 9.81, s' = s * 1e3 (1/ks)
# so L = 4, dt = 2.5e-4 (frame_dt 0.01, T = 0.14: log T -2.0 inside pretraining's -4.0..4.8, log L 1.39), fields
# numerically identical to the SI solver; only the sponge static is scaled (x KS).  Viz: lx = L * KM m, dt = frame_dt * KS s.
KM, KS = 1000.0, 1000.0  # metres per horizontal length unit, seconds per time unit
SWE_MIX = {**demos.SWE, "L": demos.SWE["L"] / KM, "dt": demos.SWE["dt"] / KS, "rhs": {
    "eta": "-dx((eta - b)*u) - dy((eta - b)*v) + 0.008*lap(eta) - s*eta",
    "u": "-u*dx(u) - v*dy(u) - 9.81*dx(eta) + 0.008*lap(u) - s*u",
    "v": "-u*dx(v) - v*dy(v) - 9.81*dy(eta) + 0.008*lap(v) - s*v"}}
SPECS = {
    "swe": {**SWE_MIX, "frames": 15, "substeps": 40},
    "ap": {"fields": ["u", "v"], "static": ["D"], "L": 64.0, "N": 128, "order": 2, "dt": 0.02, "substeps": 125,
           "frames": 15, "rhs": {"u": "dx(D*dx(u)) + dy(D*dy(u)) + 8*u*(1 - u)*(u - 0.15) - u*v",
                                 "v": demos.AP["rhs"]["v"]}},
}
for _s in SPECS.values():
    _s["frame_dt"] = _s["dt"] * _s["substeps"]

# (name, lo, hi); "on" flags are in [0, 1] and thresholded at 0.5
_SRC = [("on", 0, 1), ("x", 0.08, 0.36), ("y", 0.2, 0.8), ("angle", -0.6, 0.6), ("amp", 0.5, 2.0), ("width", 0.03, 0.07),
        ("length", 0.08, 0.22), ("sub_on", 0, 1), ("sub_side", 0, 1), ("sub_off", 1.5, 2.2), ("sub_amp", 0.3, 0.7)]
_FRONT = [("on", 0, 1), ("x", 0, 1), ("y", 0, 1), ("angle", 0, 2 * np.pi), ("width", 0.02, 0.04), ("half_len", 0.12, 0.3),
          ("wake", 0.08, 0.2), ("v_amp", 1.0, 2.0)]
_STIM = [("on", 0, 1), ("x", 0, 1), ("y", 0, 1), ("radius", 0.03, 0.08)]
_SCAR = [("scar_x", 0, 1), ("scar_y", 0, 1), ("scar_radius", 0.08, 0.2), ("scar_D", 0.02, 0.05)]
RANGES = {
    "swe": [(f"src{j}_{n}", lo, hi) for j in range(2) for n, lo, hi in _SRC],
    "ap": [(f"front{j}_{n}", lo, hi) for j in range(2) for n, lo, hi in _FRONT]
    + [(f"stim{j}_{n}", lo, hi) for j in range(2) for n, lo, hi in _STIM] + _SCAR,
}
KNOBS = {k: [n for n, _, _ in v] for k, v in RANGES.items()}
# probabilities of the on-flags (first source / front always on; as in baseline_FNO/data.py)
ON_P = {"src0_on": 1.0, "src1_on": 0.5, "src0_sub_on": 0.6, "src1_sub_on": 0.6, "front0_on": 1.0, "front1_on": 0.5,
        "stim0_on": 2 / 3, "stim1_on": 1 / 3}


def spec(name):
    return dict(SPECS[name])


def canonical_graph(name):
    return canon.graph(SPECS[name])


def sample_params(name, n, rng):
    P = np.empty((n, len(RANGES[name])), np.float32)
    for k, (kn, lo, hi) in enumerate(RANGES[name]):
        P[:, k] = (rng.random(n) < ON_P[kn]).astype(np.float32) if kn in ON_P else rng.uniform(lo, hi, n)
        if kn.endswith("sub_side"):
            P[:, k] = rng.integers(0, 2, n)
    return P


def features(name, params):
    """Knob features in [0, 1] for clustering; knobs of an absent part are zeroed with its flag."""
    lo = np.array([r[1] for r in RANGES[name]], np.float32)
    hi = np.array([r[2] for r in RANGES[name]], np.float32)
    f = (params - lo) / (hi - lo)
    names = KNOBS[name]
    for k, kn in enumerate(names):
        part = kn.split("_")[0]
        if f"{part}_on" in names and kn != f"{part}_on":
            f[:, k] *= params[:, names.index(f"{part}_on")] > 0.5
        if "_sub_" in kn and not kn.endswith("sub_on"):
            f[:, k] *= params[:, names.index(f"{part}_sub_on")] > 0.5
    return f


def _grid(N):
    x = np.arange(N) / N
    return np.meshgrid(x, x)  # X varies along the last axis


_wrap = lambda d: (d + 0.5) % 1 - 0.5
_step = lambda d: 0.5 * (1 + np.tanh(d / 0.012))


def _swe_ic(p, X, Y):
    k = {n: v for n, v in zip(KNOBS["swe"], p)}
    eta = np.zeros_like(X)
    for j in range(2):
        g = lambda n: k[f"src{j}_{n}"]
        if g("on") < 0.5:
            continue
        x, y, th = _wrap(X - g("x")), _wrap(Y - g("y")), g("angle")
        a, c = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
        w, ln, A = g("width"), g("length"), g("amp")
        eta += A * np.exp(-((a / w) ** 2) - (c / ln) ** 2)
        if g("sub_on") > 0.5:  # subsidence lobe on one side: a leading-depression N-wave
            off = (1 if g("sub_side") > 0.5 else -1) * g("sub_off") * w
            eta -= A * g("sub_amp") * np.exp(-(((a - off) / w) ** 2) - (c / ln) ** 2)
    z = np.zeros_like(eta)
    return np.stack([eta, z, z])


def _ap_ic(p, X, Y):
    k = {n: v for n, v in zip(KNOBS["ap"], p)}
    u, v = np.zeros_like(X), np.zeros_like(X)
    for j in range(2):  # broken wavefronts: excited band with a refractory wake; open ends curl into spirals
        g = lambda n: k[f"front{j}_{n}"]
        if g("on") < 0.5:
            continue
        x, y, th = _wrap(X - g("x")), _wrap(Y - g("y")), g("angle")
        a, c = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
        w, along = g("width"), _step(g("half_len") - np.abs(c))
        u = np.maximum(u, _step(w - np.abs(a)) * along)
        v = np.maximum(v, g("v_amp") * _step(a + w + g("wake")) * _step(-w - a) * along)
    for j in range(2):  # extra point stimuli (target waves)
        g = lambda n: k[f"stim{j}_{n}"]
        if g("on") > 0.5:
            u = np.maximum(u, _step(g("radius") - np.hypot(_wrap(X - g("x")), _wrap(Y - g("y")))))
    r = np.hypot(_wrap(X - k["scar_x"]), _wrap(Y - k["scar_y"]))
    D = 1 - (1 - k["scar_D"]) * 0.5 * (1 + np.tanh((k["scar_radius"] - r) / 0.025))  # smooth low-conductivity patch
    return np.stack([u, v * (1 - u)]), D[None]


_COAST = {}


def make_ics(name, params):
    """params (n, P) -> ics (n, nf, N, N), statics (n, S, N, N), float32 raw."""
    N = SPECS[name]["N"]
    X, Y = _grid(N)
    if name == "swe":
        if "bs" not in _COAST:
            b, s, _, _ = demos.coast(SPECS["swe"])  # unit-square geometry, b in metres, s in 1/s
            _COAST["bs"] = np.stack([b, s * KS]).astype(np.float32)
        ics = np.stack([_swe_ic(p, X, Y) for p in params]).astype(np.float32)
        return ics, np.broadcast_to(_COAST["bs"], (len(params), *_COAST["bs"].shape)).copy()
    out = [_ap_ic(p, X, Y) for p in params]
    return np.stack([o[0] for o in out]).astype(np.float32), np.stack([o[1] for o in out]).astype(np.float32)


def sample(name, n, rng):
    P = sample_params(name, n, rng)
    return (P, *make_ics(name, P))


def simulate(name, ics, statics, batch=25):
    """Solver trajectories (n, frames, nf, N, N) float32, frame 0 = IC."""
    import jax.numpy as jnp

    s = SPECS[name]
    key, nf, out = pde.solver_key(s), len(s["fields"]), []
    for i in range(0, len(ics), batch):
        u0 = jnp.asarray(np.concatenate([ics[i : i + batch], statics[i : i + batch]], 1))
        out.append(np.asarray(pde.rollout(key, u0, s["dt"], s["substeps"], s["frames"])[:, :, :nf]))
    return np.concatenate(out)


def check_swe_mix(n=3):
    """The mixed-unit spec reproduces the SI spec (demos.SWE) on n ICs; returns the max relative deviation of the fields
    (identical numerically: eta in m, u = m/s = km/ks)."""
    import jax.numpy as jnp

    _, ic, st = sample("swe", n, np.random.default_rng(1))
    m = {**demos.SWE, "frames": 15, "substeps": 40}
    u0 = np.concatenate([ic, st[:, :1], st[:, 1:] / KS], 1)  # sponge 1/ks -> 1/s
    ref = np.asarray(pde.rollout(pde.solver_key(m), jnp.asarray(u0), m["dt"], m["substeps"], m["frames"]))[:, :, :3]
    mix = simulate("swe", ic, st)
    return float(np.abs(mix - ref).max() / np.abs(ref).max())


def write_test(name, out, n=100, seed=2026):
    """Fixed test set in the pretraining layout: spec.json, u.npy f16 (n, frames, nf, N, N) = (u - shift) / scale,
    static.npy f16 (n, S, N, N) raw, params.npy (n, P).  Non-finite rollouts are redrawn."""
    r = np.random.default_rng(seed)
    P, U, S = [], [], []
    t0 = time.time()
    while sum(map(len, P)) < n:
        p, ic, st = sample(name, n, r)
        tr = simulate(name, ic, st)
        ok = np.isfinite(tr).all((1, 2, 3, 4))
        P.append(p[ok]), U.append(tr[ok]), S.append(st[ok])
    P, U, S = (np.concatenate(x)[:n] for x in (P, U, S))
    s = spec(name)
    shift, scale = U.mean((0, 1, 3, 4)), U.std((0, 1, 3, 4)) + 1e-8
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/u.npy", ((U - shift[:, None, None]) / scale[:, None, None]).astype(np.float16))
    np.save(f"{out}/static.npy", S.astype(np.float16))
    np.save(f"{out}/params.npy", P)
    T = (s["frames"] - 1) * s["frame_dt"]
    act = float((np.abs(U[:, -1, 0]).max((-2, -1)) > (0.5 if name == "ap" else 0.05)).mean())  # 5 cm wave
    g = canonical_graph(name)
    meta = {**s, "id": name.upper(), "family": f"target/{name}", "T": T, "latex": pde.latex(s), "graph": pde.graph(s),
            "canonical": {"text": g["text"], "terms": g["terms"]}, "norm": {"shift": shift.tolist(), "scale": scale.tolist()},
            "samples": [{"ic": i, "start_frame": 0, "ic_family": name} for i in range(n)], "knobs": KNOBS[name],
            "knob_ranges": RANGES[name], "seed": seed, "active_at_last_frame": act,
            "range": {f: [float(U[:, :, i].min()), float(U[:, :, i].max())] for i, f in enumerate(s["fields"])},
            "layout": "u.npy float16 (ic, frame, field, y, x) = (u - shift) / scale; static.npy (ic, static, y, x) raw"}
    json.dump(meta, open(f"{out}/spec.json", "w"))
    print(f"{name}: {n} ICs, T {T}, L {s['L']}, active at the last frame {act:.0%}, ranges {meta['range']}, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/targets")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--only", nargs="+", default=list(SPECS))
    a = ap.parse_args()
    if "swe" in a.only:
        dev = check_swe_mix()
        print(f"swe mixed units vs SI solver: max deviation {dev:.2e} of max |field|", flush=True)
        assert dev < 1e-4, dev
    for nm in a.only:
        p, ic, st = sample(nm, 4, np.random.default_rng(0))  # self-check: shapes and knob features
        assert ic.shape == (4, len(SPECS[nm]["fields"]), 128, 128) and st.shape[:2] == (4, len(SPECS[nm]["static"]))
        f = features(nm, p)
        assert f.shape == p.shape and (f >= -1e-6).all() and (f <= 1 + 1e-6).all()
        write_test(nm, f"{a.out}/{nm}", a.n)

"""The two held-out equations (shallow water, Aliev-Panfilov) in the same spec format, simulated by the generic solver.

  python demos.py swe  [--out demos/swe]     -> h.npy (T, Ny, Nx) depth, b.npy (Ny, Nx) bed, meta.json
  python demos.py ap   [--out demos/ap]      -> u.npy (T, Ny, Nx) potential, meta.json

Each also writes spec.json (with its graph) so the model sees exactly what it would see for a training equation.
"""

import argparse
import json
import math
import os

import jax.numpy as jnp
import numpy as np

import pde

# Coastal tsunami: shallow water over bathymetry b (bed elevation, < 0 under water), surface elevation eta, depth
# eta - b.  Spectral + periodic means no dry land, so land and the open-ocean edges are a sponge: a static damping
# rate s (1/s) absorbs waves there.  The viewer gets a separate visual bed (beach + hills) on the sponge-land band.
SWE = {
    "fields": ["eta", "u", "v"], "static": ["b", "s"], "L": 4000.0, "N": 128, "order": 4,
    "dt": 0.25, "substeps": 8, "frames": 240,
    "rhs": {
        "eta": "-dx((eta - b)*u) - dy((eta - b)*v) + 8*lap(eta) - s*eta",
        "u": "-u*dx(u) - v*dy(u) - 9.81*dx(eta) + 8*lap(u) - s*u",
        "v": "-u*dx(v) - v*dy(v) - 9.81*dy(eta) + 8*lap(v) - s*v",
    },
}


def coast(spec):
    """Unit-square geometry: sim bed b, sponge rate s, visual bed b_vis, land mask (all periodic and smooth)."""
    X, Y = (np.asarray(a) / spec["L"] for a in pde.grid(spec))
    T = lambda d, w: 0.5 * (1 + np.tanh(d / w))
    xc = 0.62 + 0.035 * np.sin(2 * np.pi * 2 * Y + 0.6) + 0.02 * np.sin(2 * np.pi * 5 * Y + 2)  # wiggly coastline
    sea = 12 + 46 * T(0.44 - X, 0.035)  # 58 m basin -> continental slope -> 12 m shelf
    depth = 4 + (sea - 4) * T(xc - X - 0.06, 0.04)  # shoaling to ~4 m at the coast
    land = T(X - xc, 0.008) * T(0.93 - X, 0.01)  # land band, back to deep water across the periodic seam
    depth = depth * (1 - land) + 3 * land
    r2 = (X - 0.3) ** 2 + (Y - 0.68) ** 2
    island = np.exp(-r2 / (2 * 0.03**2))
    depth = depth * (1 - 0.93 * island)  # island shoal (~4 m): waves refract around it
    edge = 1 - (1 - T(X - xc - 0.03, 0.02) * T(0.99 - X, 0.005)) * (1 - T(0.035 - X, 0.01)) * (1 - T(0.03 - Y, 0.01)) * (1 - T(Y - 0.97, 0.01))
    edge = np.maximum(edge, T(X - xc - 0.03, 0.02) * T(0.99 - X, 0.005))
    s = 0.05 * (1 - (1 - edge) * (1 - T(0.03 - X, 0.008)))
    d = X - xc  # inland distance: a low coastal plain (floods visibly) rising into hills
    hills = 0.1 + 13 * np.maximum(d, 0) + 70 * np.maximum(1 - np.exp(-(d - 0.06) / 0.08), 0) * (0.65 + 0.35 * np.sin(2 * np.pi * 3 * Y) * np.cos(2 * np.pi * 7 * X))
    landv = (X > xc) & (X < 0.93)
    b_vis = np.where(landv, hills, -depth)
    b_vis = np.where(island > 0.45, 30 * (island - 0.45) / 0.55 + 0.5, b_vis)
    return -depth, s, b_vis, landv | (island > 0.45)


def swe_ic(spec):
    b, s, b_vis, _ = coast(spec)
    X, Y = (np.asarray(a) / spec["L"] for a in pde.grid(spec))
    # thrust-fault source offshore: elongated uplift with a subsidence lobe landward of it (leading depression)
    rot = lambda x, y, t: (x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t))
    a, c = rot(X - 0.2, Y - 0.42, 0.25)
    eta = 2.0 * np.exp(-(a / 0.05) ** 2 - (c / 0.3) ** 2) - 1.0 * np.exp(-((a - 0.09) / 0.05) ** 2 - (c / 0.3) ** 2)
    z = np.zeros_like(eta)
    return np.stack([eta, z, z, b, s]), b_vis


# Aliev-Panfilov (dimensionless; 1 time unit ~ 12.9 ms), standard k, a, mu1, mu2 with two changes for 128^2:
# eps0 0.002 -> 0.01 (faster recovery, so a re-entrant wavelength fits the box instead of the wave dying on its own
# refractory tail) and a small 0.05 lap(v) (v has no diffusion, so it rings negative spectrally; a negative gate blows up).
AP = {
    "fields": ["u", "v"], "static": [], "L": 64.0, "N": 128, "order": 2,
    "dt": 0.02, "substeps": 50, "frames": 360,
    "rhs": {
        "u": "1.0*lap(u) + 8*u*(1 - u)*(u - 0.15) - u*v",
        "v": "0.05*lap(v) + (0.01 + 0.2*v/(u + 0.3))*(-v - 8*u*(u - 0.15 - 1))",
    },
}


def ap_ic(spec):
    X, Y = (np.asarray(a) / spec["L"] for a in pde.grid(spec))
    step = lambda d: 0.5 * (1 + np.tanh(d / 0.012))  # smooth edges: hard steps ring spectrally and hit the u + 0.3 pole
    lower = step(0.5 - Y)
    u = lower * step(X - 0.30) * step(0.36 - X)  # excited band, lower half only
    v = 1.5 * lower * step(X - 0.15) * step(0.30 - X)  # refractory behind it: the broken end curls into a spiral
    return np.stack([u, v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["swe", "ap"])
    ap.add_argument("--out")
    a = ap.parse_args()
    spec = SWE if a.which == "swe" else AP
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos", a.which)
    os.makedirs(out, exist_ok=True)
    if a.which == "swe":
        u0, b = swe_ic(spec)  # b = visual bed for the viewer
    else:
        u0 = ap_ic(spec)
    traj = np.asarray(pde.simulate(spec, jnp.asarray(u0, jnp.float32)[None]))[0]  # (T, C, N, N)
    assert np.isfinite(traj).all(), "blew up"
    frame_dt = spec["dt"] * spec["substeps"]
    if a.which == "swe":
        np.save(f"{out}/h.npy", np.maximum(traj[:, 0] - b, 0).astype(np.float32))  # dry on (visual) land
        np.save(f"{out}/b.npy", b.astype(np.float32))
        meta = {"dt": frame_dt, "time_unit": "s", "lx": spec["L"], "ly": spec["L"], "title": "Coastal tsunami (solver)", "vexag": 4, "wexag": 6}
    else:
        np.save(f"{out}/u.npy", traj[:, 0].astype(np.float32))
        meta = {"dt": frame_dt * 12.9, "time_unit": "ms", "geometry": "sheet", "range": [0, 1], "title": "Aliev-Panfilov (solver)"}
    json.dump(meta, open(f"{out}/meta.json", "w"))
    json.dump({**spec, "latex": pde.latex(spec), "graph": pde.graph(spec)}, open(f"{out}/spec.json", "w"))
    print(out, traj.shape, "range", float(traj[:, 0].min()), float(traj[:, 0].max()))


if __name__ == "__main__":
    main()

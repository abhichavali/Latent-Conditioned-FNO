"""The two held-out demos in the same spec format, simulated by the generic solver and written for the viz pages.

  python demos.py swe  [--out demos/swe]     -> h.npy (T, Ny, Nx) depth, b.npy (Ny, Nx) bed, meta.json   (viz/tsunami)
  python demos.py ap   [--out demos/ap]      -> u.npy (T, Ny, Nx) potential, meta.json                    (viz/heart)
  open http://localhost:8000/viz/tsunami/?data=../../data_collection/demos/swe/

Each also writes spec.json (with its graph) so the model sees exactly what it would see for a training equation.
"""

import argparse
import json
import math
import os

import jax.numpy as jnp
import numpy as np

import pde

# Shallow water over bathymetry b (bed elevation, < 0 under water), surface elevation eta, depth h = eta - b.
# Periodic and fully wet (spectral), so the "coast" is a shallow shelf rather than dry land.
SWE = {
    "fields": ["eta", "u", "v"], "static": ["b"], "L": 4000.0, "N": 128, "order": 4,
    "dt": 0.25, "substeps": 8, "frames": 240,
    "rhs": {
        "eta": "-dx((eta - b)*u) - dy((eta - b)*v) + 20*lap(eta)",
        "u": "-u*dx(u) - v*dy(u) - 9.81*dx(eta) + 20*lap(u)",
        "v": "-u*dx(v) - v*dy(v) - 9.81*dy(eta) + 20*lap(v)",
    },
}

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


def swe_ic(spec):
    X, Y = (np.asarray(a) for a in pde.grid(spec))
    L = spec["L"]
    xs = X / L
    # deep ocean (58 m) shoaling onto a 6 m shelf band, with an island; periodic in x and y
    shelf = 0.5 * (np.tanh((xs - 0.62) / 0.04) - np.tanh((xs - 0.95) / 0.04))
    b = -58 + 52 * shelf + 40 * np.exp(-((xs - 0.35) ** 2 + (Y / L - 0.7) ** 2) / 0.003)
    eta = 2.0 * np.exp(-((xs - 0.15) ** 2 + (Y / L - 0.35) ** 2) / 0.004)  # 2 m hump offshore
    z = np.zeros_like(eta)
    return np.stack([eta, z, z, b]), b


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
        u0, b = swe_ic(spec)
    else:
        u0 = ap_ic(spec)
    traj = np.asarray(pde.simulate(spec, jnp.asarray(u0, jnp.float32)[None]))[0]  # (T, C, N, N)
    assert np.isfinite(traj).all(), "blew up"
    frame_dt = spec["dt"] * spec["substeps"]
    if a.which == "swe":
        np.save(f"{out}/h.npy", np.maximum(traj[:, 0] - b, 0).astype(np.float32))
        np.save(f"{out}/b.npy", b.astype(np.float32))
        meta = {"dt": frame_dt, "time_unit": "s", "lx": spec["L"], "ly": spec["L"], "title": "Shallow water (solver)", "vexag": 4, "wexag": 6}
    else:
        np.save(f"{out}/u.npy", traj[:, 0].astype(np.float32))
        meta = {"dt": frame_dt * 12.9, "time_unit": "ms", "geometry": "sheet", "range": [0, 1], "title": "Aliev-Panfilov (solver)"}
    json.dump(meta, open(f"{out}/meta.json", "w"))
    json.dump({**spec, "latex": pde.latex(spec), "graph": pde.graph(spec)}, open(f"{out}/spec.json", "w"))
    print(out, traj.shape, "range", float(traj[:, 0].min()), float(traj[:, 0].max()))


if __name__ == "__main__":
    main()

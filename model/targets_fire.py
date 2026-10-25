"""Wildfire-spread target ("fire"), registered in targets.py (same API and test-set layout as swe / ap).

  u  fire intensity (heat) in [0, ~1];  f  remaining fuel in [0, 1].  No statics: all heterogeneity is in the IC of f.
    du/dt = lap u + 8 u (1 - u)(u - 0.1) - 2 u (1 - f) - 0.35 dx u - 0.2 dy u
    df/dt = -0.4 u f
  A bistable (pushed) front, as AP's cubic: u = 0 is stable (rate -2.8 without fuel loss), so solver noise never
  ignites.  Missing fuel (1 - f) damps the excitation, like AP's recovery variable but never recovering, so
    - a burning front needs f >~ 0.45; sparser fuel slows it, then stops it (plane fronts, cells per frame downwind /
      upwind: f 0.5 1.4 / 0.1, f 0.8 3.0 / 1.3, f 1.0 3.8 / 2.1 at b = 0.3), and the fire goes out behind it once
      f ~ 0.15-0.2 (burn scar);  (plane-front numbers at b = 0.3, frame_dt 1; the target uses b = 0.4, frame_dt 1.5)
    - in a fuel-free strip u decays at rate 2.8 over sqrt(1 / 2.8) ~ 1.2 cells: breaks of 2 cells are jumped
      downwind, 3+ cells stop it;
    - the constant wind (0.4 toward +x, 30 deg) advects heat: faster downwind, slow backing fire upwind.
  Every term is in the pretraining library (cubic monomials in u, u f, linear operators).  L = 64 (dx 0.5, front
  10-90% rise ~4-5 cells), dt 0.015 x 100 = frame_dt 1.5, T = 21 (log L 4.16, log T 3.04).  On 64 sampled ICs:
  86% spread, 81% still burning at the last frame, median burned area 28% of the box, burned-area radius +2.3
  cells per frame (downwind fronts ~4-5).

Knobs: 2 ignitions (on, x, y, radius; the first always on, x upwind so the downwind run fits the box), the fuel map
(mean, contrast = std, correlation length of a smooth random field; its noise seed is a hash of the knob vector, so
make_ics(params) is a pure function), 3 firebreaks (on, x, y, angle, half length, width; fuel-free strips) and a lake.
"""

import zlib

import numpy as np

N = 128
SPEC = {"fields": ["u", "f"], "static": [], "L": 64.0, "N": N, "order": 2, "dt": 0.015, "substeps": 100, "frames": 15,
        "rhs": {"u": "lap(u) + 8*u*(1 - u)*(u - 0.1) - 2*u*(1 - f) - 0.35*dx(u) - 0.2*dy(u)", "f": "-0.4*u*f"}}
SPEC["frame_dt"] = SPEC["dt"] * SPEC["substeps"]

_IGN = [("on", 0, 1), ("x", 0.15, 0.55), ("y", 0.2, 0.8), ("radius", 0.035, 0.06)]
_BRK = [("on", 0, 1), ("x", 0, 1), ("y", 0, 1), ("angle", 0, np.pi), ("half_len", 0.1, 0.45), ("width", 0.015, 0.04)]
RANGES = ([(f"ign{j}_{n}", lo, hi) for j in range(2) for n, lo, hi in _IGN]
          + [("fuel_mean", 0.65, 1.0), ("fuel_contrast", 0.05, 0.25), ("fuel_corr", 0.04, 0.15)]
          + [(f"brk{j}_{n}", lo, hi) for j in range(3) for n, lo, hi in _BRK]
          + [("lake_on", 0, 1), ("lake_x", 0, 1), ("lake_y", 0, 1), ("lake_radius", 0.05, 0.12)])
KNOBS = [n for n, _, _ in RANGES]
ON_P = {"ign0_on": 1.0, "ign1_on": 0.4, "brk0_on": 0.8, "brk1_on": 0.5, "brk2_on": 0.3, "lake_on": 0.4}

_x = np.arange(N) / N
X, Y = np.meshgrid(_x, _x)  # X varies along the last axis
_wrap = lambda d: (d + 0.5) % 1 - 0.5
_step = lambda d, w=0.004: 0.5 * (1 + np.tanh(d / w))


def _fuel(p, k):
    """Smooth random vegetation density: Gaussian random field (correlation length fuel_corr), mean / std knobs."""
    r = np.random.default_rng(zlib.crc32(np.asarray(p, np.float32).tobytes()))
    q = np.fft.fftfreq(N) * N
    g = np.fft.ifft2(np.fft.fft2(r.standard_normal((N, N))) * np.exp(-0.5 * (2 * np.pi * k["fuel_corr"]) ** 2 * (q[:, None] ** 2 + q ** 2))).real
    return np.clip(k["fuel_mean"] + k["fuel_contrast"] * g / g.std(), 0, 1)


def make_ic(p):
    k = dict(zip(KNOBS, p))
    f, u = _fuel(p, k), np.zeros((N, N))
    for j in range(3):  # firebreaks: fuel-free strips (roads, rivers) of a given width around a segment
        g = lambda n: k[f"brk{j}_{n}"]
        if g("on") > 0.5:
            x, y, th = _wrap(X - g("x")), _wrap(Y - g("y")), g("angle")
            a, c = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
            d = np.hypot(np.maximum(np.abs(a) - g("half_len"), 0), c)
            f *= 1 - _step(g("width") / 2 - d)
    if k["lake_on"] > 0.5:
        f *= 1 - _step(k["lake_radius"] - np.hypot(_wrap(X - k["lake_x"]), _wrap(Y - k["lake_y"])))
    for j in range(2):
        g = lambda n: k[f"ign{j}_{n}"]
        if g("on") > 0.5:
            u = np.maximum(u, _step(g("radius") - np.hypot(_wrap(X - g("x")), _wrap(Y - g("y"))), 0.008))
    return np.stack([u, f])


def make_ics(params):
    """params (n, P) -> ics (n, 2, N, N), statics (n, 0, N, N)."""
    return np.stack([make_ic(p) for p in params]).astype(np.float32), np.zeros((len(params), 0, N, N), np.float32)


def stats(U):
    """Trajectories (n, frames, 2, N, N) -> burned fraction of the box at the last frame, still burning at the last frame
    (max u > 0.3), spread (burned area > 4x the ignition core), radius growth of the burned area in cells per frame."""
    f0 = U[:, :1, 1]
    burned = (U[:, :, 1] < 0.5 * f0) & (f0 > 0.2)
    area = burned.mean((-2, -1))
    R = np.sqrt(area * N * N / np.pi)
    ign = (U[:, 0, 0] > 0.5).mean((-2, -1))
    return {"burned_frac_median": float(np.median(area[:, -1])), "burning_at_last_frame": float((U[:, -1, 0].max((-2, -1)) > 0.3).mean()),
            "spread": float((area[:, -1] > 4 * ign).mean()), "radius_growth_cells_per_frame_median": float(np.median((R[:, -1] - R[:, 4]) / (U.shape[1] - 5))),
            "u_range": [float(U[:, :, 0].min()), float(U[:, :, 0].max())], "f_range": [float(U[:, :, 1].min()), float(U[:, :, 1].max())]}

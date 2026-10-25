"""Scenario banks for the deck's live demo (presentation/README.md "Scenario bank") and measured timings.

  python demo_bank.py tsunami [--models DIR] [--out OUT]   # OUT/tsunami: epicentre grid (Simula + solver), risk map
  python demo_bank.py heart   [--models DIR] [--out OUT]   # OUT/heart: scars x ablation lines, verdicts vs the solver
  python demo_bank.py timing  [--models DIR] [--out OUT]   # OUT/timing.json (run on an idle GPU)
  python demo_bank.py check                                # CPU self-check of the geometry helpers

Simula = active_v2.load_adapted(<models>/<target>): the adapted per-target model, queried directly at fine times
t in [0, 1] of its 15-frame window (time-conditioned, no interpolation); later windows are chained (the predicted last
state is the next window's IC).  Solver = pde.rollout, the GPU pseudo-spectral solver the targets were made with.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import targets  # noqa: E402
import demos  # noqa: E402  (data_collection, on the path via targets)

MODELS = "/workspace/results/pretrain_AB_s200000/final_models"  # adapted models <MODELS>/{swe,ap}/ (active_v2 export)
AP_MS = 12.9  # ms per AP time unit


def load(models, target):
    import active_v2

    return active_v2.load_adapted(f"{models}/{target}")


def chain(pred, ics, st, windows, q):
    """Simula rollout: (n, windows * (q - 1) + 1, nf, N, N), frame 0 = the IC, q times per window."""
    t, out, cur = np.linspace(0, 1, q)[1:], [ics[:, None]], ics
    for _ in range(windows):
        out.append(pred(cur, st, t))
        cur = out[-1][:, -1]
    return np.concatenate(out, 1)


def solve(name, ics, st, windows=1, q=None, batch=25):
    """Solver rollout at the same frame times as chain (q must split the window into whole substeps)."""
    import jax.numpy as jnp
    import pde

    s = targets.SPECS[name]
    F = s["frames"] - 1
    q = q or F + 1
    sub, r = divmod(s["substeps"] * F, q - 1)
    assert r == 0, (name, q)
    key, nf, out = pde.solver_key(s), len(s["fields"]), []
    for i in range(0, len(ics), batch):
        u0 = jnp.asarray(np.concatenate([ics[i : i + batch], st[i : i + batch]], 1))
        out.append(np.asarray(pde.rollout(key, u0, s["dt"], sub, windows * (q - 1) + 1)[:, :, :nf]))
    return np.concatenate(out)


def frac(x, N=128):
    """Unit-square coordinate (X = i / N) -> the deck's grid fraction i / (N - 1)."""
    return round(float(x) * N / (N - 1), 4)


def params(name, n, **knobs):
    """n knob vectors at their range midpoints (on-flags off), then knobs set by name (scalars or (n,) arrays)."""
    P = np.array([[0.0 if k.endswith("_on") else (lo + hi) / 2 for k, lo, hi in targets.RANGES[name]]] * n, np.float32)
    for k, v in knobs.items():
        P[:, targets.KNOBS[name].index(k)] = v
    return P


def relL2(p, y):
    return float(np.linalg.norm((p - y).ravel()) / np.linalg.norm(y.ravel()))


def save(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, x)


def upsample(x, k):
    """(n, T, [C,] N, N) -> (n, (T - 1) k + 1, [C,] N, N): motion-compensated in-between frames (baseline_FNO/interp.py;
    channel 0 drives the flow)."""
    if k == 1:
        return x
    sys.path.append(os.path.join(HERE, "..", "baseline_FNO"))
    import interp

    return np.stack([interp.upsample(f.reshape(len(f), -1, *f.shape[-2:]), k).reshape(-1, *f.shape[1:]) for f in x])


# ---------------------------------------------------------------- tsunami: epicentre grid + risk map
SWE_FIXED = dict(src0_on=1, src0_angle=0.0, src0_amp=1.6, src0_width=0.05, src0_length=0.18, src0_sub_on=1,
                 src0_sub_side=1, src0_sub_off=1.85, src0_sub_amp=0.5)  # dramatic but typical; depression landward


def tsunami(a):
    out, s = f"{a.out}/tsunami", targets.SPECS["swe"]
    pred = load(a.models, "swe")
    _, _, bvis, _ = demos.coast(s)
    land, q, k = bvis > 0, a.q_swe, a.interp
    qo = (q - 1) * k + 1  # stored frames: Simula at q model times, k - 1 interpolated frames in between
    (_, xlo, xhi), (_, ylo, yhi) = (r for r in targets.RANGES["swe"] if r[0] in ("src0_x", "src0_y"))
    X, Y = np.meshgrid(np.linspace(xlo, xhi, a.nx), np.linspace(ylo, yhi, a.ny))
    P = params("swe", X.size, **SWE_FIXED, src0_x=X.ravel(), src0_y=Y.ravel())
    ics, st = targets.make_ics("swe", P)
    t0 = time.time()
    eta = upsample(chain(pred, ics, st, 1, q)[:, :, 0], k)
    eta_s = solve("swe", ics, st, 1, qo)[:, :, 0]  # the solver's true frames at every stored time
    h, h_s = np.maximum(eta - bvis, 0), np.maximum(eta_s - bvis, 0)
    flood = (h.max(1) * land).sum((1, 2))  # flood volume proxy: sum over land of the max depth
    on = np.arange(1, qo) % ((qo - 1) // (s["frames"] - 1)) == 0  # fine times that are the solver's own frames
    err = np.array([[relL2(eta[i, 1:][m], eta_s[i, 1:][m]) for m in (on, ~on)] for i in range(len(P))])
    print(f"tsunami: {len(P)} scenarios {time.time() - t0:.0f}s; eta relL2 on-frame {err[:, 0].mean():.3f} "
          f"off-frame {err[:, 1].mean():.3f}", flush=True)
    sc = []
    for i, (x, y) in enumerate(zip(X.ravel(), Y.ravel())):
        save(f"{out}/e{i}/h.npy", h[i].astype(np.float32))
        save(f"{out}/e{i}/h_solver.npy", h_s[i].astype(np.float32))
        sc.append({"id": f"e{i}", "x": frac(x), "y": frac(y), "frames": f"e{i}/h.npy", "solver": f"e{i}/h_solver.npy",
                   "flood_m3": round(float(flood[i]) * (s["L"] * 1e3 / s["N"]) ** 2, 1), "eta_relL2": round(float(err[i].mean()), 4)})
    save(f"{out}/b.npy", bvis.astype(np.float32))
    # risk map: R random quakes from the sampler, Simula at its own 14 frames, per-quake max flood depth on land
    _, ri, rs = targets.sample("swe", a.risk, np.random.default_rng(a.seed))
    t1, mx = time.time(), []
    for i in range(0, a.risk, 100):
        e = pred(ri[i : i + 100], rs[i : i + 100], np.arange(1, s["frames"]) / (s["frames"] - 1))[:, :, 0]
        mx.append((np.maximum(e - bvis, 0).max(1) * land).astype(np.float32))
    mx, K = np.concatenate(mx), a.risk_k
    risk = np.stack([mx[: a.risk * (k + 1) // K].max(0) for k in range(K)])
    save(f"{out}/risk_map.npy", risk.astype(np.float32))
    print(f"risk map: {a.risk} quakes {time.time() - t1:.0f}s, max depth {risk[-1].max():.2f} m, "
          f"flooded cells {(risk[-1] > 0.01).sum()}", flush=True)
    d = sc[int(np.argmax(flood))]["id"]
    idx = {"version": 1, "mock": False, "note": (
        f"Simula ({a.models}/swe) vs the GPU pseudo-spectral solver, {len(P)} epicentres on the sampler's source region "
        f"(x {xlo}-{xhi}, y {ylo}-{yhi} of the square), other knobs fixed {SWE_FIXED}; Simula at its {q} frames per "
        f"140 s window, {k - 1} motion-compensated frames in between (baseline_FNO/interp.py); solver track = true "
        f"solver frames at every stored time. eta relL2 vs solver: {err[:, 0].mean():.3f} at Simula's frames, "
        f"{err[:, 1].mean():.3f} at the interpolated ones. h = max(eta - b, 0), b = demos.coast visual bed. Risk: {a.risk} random "
        f"sampler quakes (seed {a.seed}), Simula at 10 s frames, cumulative max over the first k/{K}."),
        "grid": {"nx": s["N"], "ny": s["N"], "lx": s["L"] * 1e3, "ly": s["L"] * 1e3}, "bed": "b.npy",
        "dt": round(s["frame_dt"] * targets.KS * (s["frames"] - 1) / (qo - 1), 4), "time_unit": "s",
        "view": {"vexag": 4, "wexag": 6}, "default": d, "scenarios": sc,
        "risk": {"frames": "risk_map.npy", "count": a.risk, "unit": "m"}}
    json.dump(idx, open(f"{out}/index.json", "w"), indent=1)
    print(f"default {d}, wrote {out}", flush=True)


# ---------------------------------------------------------------- heart: scars x ablation lines
def seg_mask(p0, p1, half, N=128, edge=0.007):
    """Smooth mask (N, N) of a line segment (unit-square endpoints), 1 within `half` of it (no periodic wrap)."""
    X, Y = targets._grid(N)
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    d = p1 - p0
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / (d @ d), 0, 1)
    r = np.hypot(X - p0[0] - t * d[0], Y - p0[1] - t * d[1])
    return 0.5 * (1 + np.tanh((half - r) / edge))


def ray(p, u, s, lo=0.03, hi=0.97):
    """Longest step <= s from p along u that stays inside [lo, hi]^2."""
    return min([s] + [(b - p[k]) / u[k] for k in (0, 1) for b in (lo, hi) if abs(u[k]) > 1e-9 and (b - p[k]) / u[k] > 0])


def lines_from(c, r, rng=None, n_rand=0, lengths=(0.3, 0.45), n_ang=8):
    """Candidate ablation lines inside the box: from the scar edge outward (n_ang angles x lengths), plus n_rand random
    long segments anywhere (through or between the spiral cores)."""
    out = []
    for L in lengths:
        for th in np.arange(n_ang) * 2 * np.pi / n_ang:
            u = np.array([np.cos(th), np.sin(th)])
            p0 = np.asarray(c) + 0.8 * r * u
            s = ray(p0, u, L)
            if s > 0.1:
                out.append((p0, p0 + s * u))
    for _ in range(n_rand):
        p, th, L = rng.uniform(0.15, 0.85, 2), rng.uniform(0, np.pi), rng.uniform(0.3, 0.8)
        u = np.array([np.cos(th), np.sin(th)])
        out.append((p - ray(p, -u, L / 2) * u, p + ray(p, u, L / 2) * u))
    return out


def active(u, thr=0.3):  # extinct = max u below thr (Simula's particle mean blurs peaks)
    return u.reshape(len(u), -1).max(1) > thr


def scar_only(a, out, pred, scars, ic, st, e_ic, tally, W=2):
    """Scar placements x (Simula, solver) over W windows at the same times, with each one's re-entry verdict."""
    k = a.interp
    f = upsample(chain(pred, ic, st, W, 15)[:, :, 0], k)  # Simula at its own frames + motion-compensated in between
    y = solve("ap", ic, st, W, 14 * k + 1)[:, :, 0]  # the solver's true frames at every stored time
    sim, sol = active(f[:, -1]), active(y[:, -1])
    ent = []
    for i, (cx, cy) in enumerate(scars):
        save(f"{out}/s{i}/u.npy", f[i].astype(np.float16))
        save(f"{out}/s{i}/u_solver.npy", y[i].astype(np.float16))
        ent.append({"id": f"s{i}", "x": frac(cx), "y": frac(cy), "r": frac(a.scar_r), "frames": f"s{i}/u.npy",
                    "solver": f"s{i}/u_solver.npy", "spiral": bool(sim[i]), "solver_spiral": bool(sol[i]),
                    "u_relL2": round(relL2(f[i, 1:], y[i, 1:]), 4), "ablations": []})
    ag = sim == sol
    print(f"scar_only: solver re-entry {sol.sum()}/{len(scars)}, Simula {sim.sum()}, agree {ag.sum()}/{len(scars)}", flush=True)
    idx = {"version": 1, "mock": False, "variant": "scar_only", "note": (
        f"Simula ({a.models}/ap) vs the GPU pseudo-spectral solver. No ablations: low-conductivity lines (D 0.02, and "
        f"0.005 / 0.001 below the training range) stopped none of 328 candidate lines in the solver (the tissue stays "
        f"excitable). One sampler IC (a broken front, no stimuli), chosen among {a.n_ic} for the most agreement, then mixed "
        f"outcomes. Over all {a.n_ic} ICs x {len(scars)} scars Simula's verdict agrees with the solver on "
        f"{tally[:, 0].sum()}/{tally[:, 1].sum()}; on the {int(tally[:, 2].sum())} ICs where the scar placement decides "
        f"(mixed solver outcomes) on {tally[tally[:, 2] > 0, 0].sum()}/{tally[tally[:, 2] > 0, 1].sum()}. Scar r {a.scar_r} D {a.scar_D} on a {a.n_scar}x{a.n_scar} grid; {W} chained Simula windows at its 15 "
        f"frames, {k - 1} motion-compensated frames in between; solver = true frames at the same times. spiral = activity "
        f"left at the end (max u > 0.3; else extinct). Simula agrees with the solver on {ag.sum()}/{len(ag)} scars; "
        f"solver re-entry {sol.sum()}/{len(ag)}. u relL2 vs solver over the rollout {np.mean([e['u_relL2'] for e in ent]):.3f} "
        f"(IC search: {e_ic:.3f} at Simula's own frames)."),
        "grid": {"nx": 128, "ny": 128}, "geometry": "sheet", "range": [0, 1], "time_unit": "ms",
        "dt": round(targets.SPECS["ap"]["frame_dt"] / k * AP_MS, 4),
        "agreement": {"scars": len(ag), "agree": int(ag.sum()), "solver_spiral": int(sol.sum()), "simula_spiral": int(sim.sum())},
        "scars": ent}
    json.dump(idx, open(f"{out}/index.json", "w"), indent=1)
    print(f"wrote {out}", flush=True)


def heart(a):
    out, N = f"{a.out}/heart", 128
    pred = load(a.models, "ap")
    g = np.linspace(0.2, 0.8, a.n_scar)
    scars = [(x, y) for y in g for x in g]
    R, Dm, W, abl = a.scar_r, a.scar_D, 2, a.variant == "ablation"  # W: windows of the scar rollout (and of an ablation)
    # 1 one spiral-seeding IC from the sampler (a single broken front, no point stimuli).  ablation: re-enters for every
    #   scar (solver active after 2 W windows, Simula after W), then the best Simula-solver match.  scar_only: the scar
    #   placement decides (solver re-entry for some scars, extinction for others), then the most Simula-solver agreement
    cand = targets.sample_params("ap", a.n_ic, np.random.default_rng(a.seed))
    for k in ("front1_on", "stim0_on", "stim1_on"):
        cand[:, targets.KNOBS["ap"].index(k)] = 0
    rows, tally = [], []
    for j, p in enumerate(cand):
        P = params("ap", len(scars), scar_x=[c[0] for c in scars], scar_y=[c[1] for c in scars], scar_radius=R, scar_D=Dm)
        ns = targets.KNOBS["ap"].index("scar_x")
        P[:, :ns] = p[:ns]  # fronts + stimuli from the candidate, scar from the grid
        ic, st = targets.make_ics("ap", P)
        y = solve("ap", ic, st, 2 * W if abl else W)
        m = chain(pred, ic, st, W, 15)
        e = np.mean([relL2(m[i, 1:, 0], y[i, 1 : W * 14 + 1, 0]) for i in range(len(P))])
        sol, sim = active(y[:, -1, 0]), active(m[:, -1, 0])
        mix = int(min(sol.sum(), (~sol).sum()))  # scar placement decides re-entry (some re-enter, some die out)
        ok = (int((sol & sim).sum()),) if abl else (int((sol == sim).sum()), int(mix > 0), mix)
        rows.append((ok, e, P, ic, st))
        tally.append((int((sol == sim).sum()), len(P), mix > 0))
        print(f"IC {j}: solver re-entry {sol.sum()}/{len(P)}, Simula {sim.sum()}, agree {(sol == sim).sum()}, u relL2 {e:.3f}", flush=True)
    ok, e_ic, P, ic, st = min(rows, key=lambda r: (tuple(-x for x in r[0]), r[1]))
    if not abl:
        return scar_only(a, out, pred, scars, ic, st, e_ic, np.array(tally))
    # 2 scar rollouts (Simula, q per window) -> the established arrhythmia; 3 candidate ablation lines from it
    m = chain(pred, ic, st, W, 15)
    y = solve("ap", ic, st, 2 * W)
    start = y[:, W * 14]  # the solver's established arrhythmia after the scar rollout's W windows (u, v)
    ctrl_sim = active(chain(pred, start, st, W, 15)[:, -1, 0])  # no line: Simula from that state
    ctrl_sol = active(y[:, -1, 0])  # no line: the solver keeps spiralling (IC choice)
    print(f"controls (no line, still active after {W} windows): Simula {ctrl_sim.sum()}/{len(scars)}, solver "
          f"{ctrl_sol.sum()}/{len(scars)}", flush=True)
    lr = np.random.default_rng(a.seed + 1)
    cands = [lines_from(c, R, lr, a.n_rand) for c in scars]
    abl_st = np.stack([np.minimum(st[i, 0], 1 - (1 - a.line_D) * seg_mask(p0, p1, a.line_half))[None]
                       for i in range(len(scars)) for p0, p1 in cands[i]]).astype(np.float32)
    owner = np.array([i for i in range(len(scars)) for _ in cands[i]])
    ys = solve("ap", start[owner], abl_st, W)[:, -1, 0]  # verification: the solver from the same state, same D
    ms = chain(pred, start[owner], abl_st, W, 15)[:, -1, 0]
    sim_stop, sol_stop = ~active(ms), ~active(ys)
    agree = sim_stop == sol_stop
    print(f"ablations: {len(owner)} candidates, Simula stops {sim_stop.sum()}, solver stops {sol_stop.sum()}, "
          f"agree {agree.mean():.0%}", flush=True)
    rng, entries, shown = np.random.default_rng(a.seed), [], []
    for i, (cx, cy) in enumerate(scars):
        k = np.flatnonzero(owner == i)
        ag = [j for j in k if agree[j]]
        pick = [j for j in ag if sim_stop[j]][:1] + [j for j in ag if not sim_stop[j]][:1]  # both outcomes if possible
        pick += [j for j in ag if j not in pick][: 3 - len(pick)]
        pick += [j for j in k if j not in pick][: 3 - len(pick)]  # disagreeing lines only if too few agree
        pick = list(rng.permutation(pick))
        fr = upsample(chain(pred, start[owner[pick]], abl_st[pick], W, 15)[:, :, 0], a.interp)
        save(f"{out}/s{i}/u.npy", upsample(m[i : i + 1, :, 0], a.interp)[0].astype(np.float16))
        abl = []
        for n, (j, f) in enumerate(zip(pick, fr)):
            lid = "abc"[n]
            p0, p1 = cands[i][int(j - k[0])]
            save(f"{out}/s{i}/{lid}.npy", f.astype(np.float16))
            abl.append({"id": lid, "label": f"Line {lid.upper()}", "line": [[frac(p0[0]), frac(p0[1])], [frac(p1[0]), frac(p1[1])]],
                        "frames": f"s{i}/{lid}.npy", "stops_arrhythmia": bool(sim_stop[j]), "solver_stops": bool(sol_stop[j])})
            shown.append(bool(agree[j]))
        entries.append({"id": f"s{i}", "x": frac(cx), "y": frac(cy), "r": frac(R), "frames": f"s{i}/u.npy", "ablations": abl})
    dt = targets.SPECS["ap"]["frame_dt"] / a.interp
    idx = {"version": 1, "mock": False, "variant": "ablation", "note": (
        f"Simula ({a.models}/ap). IC: one sampler front (no stimuli), chosen among {a.n_ic} for re-entry with every scar "
        f"(scars re-entering: {ok}/{len(scars)}, solver after {2 * W} windows and Simula after {W}); Simula u relL2 vs solver over {W} windows {e_ic:.3f}. "
        f"Scars r {R} D {Dm}; {W} chained Simula windows at its 15 frames, {a.interp - 1} motion-compensated frames in "
        f"between. Ablation lines: D = {a.line_D} within {a.line_half} of the segment, applied to the solver's state at the end "
        f"of the scar rollout (the established arrhythmia), {W} Simula windows. stops_arrhythmia = Simula's verdict "
        f"(max u < 0.3 at the end); solver_stops = the solver from the same state and D. Without a line the solver "
        f"keeps spiralling for {ctrl_sol.sum()}/{len(scars)} scars, Simula for {ctrl_sim.sum()}/{len(scars)}. Simula and solver agree on "
        f"{agree.sum()}/{len(agree)} candidate lines ({agree.mean():.0%}); shown lines agree {sum(shown)}/{len(shown)}."),
        "grid": {"nx": N, "ny": N}, "geometry": "sheet", "dt": round(dt * AP_MS, 4), "time_unit": "ms", "range": [0, 1],
        "agreement": {"candidates": int(len(agree)), "agree": int(agree.sum()), "shown": len(shown), "shown_agree": int(sum(shown))},
        "scars": entries}
    json.dump(idx, open(f"{out}/index.json", "w"), indent=1)
    print(f"wrote {out}; shown lines agree {sum(shown)}/{len(shown)}", flush=True)


# ---------------------------------------------------------------- fire: ignitions x firebreaks, does the town burn?
TOWN, LAKE = (0.74, 0.62, 0.07), (0.6, 0.86, 0.07)  # (x, y, r) in the unit square; wind 0.4 toward +x at 30 deg


def fire_ic(fuel, ign, brk=None):
    """targets_fire.make_ic with one fixed fuel map (its noise seed otherwise hashes the whole knob vector): ignition
    (x, y, r), one firebreak (x, y, angle, half_len, width) or None, and the lake -> ic (2, N, N)."""
    import targets_fire as tf

    kw = dict(ign0_on=1, ign0_x=ign[0], ign0_y=ign[1], ign0_radius=ign[2], lake_on=1, lake_x=LAKE[0], lake_y=LAKE[1],
              lake_radius=LAKE[2])
    if brk is not None:
        kw.update(brk0_on=1, **{f"brk0_{n}": v for n, v in zip(("x", "y", "angle", "half_len", "width"), brk)})
    f0, tf._fuel = tf._fuel, lambda p, k: fuel.copy()  # make_ic cuts the breaks into it in place
    try:
        return tf.make_ic(params("fire", 1, **kw)[0]).astype(np.float32)
    finally:
        tf._fuel = f0


def fire_d(tr, f0):
    """(..., T, 2, N, N) u, f and the IC fuel f0 -> display field d (..., T, N, N) in [0, 1] (umax = fmax = 1)."""
    u, f, f0 = tr[..., 0, :, :], tr[..., 1, :, :], f0[..., None, :, :]
    d = np.where(f0 < 0.1, 0.45, 0.05 + 0.15 * np.clip(f, 0, 1))  # unburned fuel; fuel-free (lake, firebreak)
    d = np.where(f0 - f > 0.1, 0.32, d)  # burned
    return np.where(u > 0.1, 0.55 + 0.45 * np.clip(u, 0, 1), d).astype(np.float16)  # burning


def town_burns(tr, f0):
    """(n, T, 2, N, N) -> the town (TOWN disk) burned or burning over > 10% of its area at the end (targets_fire.stats
    conventions: burned = fuel below half its IC value, burning = u > 0.3)."""
    X, Y = targets._grid(f0.shape[-1])
    t = np.hypot(X - TOWN[0], Y - TOWN[1]) < TOWN[2]
    return (((tr[:, -1, 1] < 0.5 * f0) | (tr[:, -1, 0] > 0.3))[:, t]).mean(1) > 0.1


def pick_mix(idx, stop, agree, n):
    """Up to n of the candidates idx, agreeing ones only, alternating the two outcomes."""
    a, b = [j for j in idx if agree[j] and stop[j]], [j for j in idx if agree[j] and not stop[j]]
    return [j for p in zip(a + [None] * len(b), b + [None] * len(a)) for j in p if j is not None][:n]


def fire(a):
    import targets_fire as tf

    out, k, N = f"{a.out}/fire", a.interp, 128
    pred = load(a.models, "fire")
    pb = params("fire", 1, fuel_mean=0.88, fuel_contrast=0.15, fuel_corr=0.08)[0]
    fuel = tf._fuel(pb, dict(zip(tf.KNOBS, pb)))
    tx, ty, _ = TOWN
    sites = [(tx - d * np.cos(np.pi / 6 + ph), ty - d * np.sin(np.pi / 6 + ph), 0.05)  # upwind of the town
             for d in (0.3, 0.42) for ph in np.radians([-50, -25, 0, 25, 50])]

    def breaks(s):  # across the site -> town path: position along it x half length x width (2.3 cells: jumped; 4.5: holds)
        dx, dy = tx - s[0], ty - s[1]
        return [(s[0] + q * dx, s[1] + q * dy, np.arctan2(dy, dx) + np.pi / 2, hl, w)
                for q in (0.4, 0.6) for hl in (0.08, 0.16, 0.26) for w in (0.018, 0.035)]

    cand = [(i, b) for i, s in enumerate(sites) for b in [None] + breaks(s)]
    ics = np.stack([fire_ic(fuel, sites[i], b) for i, b in cand])
    st, t14 = np.zeros((len(ics), 0, N, N), np.float32), np.arange(1, 15) / 14
    tj = f"{a.out}/timing.json"
    T = json.load(open(tj)) if os.path.exists(tj) else {}
    T["fire"] = {"simula_seconds": clock(lambda: pred(ics[:1], st[:1], t14)),
                 "solver_seconds": clock(lambda: solve("fire", ics[:1], st[:1], batch=1)),
                 "what": f"one fire rollout, 15 frames over 21 t.u., batch 1 ({len(np.load(f'{a.models}/fire/z.npy'))} particle)"}
    json.dump(T, open(tj, "w"), indent=1)
    print("timing", T["fire"], flush=True)
    m, y = chain(pred, ics, st, 1, 15), solve("fire", ics, st, 1)
    sim, sol = town_burns(m, ics[:, 1]), town_burns(y, ics[:, 1])
    agree, owner = sim == sol, np.array([i for i, _ in cand])
    nb = np.array([b is None for _, b in cand])
    print(f"fire: no break: town burns solver {sol[nb].sum()}/{nb.sum()}, Simula {sim[nb].sum()}; lines: solver saves "
          f"{(~sol[~nb]).sum()}/{(~nb).sum()}, Simula {(~sim[~nb]).sum()}, agree {agree[~nb].sum()}", flush=True)
    rows = []  # sites where the town burns without a break (solver), ranked: Simula agrees, a mix of shown outcomes
    for i in range(len(sites)):
        j0 = int(np.flatnonzero(nb & (owner == i))[0])
        idx = sorted(np.flatnonzero(~nb & (owner == i)), key=lambda j: not agree[j])  # agreeing ones first
        sh = pick_mix(idx, ~sol, np.ones(len(sol), bool), a.n_lines)  # a mix of the solver's saves and fails
        if sol[j0]:
            rows.append(((int(agree[j0]), len({bool(sol[j]) for j in sh}), sum(agree[j] for j in sh)), i, j0, sh))
    rows = sorted(rows, key=lambda r: r[0], reverse=True)[: a.n_sites]
    J = [j for _, _, j0, sh in rows for j in [j0] + sh]
    fd = fire_d(upsample(m[J], k), ics[J, 1])  # Simula at its own frames + motion-compensated in between
    yd = fire_d(solve("fire", ics[J], st[J], 1, 14 * k + 1), ics[J, 1])  # the solver's true frames at every stored time
    D, ent = dict(zip(J, zip(fd, yd))), []
    for n, (_, i, j0, sh) in enumerate(rows):
        sid = f"s{n}"
        save(f"{out}/{sid}/u.npy", D[j0][0]), save(f"{out}/{sid}/u_solver.npy", D[j0][1])
        abl = []
        for c, j in enumerate(sh):
            x, y_, th, hl, w = cand[j][1]
            e = hl * np.array([np.cos(th), np.sin(th)])
            lid = "abcdef"[c]
            save(f"{out}/{sid}/{lid}.npy", D[j][0]), save(f"{out}/{sid}/{lid}_solver.npy", D[j][1])
            abl.append({"id": lid, "label": f"Break {lid.upper()}", "line": [[frac(x - e[0]), frac(y_ - e[1])], [frac(x + e[0]), frac(y_ + e[1])]],
                        "width": frac(w), "frames": f"{sid}/{lid}.npy", "solver": f"{sid}/{lid}_solver.npy",
                        "stops_arrhythmia": bool(~sim[j]), "solver_stops": bool(~sol[j])})
        ent.append({"id": sid, "x": frac(sites[i][0]), "y": frac(sites[i][1]), "r": frac(sites[i][2]), "frames": f"{sid}/u.npy",
                    "solver": f"{sid}/u_solver.npy", "spiral": bool(sim[j0]), "solver_spiral": bool(sol[j0]), "ablations": abl})
    shown = [bool(agree[j]) for _, _, _, sh in rows for j in sh]
    saves = sum(bool(~sim[j]) for _, _, _, sh in rows for j in sh)
    L = ~nb
    idx = {"version": 1, "mock": False, "variant": "firebreak", "grid": {"nx": N, "ny": N}, "geometry": "sheet",
           "dt": targets.SPECS["fire"]["frame_dt"] / k, "time_unit": "t.u.", "range": [0, 1],
           "asset": {"x": frac(TOWN[0]), "y": frac(TOWN[1]), "r": frac(TOWN[2]), "label": "Town"}, "wind": [0.866, 0.5],
           "agreement": {"lines": int(L.sum()), "lines_agree": int(agree[L].sum()), "no_break": int(nb.sum()),
                         "no_break_agree": int(agree[nb].sum()), "shown": len(shown), "shown_agree": int(sum(shown)), "shown_saves": saves},
           "note": (
        f"Simula ({a.models}/fire, 1 particle) vs the GPU pseudo-spectral solver, targets_fire (u heat, f fuel; wind 0.4 "
        f"toward +x at 30 deg). One fixed fuel map (mean 0.88, contrast 0.15, corr 0.08), a lake {LAKE}, the town {TOWN} "
        f"(x, y, r of the unit square). {len(sites)} ignition sites (r 0.05) 0.3 / 0.42 upwind, each with 12 candidate "
        f"firebreaks (fuel-free strips across the path, 2.3 or 4.5 cells wide); shown: the sites where the solver burns "
        f"the town without a break, and up to {a.n_lines} breaks per site mixing the solver's saves and fails (agreeing "
        f"ones first). Simula saves the town with {(~sim[~nb]).sum()}/{(~nb).sum()} breaks, the solver with "
        f"{(~sol[~nb]).sum()}: Simula's fire does not stop at the breaks, so every shown save is a disagreement. Verdict: the town burned (fuel < half its IC value) or burning (u > 0.3) over > 10% of its area at the end of the 21 t.u. window "
        f"(spiral = burns with no break; stops_arrhythmia = Simula: town safe; solver_stops = solver: town safe). "
        f"Agreement over all candidates: lines {agree[L].sum()}/{L.sum()}, no-break {agree[nb].sum()}/{nb.sum()}. "
        f"Frames: Simula at its 15 frames, {k - 1} motion-compensated frame(s) in between (on u, f); solver = true frames "
        f"at the same times. d encoding (same for both): burning (u > 0.1) 0.55 + 0.45 clip(u); burned (fuel dropped by "
        f"> 0.1 from the IC) 0.32; fuel-free (IC fuel < 0.1, not burning) 0.45; unburned fuel 0.05 + 0.15 f."),
           "scars": ent}
    json.dump(idx, open(f"{out}/index.json", "w"), indent=1)
    print(f"wrote {out}: {len(ent)} sites, {len(shown)} breaks shown ({saves} saves), all agree {all(shown)}", flush=True)


# ---------------------------------------------------------------- timings
def clock(f, reps=3):
    f()  # warm-up / compile
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()  # outputs are numpy: device work is finished (np.asarray blocks until ready)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def timing(a):
    import jax

    sw, ap = load(a.models, "swe"), load(a.models, "ap")
    t14 = np.arange(1, 15) / 14
    _, si, ss = targets.sample("swe", a.batch_n, np.random.default_rng(a.seed))
    _, ai, ast = targets.sample("ap", 1, np.random.default_rng(a.seed))
    T = {"tsunami": {"simula_seconds": clock(lambda: sw(si[:1], ss[:1], t14)),
                     "solver_seconds": clock(lambda: solve("swe", si[:1], ss[:1], batch=1)),
                     "what": "one quake, 15 frames over 140 s"},
         "heart": {"simula_seconds": clock(lambda: chain(ap, ai, ast, 2, 15)),
                   "solver_seconds": clock(lambda: solve("ap", ai, ast, 2, batch=1)),
                   "what": "one AP rollout, 2 windows (70 t.u. = 903 ms of tissue time), 29 frames"}}
    print(T, flush=True)
    def best(f, opts):  # each option timed on all batch_n quakes; one that does not fit in memory is skipped
        r = []
        for o in opts:
            try:
                r.append((clock(lambda: f(o), reps=1), o))
            except Exception as e:  # noqa: BLE001  (XLA out of memory)
                print(f"batch {o}: {type(e).__name__}", flush=True)
        print(r, flush=True)
        return min(r)

    simb, c = best(lambda c: sw(si, ss, t14, c), a.chunks)
    solb, bs = best(lambda b: solve("swe", si, ss, batch=b), a.solver_batches)
    T["batch"] = {"simula_seconds": simb, "solver_seconds": solb, "scenarios": a.batch_n, "simula_batch": c,
                  "solver_batch": bs, "what": f"{a.batch_n} random sampler quakes, 15 frames each, all run on both "
                  "sides; each at its fastest batch size (ICs per device call)"}
    T["hardware"] = f"{jax.devices()[0].device_kind}; solver = our GPU pseudo-spectral solver (pde.rollout, JAX)"
    T["simula"] = (f"{a.models}: {len(np.load(f'{a.models}/swe/z.npy'))} particle(s), single precision; wall clock after "
                   "warm-up, incl. host-side IC normalization and the copy of every frame back to the host")
    json.dump(T, open(f"{a.out}/timing.json", "w"), indent=1)
    print(json.dumps(T), flush=True)


def timing_seq(a):
    """batch_n quakes run one at a time on both sides (no batching); added to timing.json as "sequential"."""
    sw = load(a.models, "swe")
    t14 = np.arange(1, 15) / 14
    _, si, ss = targets.sample("swe", a.batch_n, np.random.default_rng(a.seed))
    one = lambda i: (sw(si[i : i + 1], ss[i : i + 1], t14), solve("swe", si[i : i + 1], ss[i : i + 1], batch=1))
    one(0)  # warm-up / compile both
    seq = lambda f: clock(lambda: [f(i) for i in range(len(si))], reps=1)
    T = json.load(open(f"{a.out}/timing.json"))
    T["sequential"] = {"simula_seconds": seq(lambda i: sw(si[i : i + 1], ss[i : i + 1], t14)),
                       "solver_seconds": seq(lambda i: solve("swe", si[i : i + 1], ss[i : i + 1], batch=1)),
                       "scenarios": a.batch_n, "what": f"{a.batch_n} random sampler quakes, 15 frames each, run one at a "
                       "time on both sides (batch 1)"}
    json.dump(T, open(f"{a.out}/timing.json", "w"), indent=1)
    print(json.dumps(T["sequential"]), flush=True)


def check():
    m = seg_mask((0.2, 0.5), (0.6, 0.5), 0.02)
    assert m[64, 50] > 0.99 and m[64, 100] < 0.01 and m[40, 50] < 0.01  # on the line, past its end, off to the side
    L = lines_from((0.5, 0.3), 0.12, np.random.default_rng(0), 20)
    assert len(L) == 36 and all(0.03 - 1e-9 <= v <= 0.97 + 1e-9 for p in L for q in p for v in q)
    P = params("swe", 3, src0_x=[0.1, 0.2, 0.3], src0_on=1)
    assert P[:, targets.KNOBS["swe"].index("src0_x")].tolist() == [np.float32(0.1), np.float32(0.2), np.float32(0.3)]
    assert frac(127 / 128) == 1.0
    assert pick_mix(range(5), [1, 1, 0, 0, 1], [1, 1, 1, 0, 1], 4) == [0, 2, 1, 4]
    tr = np.zeros((1, 2, 2, 128, 128), np.float32)
    tr[..., 1, :, :] = 1
    tr[0, 1, 0, 0, 0], tr[0, 1, 1, 0, 1] = 1.0, 0.5  # burning; burned
    f0 = np.ones((1, 128, 128), np.float32)
    f0[0, 0, 2] = 0  # fuel-free
    d = fire_d(tr, f0)[0, 1, 0]
    assert abs(d[0] - 1.0) < 1e-3 and abs(d[1] - 0.32) < 1e-3 and abs(d[2] - 0.45) < 1e-3 and abs(d[3] - 0.2) < 1e-3
    print("check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tsunami", "heart", "fire", "timing", "timing_seq", "check"])
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--out", default="/workspace/results/demo_bank")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nx", type=int, default=4, help="tsunami: epicentres across the source region (x)")
    ap.add_argument("--ny", type=int, default=7, help="tsunami: epicentres along the coast (y)")
    ap.add_argument("--q_swe", type=int, default=15, help="tsunami: Simula frames per 140 s window (15: its own)")
    ap.add_argument("--interp", type=int, default=4, help="stored frames per Simula frame (motion-compensated)")
    ap.add_argument("--risk", type=int, default=1000)
    ap.add_argument("--risk_k", type=int, default=10)
    ap.add_argument("--n_ic", type=int, default=16, help="heart: candidate ICs searched")
    ap.add_argument("--variant", default="scar_only", choices=["scar_only", "ablation"])
    ap.add_argument("--n_scar", type=int, default=3, help="heart: scar grid per axis")
    ap.add_argument("--n_sites", type=int, default=8, help="fire: ignition sites shown")
    ap.add_argument("--n_lines", type=int, default=5, help="fire: firebreaks shown per site")
    ap.add_argument("--scar_r", type=float, default=0.12)
    ap.add_argument("--scar_D", type=float, default=0.03)
    ap.add_argument("--line_half", type=float, default=0.025)
    ap.add_argument("--line_D", type=float, default=0.02)
    ap.add_argument("--n_rand", type=int, default=40, help="heart: random candidate lines per scar")
    ap.add_argument("--batch_n", type=int, default=1000)
    ap.add_argument("--chunks", type=int, nargs="+", default=[4, 8, 16, 32, 64], help="timing: Simula ICs per call")
    ap.add_argument("--solver_batches", type=int, nargs="+", default=[100, 250, 500, 1000])
    a = ap.parse_args()
    check() if a.cmd == "check" else {"tsunami": tsunami, "heart": heart, "fire": fire, "timing": timing, "timing_seq": timing_seq}[a.cmd](a)

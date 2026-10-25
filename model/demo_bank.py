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
    """(n, T, N, N) -> (n, (T - 1) k + 1, N, N): motion-compensated in-between frames (baseline_FNO/interp.py)."""
    if k == 1:
        return x
    sys.path.append(os.path.join(HERE, "..", "baseline_FNO"))
    import interp

    return np.stack([interp.upsample(f[:, None], k)[:, 0] for f in x])


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


def check():
    m = seg_mask((0.2, 0.5), (0.6, 0.5), 0.02)
    assert m[64, 50] > 0.99 and m[64, 100] < 0.01 and m[40, 50] < 0.01  # on the line, past its end, off to the side
    L = lines_from((0.5, 0.3), 0.12, np.random.default_rng(0), 20)
    assert len(L) == 36 and all(0.03 - 1e-9 <= v <= 0.97 + 1e-9 for p in L for q in p for v in q)
    P = params("swe", 3, src0_x=[0.1, 0.2, 0.3], src0_on=1)
    assert P[:, targets.KNOBS["swe"].index("src0_x")].tolist() == [np.float32(0.1), np.float32(0.2), np.float32(0.3)]
    assert frac(127 / 128) == 1.0
    print("check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tsunami", "heart", "timing", "check"])
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
    ap.add_argument("--scar_r", type=float, default=0.12)
    ap.add_argument("--scar_D", type=float, default=0.03)
    ap.add_argument("--line_half", type=float, default=0.025)
    ap.add_argument("--line_D", type=float, default=0.02)
    ap.add_argument("--n_rand", type=int, default=40, help="heart: random candidate lines per scar")
    ap.add_argument("--batch_n", type=int, default=1000)
    ap.add_argument("--chunks", type=int, nargs="+", default=[4, 8, 16, 32, 64], help="timing: Simula ICs per call")
    ap.add_argument("--solver_batches", type=int, nargs="+", default=[100, 250, 500, 1000])
    a = ap.parse_args()
    check() if a.cmd == "check" else {"tsunami": tsunami, "heart": heart, "timing": timing}[a.cmd](a)

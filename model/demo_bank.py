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


# ---------------------------------------------------------------- tsunami: epicentre grid + risk map
SWE_FIXED = dict(src0_on=1, src0_angle=0.0, src0_amp=1.6, src0_width=0.05, src0_length=0.18, src0_sub_on=1,
                 src0_sub_side=1, src0_sub_off=1.85, src0_sub_amp=0.5)  # dramatic but typical; depression landward


def tsunami(a):
    out, s = f"{a.out}/tsunami", targets.SPECS["swe"]
    pred = load(a.models, "swe")
    _, _, bvis, _ = demos.coast(s)
    land, q = bvis > 0, a.q_swe
    (_, xlo, xhi), (_, ylo, yhi) = (r for r in targets.RANGES["swe"] if r[0] in ("src0_x", "src0_y"))
    X, Y = np.meshgrid(np.linspace(xlo, xhi, a.nx), np.linspace(ylo, yhi, a.ny))
    P = params("swe", X.size, **SWE_FIXED, src0_x=X.ravel(), src0_y=Y.ravel())
    ics, st = targets.make_ics("swe", P)
    t0 = time.time()
    eta = chain(pred, ics, st, 1, q)[:, :, 0]
    eta_s = solve("swe", ics, st, 1, q)[:, :, 0]
    h, h_s = np.maximum(eta - bvis, 0), np.maximum(eta_s - bvis, 0)
    flood = (h.max(1) * land).sum((1, 2))  # flood volume proxy: sum over land of the max depth
    on = np.arange(1, q) % ((q - 1) // (s["frames"] - 1)) == 0  # fine times that are the solver's own frames
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
        f"(x {xlo}-{xhi}, y {ylo}-{yhi} of the square), other knobs fixed {SWE_FIXED}; Simula queried directly at "
        f"{q} times per 140 s window. eta relL2 vs solver: {err[:, 0].mean():.3f} at the solver's 10 s frames, "
        f"{err[:, 1].mean():.3f} in between. h = max(eta - b, 0), b = demos.coast visual bed. Risk: {a.risk} random "
        f"sampler quakes (seed {a.seed}), Simula at 10 s frames, cumulative max over the first k/{K}."),
        "grid": {"nx": s["N"], "ny": s["N"], "lx": s["L"] * 1e3, "ly": s["L"] * 1e3}, "bed": "b.npy",
        "dt": round(s["frame_dt"] * targets.KS * (s["frames"] - 1) / (q - 1), 4), "time_unit": "s",
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


def lines_from(c, r, lengths=(0.3, 0.45), n_ang=8, lo=0.03, hi=0.97):
    """Candidate ablation lines from the scar edge outward: n_ang angles x lengths, clipped inside [lo, hi]^2."""
    out = []
    for L in lengths:
        for th in np.arange(n_ang) * 2 * np.pi / n_ang:
            u = np.array([np.cos(th), np.sin(th)])
            p0 = np.asarray(c) + 0.8 * r * u
            s = min([L] + [(b - p0[k]) / u[k] for k in (0, 1) for b in (lo, hi) if abs(u[k]) > 1e-9 and (b - p0[k]) / u[k] > 0])
            if s > 0.1:
                out.append((p0, p0 + s * u))
    return out


def active(u, thr=0.3):  # extinct = max u below thr (Simula's particle mean blurs peaks)
    return u.reshape(len(u), -1).max(1) > thr


def heart(a):
    out, N = f"{a.out}/heart", 128
    pred = load(a.models, "ap")
    scars = [(x, y) for y in (0.3, 0.7) for x in (0.2, 0.5, 0.8)]
    R, Dm, W = a.scar_r, a.scar_D, 2  # windows for the scar rollout and for each ablation
    # 1 one spiral-seeding IC from the sampler (a single broken front, no point stimuli) that re-enters for every scar:
    #   solver active after 2 W windows for all scars, Simula active after W; then the best Simula-solver match
    cand = targets.sample_params("ap", a.n_ic, np.random.default_rng(a.seed))
    for k in ("front1_on", "stim0_on", "stim1_on"):
        cand[:, targets.KNOBS["ap"].index(k)] = 0
    rows = []
    for j, p in enumerate(cand):
        P = params("ap", len(scars), scar_x=[c[0] for c in scars], scar_y=[c[1] for c in scars], scar_radius=R, scar_D=Dm)
        ns = targets.KNOBS["ap"].index("scar_x")
        P[:, :ns] = p[:ns]  # fronts + stimuli from the candidate, scar from the grid
        ic, st = targets.make_ics("ap", P)
        y = solve("ap", ic, st, 2 * W)
        m = chain(pred, ic, st, W, 15)
        ok = int((active(y[:, -1, 0]) & active(m[:, -1, 0])).sum())  # scars re-entering for both
        rows.append((ok, np.mean([relL2(m[i, 1:, 0], y[i, 1 : W * 14 + 1, 0]) for i in range(len(P))]), P, ic, st))
        print(f"IC {j}: re-entry {ok}/{len(P)}, Simula u relL2 over {W} windows {rows[-1][1]:.3f}", flush=True)
    ok, e_ic, P, ic, st = min(rows, key=lambda r: (-r[0], r[1]))  # most re-entering scars, then best match
    # 2 scar rollouts (Simula, q per window) -> the established arrhythmia; 3 candidate ablation lines from it
    m = chain(pred, ic, st, W, a.q_ap)
    start = m[:, -1]
    cands = [lines_from(c, R) for c in scars]
    abl_st = np.stack([np.minimum(st[i, 0], 1 - (1 - Dm) * seg_mask(p0, p1, a.line_half))[None]
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
        fr = chain(pred, start[owner[pick]], abl_st[pick], W, a.q_ap)[:, :, 0]
        save(f"{out}/s{i}/u.npy", m[i, :, 0].astype(np.float16))
        abl = []
        for n, (j, f) in enumerate(zip(pick, fr)):
            lid = "abc"[n]
            p0, p1 = cands[i][int(j - k[0])]
            save(f"{out}/s{i}/{lid}.npy", f.astype(np.float16))
            abl.append({"id": lid, "label": f"Line {lid.upper()}", "line": [[frac(p0[0]), frac(p0[1])], [frac(p1[0]), frac(p1[1])]],
                        "frames": f"s{i}/{lid}.npy", "stops_arrhythmia": bool(sim_stop[j]), "solver_stops": bool(sol_stop[j])})
            shown.append(bool(agree[j]))
        entries.append({"id": f"s{i}", "x": frac(cx), "y": frac(cy), "r": frac(R), "frames": f"s{i}/u.npy", "ablations": abl})
    dt = targets.SPECS["ap"]["frame_dt"] * 14 / (a.q_ap - 1)
    idx = {"version": 1, "mock": False, "note": (
        f"Simula ({a.models}/ap). IC: one sampler front (no stimuli), chosen among {a.n_ic} for re-entry with every scar "
        f"(scars re-entering: {ok}/{len(scars)}, solver after {2 * W} windows and Simula after {W}); Simula u relL2 vs solver over {W} windows {e_ic:.3f}. "
        f"Scars r {R} D {Dm}; {W} chained windows of {a.q_ap} Simula times. Ablation lines: D = {Dm} within {a.line_half} "
        f"of the segment, from the scar rollout's last (Simula) state, {W} windows. stops_arrhythmia = Simula's verdict "
        f"(max u < 0.3 at the end); solver_stops = the solver from the same state and D. Simula and solver agree on "
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
    bs = a.solver_batch
    simb = clock(lambda: [sw(si[i : i + 100], ss[i : i + 100], t14) for i in range(0, a.batch_n, 100)], reps=1)
    sol_n = a.batch_n if a.solver_full else min(a.batch_n, 100)
    solb = clock(lambda: solve("swe", si[:sol_n], ss[:sol_n], batch=bs), reps=1) * a.batch_n / sol_n
    T["batch"] = {"simula_seconds": simb, "solver_seconds": solb, "scenarios": a.batch_n,
                  "solver_measured_scenarios": sol_n, "solver_batch": bs,
                  "what": f"{a.batch_n} random sampler quakes, 15 frames each; Simula in chunks of 100 ICs"
                          + ("" if sol_n == a.batch_n else f"; solver measured on {sol_n} and scaled x{a.batch_n // sol_n}")}
    T["hardware"] = f"{jax.devices()[0].device_kind}; solver = our GPU pseudo-spectral solver (pde.rollout, JAX)"
    T["simula"] = f"{a.models}: particle-mean prediction (8 particles), single precision; wall clock after warm-up"
    json.dump(T, open(f"{a.out}/timing.json", "w"), indent=1)
    print(json.dumps(T), flush=True)


def check():
    m = seg_mask((0.2, 0.5), (0.6, 0.5), 0.02)
    assert m[64, 50] > 0.99 and m[64, 100] < 0.01 and m[40, 50] < 0.01  # on the line, past its end, off to the side
    L = lines_from((0.5, 0.3), 0.12)
    assert len(L) == 16 and all(0.03 - 1e-9 <= v <= 0.97 + 1e-9 for p in L for q in p for v in q)
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
    ap.add_argument("--q_swe", type=int, default=57, help="tsunami: frames per 140 s window (57: every 2.5 s)")
    ap.add_argument("--risk", type=int, default=1000)
    ap.add_argument("--risk_k", type=int, default=10)
    ap.add_argument("--q_ap", type=int, default=57, help="heart: frames per window")
    ap.add_argument("--n_ic", type=int, default=12, help="heart: candidate ICs searched for re-entry")
    ap.add_argument("--scar_r", type=float, default=0.12)
    ap.add_argument("--scar_D", type=float, default=0.03)
    ap.add_argument("--line_half", type=float, default=0.02)
    ap.add_argument("--batch_n", type=int, default=1000)
    ap.add_argument("--solver_batch", type=int, default=100)
    ap.add_argument("--solver_full", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    check() if a.cmd == "check" else {"tsunami": tsunami, "heart": heart, "timing": timing}[a.cmd](a)

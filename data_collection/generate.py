"""Build the training set on the GPU box (resumable):

  python generate.py --out /workspace/pdegen/out

writes out/eq_XXXX/{spec.json, u.npy[, static.npy]} and appends one line per equation to out/index.jsonl.
u.npy is float16 (ic, frame, field, y, x) = (u - shift) / scale with shift/scale in spec["norm"].

Per batch of candidate equations, everything vmapped (one compile per template and batch shape):
  1. probe     geometric-time rollout of 8 ICs -> frame interval at which the state has changed by TARGET of its
               IC amplitude (fronts move ~1.5-2.5 cells per frame at 128^2, like the tsunami/heart demos)
  2. accuracy  dt vs dt/2 over 3 frames must agree to TOL, else halve dt; then validate the full window
               (no blowup, resolved spectrum, not trivial)
  3. pool      POOL ICs simulated for two consecutive 14-step windows (transient + developed dynamics);
               keep the KEEP windows of maximal joint entropy (sample.select)

Supplement (more ICs per equation of an existing dataset, without regenerating it):

  python generate.py --supplement DATASET --extra 12 --out EXTRA --pool 128 --windows 3 [--id-offset I --n N]

rebuilds each equation from its spec.json, re-simulates its identical IC pool with the stored dt / substeps and
re-runs the selection with the stored windows forced first; writes only the EXTRA new windows to
EXTRA/eq_XXXX/{u.npy, static.npy, samples.json} (same layout and normalization).  Equations whose stored picks
repeat (selection poisoned by a blown-up window before the sample.select fix) are reselected from scratch and
get all K + EXTRA windows instead (len(samples.json) tells which).  --check K instead compares the re-simulated
stored windows and the unforced selection with DATASET for K equations per template (needs u.npy).
"""

import argparse
import json
import math
import os
import queue
import re
import threading
import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

import pde
import sample
import template as T

N, F, KEEP = sample.N, 15, 73
TARGET, TOL, SMAX, CAL_B = 0.12, 0.01, 256, 8
GEO = np.array([1, 1, 1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 64, 128])  # probe frames at 1,2,3,4,6,8,12,...,128,256 steps
CUM = np.concatenate([[0], np.cumsum(GEO)])
SIZES = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)  # padded batch sizes: few compiles, <= 1.5x padding


class Batch:
    """Template arrays for equations that share a template (same nf)."""

    def __init__(self, es):
        self.es, self.nf = es, es[0]["nf"]
        self.C = np.stack([e["C"] if "C" in e else T.encode(e, self.nf) for e in es]).astype(np.float32)
        self.prm = np.array([e["prm"] for e in es], np.float32)
        self.L = np.array([e["L"] for e in es], np.float32)
        self.rows = np.stack([sample.ic_arrays(e) for e in es])
        amp = self.rows[:, : self.nf, 1]
        self.cm = (amp != 0).astype(np.float32)
        self.shift = self.rows[:, : self.nf, 0].astype(np.float32)
        self.scale = np.where(amp != 0, np.abs(amp), 1).astype(np.float32)


def roll(b, idx, dt, sched, u0):
    """Simulate equations b[idx] from u0 (len(idx), B, nf+1, N, N).  Padded to a SIZES batch so every jitted
    function sees few shapes; returns ((P, B, F', nf, N, N) float16, padded field mask (P, nf)).  Slice [:len(idx)]."""
    n = len(idx)
    P = next(s for s in SIZES if s >= n)
    pad = lambda x: np.concatenate([x, np.repeat(x[:1], P - n, 0)])
    ii = pad(np.asarray(idx))
    out = T.rollout(jnp.asarray(b.C[ii]), jnp.asarray(b.prm[ii]), jnp.asarray(b.L[ii]),
                    jnp.asarray(pad(np.asarray(dt, np.float32))), jnp.asarray(pad(np.asarray(sched, np.int32))),
                    jnp.concatenate([u0, jnp.repeat(u0[:1], P - n, 0)]), jnp.asarray(b.shift[ii]),
                    jnp.asarray(b.scale[ii]), nf=b.nf, N=N)
    return out, jnp.asarray(b.cm[ii])


@jax.jit
def stats(A, cm):
    """A (E, B, F, nf, N, N) in IC-amplitude units -> healthy (E,B,F), change since frame 0 (E,B,F),
    per-frame change (E,B,F-1), max spatial std over fields (E,B,F), spectral tail of the last frame (E,B)."""
    A = A.astype(jnp.float32)
    fin = jnp.isfinite(A).all((3, 4, 5))
    m = cm[:, None, None, :, None, None]
    A = jnp.where(jnp.isfinite(A), A, 0.0) * m
    nact = cm.sum(1)[:, None, None]
    rms = lambda d: jnp.sqrt((d**2).mean((-2, -1)).sum(-1) / nact)
    healthy = fin & (jnp.abs(A - A.mean((-2, -1), keepdims=True)).max((3, 4, 5)) < 30) & (jnp.abs(A).max((3, 4, 5)) < 1e3)
    last = A[:, :, -1]
    P = jnp.abs(jnp.fft.rfft2(last - last.mean((-2, -1), keepdims=True))) ** 2
    kx, ky = jnp.fft.rfftfreq(N, 1 / N)[None, :], jnp.fft.fftfreq(N, 1 / N)[:, None]
    kk = jnp.sqrt(kx**2 + ky**2)
    tail = (P * ((kk > N / 4) & (kk <= N / 3))).sum((-2, -1)) / ((P * (kk > 0)).sum((-2, -1)) + 1e-12)
    return healthy, rms(A - A[:, :, :1]), rms(A[:, :, 1:] - A[:, :, :-1]), A.std((-2, -1)).max(-1), (tail * cm[:, None]).max(-1)


@jax.jit
def gap(A, B, cm):
    d = (A.astype(jnp.float32) - B.astype(jnp.float32)) * cm[:, None, None, :, None, None]
    return jnp.sqrt((d**2).mean((-2, -1)).sum(-1) / cm.sum(1)[:, None, None])


def calibrate(b, u0):
    """Per equation: accepted?, dt, substeps per frame, rejection reason.  e["target"] = change per frame."""
    E = len(b.es)
    tg = np.array([e.get("target", TARGET) for e in b.es])
    tol = tg * TOL / TARGET
    dt = np.array([e["dt0"] for e in b.es], np.float64)
    S = np.ones(E, int)
    phase = np.zeros(E, int)  # 0 probe, 1 accuracy, 2 accepted, -1 rejected
    why, ups, rounds, coarse = [""] * E, np.zeros(E, int), np.zeros(E, int), np.zeros(E, bool)

    def reject(i, r):
        phase[i], why[i] = -1, r

    for _ in range(16):
        p = np.nonzero(phase == 0)[0]
        if len(p):
            h, ch = (np.asarray(x)[: len(p)] for x in stats(*roll(b, p, dt[p], np.tile(GEO, (len(p), 1)), u0[p]))[:2])
            ok, med = h.all(1), np.median(ch, 1)
            for j, i in enumerate(p):
                rounds[i] += 1
                okf = np.cumprod(ok[j]).astype(bool)
                hit = [f for f in range(1, F) if med[j, f] >= tg[i] and okf[f]]
                if hit and hit[0] == 1 and med[j, 1] > 2 * tg[i]:
                    dt[i] /= 4  # a single step already moves too far
                elif hit:
                    f = hit[0]
                    if f == 1:
                        S[i] = 1
                    else:  # log-log interpolation of the change curve between probe frames
                        c0, c1 = math.log(CUM[f - 1]), math.log(CUM[f])
                        m0, m1 = math.log(max(med[j, f - 1], 1e-9)), math.log(med[j, f])
                        S[i] = max(1, round(math.exp(c0 + (math.log(tg[i]) - m0) / max(m1 - m0, 1e-9) * (c1 - c0))))
                    phase[i] = 1
                elif not okf.all():
                    dt[i] /= 4
                elif ups[i] < 2:
                    dt[i] *= 4
                    ups[i] += 1
                else:
                    reject(i, "slow")
                if phase[i] == 0 and rounds[i] >= 7:
                    reject(i, "probe")
        a = np.nonzero(phase == 1)[0]
        lv = np.floor(np.log2(S[a]))
        for g in [a[lv == k] for k in np.unique(lv)]:  # similar step counts per call: little masked work
            A, cm = roll(b, g, dt[g], np.repeat(S[g, None], F - 1, 1), u0[g])
            half = np.zeros((len(g), F - 1), int)
            half[:, :3] = 2 * S[g, None]
            B, _ = roll(b, g, dt[g] / 2, half, u0[g])
            err = np.asarray(gap(A[:, :, :4], B[:, :, :4], cm))[: len(g)]
            err = np.nan_to_num(np.median(err[:, :, 1:].max(-1), 1), nan=np.inf)
            h, ch, step, sstd, tail = (np.asarray(x)[: len(g)] for x in stats(A, cm))
            for j, i in enumerate(g):
                if err[j] > tol[i]:
                    dt[i], S[i] = dt[i] / 2, S[i] * 2
                    if S[i] > SMAX:
                        reject(i, "stiff")
                elif err[j] < tol[i] / 8 and S[i] >= 2 and not coarse[i]:
                    coarse[i] = True
                    dt[i], S[i] = dt[i] * 2, S[i] // 2
                elif not h[j].all():
                    reject(i, "blowup")
                elif np.median(tail[j]) > 1e-2:
                    reject(i, "unresolved")
                elif np.median(sstd[j, :, -1]) < 0.02 and np.median(step[j, :, -1]) < 2e-3:
                    reject(i, "trivial")
                else:
                    phase[i] = 2
    for i in np.nonzero((phase == 0) | (phase == 1))[0]:
        reject(i, "rounds")
    return phase == 2, dt, S, why


def run_pool(b, idx, dt, S, keys, M, nw, k=None, forced=None):
    """IC pool for equations b[idx] -> selected windows (n, k, F, nf, N, N) f16, static (n, k, N, N), meta.
    forced (n, k) candidate-window indices to pick first, -1 = free (supplement)."""
    n, k = len(idx), k or KEEP
    u0, fam = sample.ic_pool(keys, b.rows[idx], M)
    A, cm = roll(b, idx, dt, np.repeat(np.asarray(S)[:, None], nw * (F - 1), 1), u0)  # nw consecutive windows
    W = jnp.concatenate([A[:, :, w * (F - 1) : w * (F - 1) + F] for w in range(nw)], 1)  # [ic 0..M-1 @ window 0, ...]
    del A
    h, _, step, _, _ = stats(W, cm)
    valid = h.all(-1) & (step.mean(-1) > 1e-3)
    nvalid = np.asarray(valid.sum(1))[:n]
    P = W.shape[0]
    if forced is not None:  # forced windows count as valid; too few valid -> also allow quiet (healthy) windows
        forced = np.concatenate([forced, np.full((P - n, k), -1)]).astype(np.int32)
        fm = np.zeros(valid.shape, bool)
        for r, row in enumerate(forced):
            fm[r, row[row >= 0]] = True
        valid = jnp.where((valid.sum(1) < k)[:, None], h.all(-1), valid) | fm
        forced = jnp.asarray(forced)
    famw = jnp.tile(jnp.concatenate([fam, jnp.repeat(fam[:1], P - n, 0)]), (1, nw))  # family of each candidate window
    late = jnp.tile(jnp.arange(nw * M) >= M, (P, 1))
    sel = sample.select(W, cm, valid, famw, late, forced, k=k)[:n]
    take = jax.vmap(lambda w, s: w[s])
    ics = sel % M
    return np.asarray(take(W[:n], sel)), np.asarray(take(u0[:, :, -1], ics)), np.asarray(sel), np.asarray(take(fam, ics)), nvalid


def spec_of(e, eid, C, dt, S, b_shift, b_scale):
    n, nf = e["n"], e["nf"]
    s = {"id": eid, "family": e["family"], "tags": list(e["tags"]), "pieces": e.get("pieces", []), "fields": e["fields"],
         "static": ["a"] if e["static"] else [], "rhs": T.rhs_strings(e["fields"], C, e["prm"], nf),
         "L": e["L"], "N": N, "order": 2, "dt": float(dt), "substeps": int(S), "frames": F, "frame_dt": float(dt * S),
         "ic": e["ic"], "static_ic": e["static"], "prm": [float(x) for x in e["prm"]],
         "template": {"nf": nf, "coef": [[int(i), int(k), float(C[i, k])] for i, k in zip(*np.nonzero(C))]},
         "norm": {"shift": b_shift[:n].tolist(), "scale": b_scale[:n].tolist()},
         "layout": "u.npy float16 (ic, frame, field, y, x) = (u - shift) / scale; static.npy (ic, y, x) raw"}
    p = e.get("perm", list(range(n)))
    if p != list(range(n)):  # field j of the saved data is internal field p[j]; rename symbols to match
        ren = {e["fields"][p[j]]: T.FIELDS[j] for j in range(n)}
        sub = lambda t: re.sub(r"\b[uvw]\b", lambda m: ren.get(m.group(0), m.group(0)), t)
        s["rhs"] = {ren[f]: sub(s["rhs"][f]) for f in e["fields"]}
        s["ic"] = {ren[f]: s["ic"][f] for f in e["fields"]}
        s["norm"] = {k: [v[i] for i in p] for k, v in s["norm"].items()}
        s["fields"] = list(T.FIELDS[:n])
    s["perm"], s["target_change"] = [int(i) for i in p], e.get("target", TARGET)
    s["latex"] = pde.latex(s)
    s["graph"] = pde.graph(s)
    return s


def save(out, eid, files):
    """out/eid/{name: array (.npy) | json object}, written to out/.tmp_eid and renamed (atomic)."""
    tmp = os.path.join(out, f".tmp_{eid}")
    os.makedirs(tmp, exist_ok=True)
    for name, x in files.items():
        if name.endswith(".npy"):
            np.save(os.path.join(tmp, name), x)
        else:
            with open(os.path.join(tmp, name), "w") as f:
                json.dump(x, f)
    os.rename(tmp, os.path.join(out, eid))


def writer(out, q):
    while (job := q.get()) is not None:
        eid, spec, u, static = job
        save(out, eid, {"u.npy": u} | ({} if static is None else {"static.npy": static}) | {"spec.json": spec})
        with open(os.path.join(out, "index.jsonl"), "a") as f:
            f.write(json.dumps({k: spec[k] for k in ("id", "family", "fields", "static", "rhs", "latex", "frame_dt", "L")}) + "\n")
        q.task_done()
    q.task_done()


def internal(s):
    """spec.json -> the sampler's internal equation (field order before the saved permutation), enough for Batch."""
    n, nf, p = len(s["fields"]), s["template"]["nf"], s["perm"]
    C = np.zeros((nf, len(T.LIB[nf])))
    for i, k, c in s["template"]["coef"]:
        C[i, k] = c
    fs = list(T.FIELDS[:n])
    ic = {fs[p[j]]: s["ic"][T.FIELDS[j]] for j in range(n)}  # saved field j = internal field p[j]
    return {"nf": nf, "n": n, "C": C, "prm": s["prm"], "L": s["L"], "fields": fs, "ic": ic, "static": s["static_ic"]}


def supplement(args):
    ids = [i for i in range(args.id_offset, args.id_offset + args.n) if os.path.exists(f"{args.supplement}/eq_{i:04d}")]
    todo = ids if args.check else [i for i in ids if not os.path.exists(f"{args.out}/eq_{i:04d}")]
    specs = [json.load(open(f"{args.supplement}/eq_{i:04d}/spec.json")) for i in todo]
    for s in specs:  # dataset A (pre-v2) has no field permutation
        s.setdefault("perm", list(range(len(s["fields"]))))
    k0 = {len(s["samples"]) for s in specs}
    assert len(k0) <= 1, k0
    k0 = k0.pop() if k0 else 0
    log, t0, done, worst, skipped, reselected = open(os.path.join(args.out, "log.jsonl"), "a"), time.time(), 0, [], [], 0
    for nf in (2, 3):
        sub = sorted((s for s in specs if s["template"]["nf"] == nf), key=lambda s: s["substeps"])
        if args.check:  # K equations with distinct stored picks, 2 with repeated ones
            rep = lambda q: q[:: max(1, len(q) // args.check)][: args.check]
            dist = lambda s: len({(x["ic"], x["start_frame"]) for x in s["samples"]}) == k0
            sub = rep([s for s in sub if dist(s)]) + rep([s for s in sub if not dist(s)])[:2]
        per = max(1, (1024 if nf == 2 else 512) // args.pool)
        for c in range(0, len(sub), per):
            g = sub[c : c + per]
            b = Batch([internal(s) for s in g])
            keys = jnp.stack([jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s["candidate"]["seed"]), s["candidate"]["index"]), 1)
                              for s in g])
            old = np.array([[x["start_frame"] // (F - 1) * args.pool + x["ic"] for x in s["samples"]] for s in g])
            keep = np.array([len(set(o)) == k0 for o in old])  # repeated picks: reselect everything
            forced = np.concatenate([np.where(keep[:, None], old, -1), np.full((len(g), args.extra), -1)], 1)
            dt, S = np.array([s["dt"] for s in g]), np.array([s["substeps"] for s in g])
            x, static, sel, fam, nvalid = run_pool(b, np.arange(len(g)), dt, S, keys, args.pool, args.windows, k0 + args.extra, forced)
            if args.check:  # stored windows reproduce, and so does the original (unforced) selection
                _, _, sel0, _, _ = run_pool(b, np.arange(len(g)), dt, S, keys, args.pool, args.windows, k0)
                for j, s in enumerate(g):
                    fresh = {"distinct": len(set(sel[j])) == len(sel[j]), "finite": bool(np.isfinite(x[j]).all())}
                    if not keep[j]:
                        print(json.dumps({"id": s["id"], "nf": nf, "family": s["family"], "reselect": fresh, "nvalid": int(nvalid[j])}))
                        continue
                    u = np.load(f"{args.supplement}/{s['id']}/u.npy").astype(np.float32)
                    err = np.abs(x[j][:k0][:, :, s["perm"]].astype(np.float32) - u)
                    rec = {"id": s["id"], "nf": nf, "family": s["family"], "max_err": float(err.max()),
                           "mean_err": float(err.mean()), "max_abs_u": float(np.abs(u).max()),
                           "selection_reproduced": bool((sel0[j] == old[j]).all()), "nvalid": int(nvalid[j]), "extra": fresh}
                    worst.append(rec)
                    print(json.dumps(rec), flush=True)
                continue
            for j, s in enumerate(g):
                a = k0 if keep[j] else 0  # first window to write
                if len(set(sel[j])) < len(sel[j]) or not np.isfinite(x[j][a:]).all():
                    skipped.append(s["id"])  # too few healthy windows for k0 + extra picks
                    continue
                files = {"u.npy": x[j][a:][:, :, s["perm"]]}
                if s["static"]:
                    files["static.npy"] = static[j][a:].astype(np.float16)
                files["samples.json"] = [{"ic": int(i % args.pool), "start_frame": int(i // args.pool) * (F - 1),
                                          "ic_family": sample.FAM_NAMES[int(f)]} for i, f in zip(sel[j][a:], fam[j][a:])]
                save(args.out, s["id"], files)
                done += 1
            reselected += int((~keep).sum())
            rec = {"nf": nf, "written": done, "todo": len(todo), "reselected": reselected, "skipped": skipped, "min_nvalid": int(nvalid.min()), "S_max": int(S.max()),
                   "elapsed_h": round((time.time() - t0) / 3600, 3),
                   "eta_h": round((len(todo) - done) * (time.time() - t0) / done / 3600, 2)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    if args.check:
        print("check max_err", max(r["max_err"] for r in worst), "selection reproduced",
              sum(r["selection_reproduced"] for r in worst), "/", len(worst), flush=True)
    else:
        print("done", sum(os.path.exists(f"{args.out}/eq_{i:04d}") for i in ids), flush=True)


def main():
    jax.config.update("jax_compilation_cache_dir", "/workspace/jaxcache")  # the IC-pool compile alone is ~4 min
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/pdegen/out")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--pool", type=int, default=128, help="ICs per equation; candidate windows = 2 * pool")
    ap.add_argument("--cand", type=int, default=256, help="candidate equations per calibration batch")
    ap.add_argument("--seed", type=int, default=0, help="different seeds give disjoint equation sets")
    ap.add_argument("--keep", type=int, default=KEEP, help="selected IC windows per equation")
    ap.add_argument("--windows", type=int, default=2, help="consecutive 14-step windows simulated per IC")
    ap.add_argument("--target", type=float, nargs=2, default=[TARGET, TARGET],
                    help="log-uniform range of the per-frame change (sets each equation's frame interval)")
    ap.add_argument("--permute", type=float, default=0.0, help="probability of shuffling field labels")
    ap.add_argument("--id-offset", type=int, default=0, help="first equation id; lets several GPUs share --out")
    ap.add_argument("--supplement", help="existing dataset dir: add --extra windows per equation, see top")
    ap.add_argument("--extra", type=int, default=12)
    ap.add_argument("--check", type=int, default=0, help="with --supplement: verify reproduction on this many eqs per nf")
    ap.add_argument("--ic-scale", type=float, nargs=2, default=sample.IC_SCALE, help="IC amplitude multiplier U*a+b; A: 0.9 0.3")
    args = ap.parse_args()
    globals()["KEEP"] = args.keep
    sample.IC_SCALE = tuple(args.ic_scale)
    os.makedirs(args.out, exist_ok=True)
    if args.supplement:
        return supplement(args)
    state_f = os.path.join(args.out, f"state_{args.id_offset}.json" if args.id_offset else "state.json")
    st = json.load(open(state_f)) if os.path.exists(state_f) else {"next_cand": 0}
    ids = {int(d[3:]) for d in os.listdir(args.out) if d.startswith("eq_")}
    written = len([i for i in ids if args.id_offset <= i < args.id_offset + args.n])
    q = queue.Queue(maxsize=8)
    th = threading.Thread(target=writer, args=(args.out, q), daemon=True)
    th.start()
    root = jax.random.PRNGKey(args.seed)
    t0, n0 = time.time(), written
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    while written < args.n:
        c0 = st["next_cand"]
        es = [sample.equation(np.random.default_rng([args.seed, c])) for c in range(c0, c0 + args.cand)]
        for e, c in zip(es, range(c0, c0 + args.cand)):
            rc = np.random.default_rng([args.seed, c, 1])
            e["cand"], e["target"] = c, float(np.exp(rc.uniform(*np.log(args.target))))
            e["perm"] = list(rc.permutation(e["n"]) if rc.random() < args.permute else range(e["n"]))
        st["next_cand"] = c0 + args.cand
        for nf in (2, 3):
            if written >= args.n:
                break
            sub = [e for e in es if e["nf"] == nf]
            if not sub:
                continue
            b = Batch(sub)
            ekeys = jnp.stack([jax.random.fold_in(root, e["cand"]) for e in sub])
            t1 = time.time()
            u0, _ = sample.ic_pool(jax.vmap(lambda k: jax.random.fold_in(k, 0))(ekeys), b.rows, CAL_B)
            acc, dt, S, why = calibrate(b, u0)
            t2 = time.time()
            idx = np.nonzero(acc)[0][: args.n - written]
            idx = idx[np.argsort(S[idx], kind="stable")]
            per = max(1, (1024 if nf == 2 else 512) // args.pool)
            for k in range(0, len(idx), per):
                g = idx[k : k + per]
                pk = jax.vmap(lambda kk: jax.random.fold_in(kk, 1))(ekeys[g])
                x, static, sel, fam, nvalid = run_pool(b, g, dt[g], S[g], pk, args.pool, args.windows)
                for j, i in enumerate(g):
                    e = sub[i]
                    if nvalid[j] < KEEP:
                        why[i] = "pool"
                        continue
                    eid = f"eq_{args.id_offset + written:04d}"
                    spec = spec_of(e, eid, b.C[i], dt[i], S[i], b.shift[i], b.scale[i])
                    spec["samples"] = [{"ic": int(s % args.pool), "start_frame": int(s // args.pool) * (F - 1),
                                        "ic_family": sample.FAM_NAMES[int(f)]} for s, f in zip(sel[j], fam[j])]
                    spec["candidate"] = {"seed": args.seed, "index": e["cand"]}
                    q.put((eid, spec, x[j][:, :, e["perm"]], static[j].astype(np.float16) if e["static"] else None))
                    written += 1
            t3 = time.time()
            rate = (written - n0) / max(t3 - t0, 1e-9)
            rec = {"cand": c0, "nf": nf, "n": len(sub), "accepted": int(acc.sum()), "written": written,
                   "reasons": dict(Counter(w for w in why if w)), "S_median": float(np.median(S[acc])) if acc.any() else None,
                   "S_p90": float(np.percentile(S[acc], 90)) if acc.any() else None, "t_calib": round(t2 - t1, 1),
                   "t_pool": round(t3 - t2, 1), "elapsed_h": round((t3 - t0) / 3600, 3),
                   "eta_h": round((args.n - written) / rate / 3600, 2) if rate > 0 else None}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        json.dump(st, open(state_f, "w"))
    q.put(None)
    q.join()
    print("done", written, flush=True)


if __name__ == "__main__":
    main()

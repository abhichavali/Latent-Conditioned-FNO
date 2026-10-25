"""Phase 4 variant: target-aware batch acquisition for the few-shot loop of active.py.  Everything but the choice of the
B sims per round is active.run: same pools (seed (seed, r)), particles, latent fit, stage B / C branch, test sets,
budget and history.jsonl schema (+ an "acq" entry), written next to it as <results>/active/<target>_<method>/.

  python active_v2.py run --ckpt CK --target swe|ap --method active_v2|generic_v2|epig_only|logdet_only [--stage b|c]
  python active_v2.py export --ckpt CK --target swe        # final model -> <results>/final_models/<target> (run does it)
  python active_v2.py summary --results /workspace/results/<run>_s<step>     # all methods, incl. active.py's
  python active_v2.py check                                                   # self-check of the scoring math (CPU)

Ensemble linear-Gaussian model.  E members (particles z on the pretrained weights, or (bootstrap adapter, particle)
pairs) predict p_e(x) at the scoring frames in physical units, field f of IC x divided by its own predicted magnitude
(rms of the ensemble mean; --norm abs: a pool-wide scale_f as in active.score), so a candidate's information and the
reference variance are relative, like the relL2 fit loss and the raw_* test metric.  Write
f(x) = mean(x) + B(x) w, B = deviations / sqrt(E), w ~ N(0, I_E), so Cov(f(x), f(x')) is the ensemble covariance.
Observing x adds precision G_x / s2 on w, where G_x = B(x)^T B(x) averaged over the output dims (fields, frames, space;
E x E, so no projection is needed), tr G_x = the per-dim particle variance (active.score / dims), and
s2 = noise_frac * median_pool tr G.  With Lam the current precision (I, then + G_pick / s2 per fantasized pick):
  epig(x)   = tr(Lam^-1 M_R) - tr((Lam + G_x / s2)^-1 M_R),  M_R = mean of G_x* over a fixed reference set R of fresh
              targets.sample ICs (not the test set): the expected drop of the predictive variance on R.  For a scalar
              output and Lam = I it is sum_{x* in R} Cov(f(x), f(x*))^2 / (Var f(x) + s2) / |R| (EPIG).
  logdet(x) = log det(Lam + G_x / s2) - log det Lam: greedy max log det(I + K_S / s2), the objective of
              data_collection/sample.py:_greedy_logdet with K = the ensemble covariance (Sylvester), in weight space.
Methods (fixes over active.py):
  active_v2    round 1: scrambled Sobol design over the knobs (before any data the prior's disagreement is mostly
               noise); later rounds: greedy fantasized EPIG on an ensemble of J bootstrap adapters (online Poisson(1)
               bootstrap over the acquired sims, warm-started across rounds, adapter j trained with particles z[j::J])
               x K_score / J of its particles.  The adapters only score; the reported update is stage B / C as in active.py.
  generic_v2   active_v2 with the generic prior N(mean, cov of the training latents) (active.py "generic")
  epig_only    EPIG score + k-means batch (active.choose), particle ensemble on the pretrained weights
  logdet_only  greedy log det batch (variance-type score), particle ensemble on the pretrained weights
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import active as al  # noqa: E402
import eval as ev  # noqa: E402
import targets  # noqa: E402
from data.canon import F_MAX  # noqa: E402

METHODS = {"active_v2": ("epig", "greedy", "adapters", "sobol"),  # score, batch, ensemble, round 1
           "epig_only": ("epig", "kmeans", "z", "score"),
           "logdet_only": ("logdet", "greedy", "z", "score")}
METHODS["generic_v2"] = METHODS["active_v2"]


# ---------------------------------------------------------------- ensemble information per candidate
@eqx.filter_jit
def _preds(fno, b, Z, t):
    """Chunk b of c ICs, Z (K, dz), t (Q,) -> normalized predictions (K, c, Q, F, N, N), one particle at a time."""
    return jax.lax.map(lambda z: jax.vmap(lambda bb: ev._fwd(fno, bb, z, t))(b), Z)


@jax.jit
def _gram(P, s, m, pres):
    """P (E, c, Q, F, N, N) normalized -> per field, averaged over (Q, x, y): Gram of the physical deviations from the
    ensemble mean / E (c, F, E, E), mean square of the physical ensemble mean (c, F); absent field slots zeroed."""
    P = P * s[None, :, None, :, None, None] + m[None, :, None, :, None, None]
    mean = P.mean(0)
    A = P - mean
    G = jnp.einsum("ecqfxy,gcqfxy->cfeg", A, A) / (A.shape[0] * A.shape[2] * A.shape[4] * A.shape[5])
    return G * pres[:, :, None, None], (mean**2).mean((1, 3, 4)) * pres


def grams(members, inp, t):
    """members [(fno, Z (K_j, dz))], inputs (ev.prep) -> G (n, F, E, E), msq (n, F), in chunks like ev.predict."""
    n, t = len(inp["u0"]), jnp.asarray(t, jnp.float32)
    c, G, M = max(1, 128 // len(t)), [], []
    for i in range(0, n, c):
        j = np.arange(i, min(i + c, n))
        jj = np.pad(j, (0, c - len(j)), mode="edge")
        b = {k: jnp.asarray(inp[k][jj]) for k in ev.KEYS}
        P = jnp.concatenate([_preds(f, b, jnp.asarray(Z), t) for f, Z in members])
        g, ms = _gram(P, jnp.asarray(inp["s"][jj]), jnp.asarray(inp["m"][jj]), jnp.asarray(inp["pres"][jj, :F_MAX], jnp.float32))
        G.append(np.asarray(g, np.float64)[: len(j)]), M.append(np.asarray(ms, np.float64)[: len(j)])
    return np.concatenate(G), np.concatenate(M)


# ---------------------------------------------------------------- batch selection in the ensemble's weight space
def gain(Jc, MR, Lam, crit):
    """Jc (n, E, E) candidate precisions G_x / s2, MR (E, E), current precision Lam -> score per candidate."""
    if crit == "epig":
        return np.trace(np.linalg.solve(Lam, MR)) - np.einsum("nij,ji->n", np.linalg.inv(Lam + Jc), MR)
    return np.linalg.slogdet(Lam + Jc)[1] - np.linalg.slogdet(Lam)[1]


def acquire(Gc, MR, B, crit, batch, feat, rng, noise_frac):
    """-> picks, score of each pick when picked, first-pick scores (n,), s2.  greedy: fantasize each pick (Lam += J_pick)
    and re-score; kmeans: active.choose on the first-pick scores."""
    s2 = noise_frac * float(np.median(np.trace(Gc, axis1=1, axis2=2))) + 1e-30
    Jc, Lam, pick, at = Gc / s2, np.eye(Gc.shape[1]), [], []
    first = gain(Jc, MR, Lam, crit)
    if batch == "kmeans":
        pick = al.choose(first, feat, B, rng)
        return pick, [float(first[k]) for k in pick], first, s2
    sc = first.copy()
    for _ in range(B):
        sc[pick] = -np.inf
        k = int(np.argmax(sc))
        pick.append(k), at.append(float(sc[k]))
        Lam = Lam + Jc[k]
        sc = gain(Jc, MR, Lam, crit)
    return pick, at, first, s2


def sobol_pool(p, B, seed):
    """Round-1 design: the first B points of a scrambled Sobol sequence over the knobs, mapped like
    targets.sample_params (on-flags: u < ON_P, sub_side: u < 0.5, ranges linear), as a pool of active.Target."""
    from scipy.stats import qmc

    R = targets.RANGES[p.base]
    U = qmc.Sobol(len(R), scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(B))))[:B]
    P = np.empty(U.shape, np.float32)
    for k, (kn, lo, hi) in enumerate(R):
        P[:, k] = (U[:, k] < targets.ON_P[kn]) if kn in targets.ON_P else (U[:, k] < 0.5) if kn.endswith("sub_side") else lo + (hi - lo) * U[:, k]
    ics, st = targets.make_ics(p.base, P)
    nf, ns = ics.shape[1], st.shape[1]
    return {"feat": targets.features(p.base, P), "ics": ics * p.fu[:nf], "st": st * p.su[:ns], "id": P}


def get_prior(a, enc, Z, p):
    """Graph prior p_phi(z | G) with sigma -> tau sigma (temperature calibration, eval.py's tau), or for generic_v2 the
    generic prior N(mean, cov of the training latents) as active.py "generic" (not tempered)."""
    if a.method == "generic_v2":
        return ev.generic_prior(Z)
    g = ev.graph_prior(enc, p.graph)
    return {"mu": g["mu"], "L": (a.tau * g["L"]).astype(np.float32)}


def updater(a, fno, prior, out):
    """The reported fine-tuning branch of active.run: stage B (active.make_ft, a.ft_full) or C (active.make_stage_c).
    -> (state, update(state, data, z, rng) -> (state, ft record), model(state) -> (fno, z for testing))."""
    if a.stage == "c":
        c_init, c_static, c_fit = al.make_stage_c(fno, a.c_lr)
        cp = c_init(prior["mu"])
        cp = eqx.tree_deserialise_leaves(f"{out}/ft.eqx", cp) if os.path.exists(f"{out}/ft.eqx") else cp

        def update(cp, data, z, rng):
            cp, info = c_fit(cp, data, rng, a.c_pull, 5, a.c_steps)
            return cp, {"train_rel": float(info["val"]), "stage": "c", "info": info}

        return cp, update, lambda cp: (eqx.combine(cp[0], c_static), np.asarray(cp[1])[None])
    w0, frozen, opt, step = al.make_ft(fno, a.ft_lr, a.ft_pull, full=a.ft_full)
    w = eqx.tree_deserialise_leaves(f"{out}/ft.eqx", w0) if os.path.exists(f"{out}/ft.eqx") else w0
    box = {"ost": opt.init(w)}  # optimizer state lives across rounds (re-initialised on resume, as in active.run)

    def update(w, data, z, rng):
        (d, nd), ls = data, []
        for _ in range(a.ft_steps):
            si, qi = rng.integers(0, nd, a.ft_batch), rng.integers(0, d["y"].shape[1], a.ft_batch)
            w, box["ost"], l = step(w, box["ost"], w0, frozen, d, si, qi, z[rng.integers(0, len(z), a.ft_batch)])
            ls.append(l)
        return w, {"train_rel": float(np.mean(jax.device_get(ls[-20:]))), "stage": "b"}

    return w, update, lambda w: (eqx.combine(w, frozen), None)


# ---------------------------------------------------------------- the loop (active.run with another acquisition)
def run(a):
    crit, batch, ens, init = METHODS[a.method]
    cfg, S, (fno, enc, Z), path = ev.load(a.run, a.ckpt)
    rname = f"{os.path.basename(a.run.rstrip('/'))}_s{int(os.path.basename(path)[5:13])}"
    out = a.out or f"/workspace/results/{rname}/active/{a.target.replace('/', '_').replace(':', '-')}_{a.method}"
    os.makedirs(out, exist_ok=True)
    log = lambda *x: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)
    p = al.problem(a.target, cfg, S, np.random.default_rng(a.seed + 100))
    assert isinstance(p, al.Target), "active_v2 samples knobs: swe / ap / swe_m only"
    prior = get_prior(a, enc, Z, p)
    fno = ev.adapt_statics(fno, p.test.items[0][0].ns)
    fit = ev.make_fitter(fno, lr=a.lr)
    ftw, ft_update, ft_model = updater(a, fno, prior, out) if a.ft_steps else (None, None, None)
    J = a.J if ens == "adapters" else 0
    if J:  # scoring ensemble: J adapters (spectral banks frozen), whatever the reported update is
        b0, bfrozen, b_opt, b_step = al.make_ft(fno, a.ft_lr, a.ft_pull, full=False)
        ws = list(eqx.tree_deserialise_leaves(f"{out}/boot.eqx", [b0] * J)) if os.path.exists(f"{out}/boot.eqx") else [b0] * J
        b_ost = [b_opt.init(x) for x in ws]
    rng = np.random.default_rng([a.seed, 7])
    hist, sf = f"{out}/history.jsonl", f"{out}/state.npz"
    tq = np.linspace(0, p.frames - 1, a.score_frames + 1)[1:].round() / (p.frames - 1)
    ref = p.pool(np.random.default_rng([a.seed, 99]), a.ref, set())  # fixed reference set R (fresh typical ICs)
    inp_R = ev.prep(ref["ics"], ref["st"], p.T, p.L)
    del ref
    if os.path.exists(sf):
        st = dict(np.load(sf, allow_pickle=True))
        eps, eps0, r0 = st["eps"], st["eps0"], int(st["round"]) + 1
        acq = {k: list(st[k]) for k in ("ics", "st", "traj", "id")}
        counts = st["boot"] if "boot" in st else np.zeros((0, J))
        rng = np.random.default_rng([a.seed, 7, r0])
        log(f"resumed at round {r0} ({len(acq['traj'])} sims)")
    else:
        eps0 = np.random.default_rng(a.seed).standard_normal((a.K, len(prior["mu"]))).astype(np.float32)
        eps, r0, counts = eps0, 1, np.zeros((0, J))
        acq = {"ics": [], "st": [], "traj": [], "id": []}
        t0 = time.time()
        e0 = al.test_error(fno, p, ev.draw(prior, eps))
        emu = al.test_error(fno, p, prior["mu"][None])
        rec = {"round": 0, "n_sims": 0, "test": e0, "test_prior_mean": emu, "time_s": time.time() - t0,
               "ckpt": path, "target": a.target, "method": a.method, "args": vars(a), "prior_sigma_mean": float(np.diag(prior["L"]).mean())}
        open(hist, "w").write(json.dumps(rec) + "\n")
        log("round 0", ev.brief({"post": e0, "mu": emu}))
    for r in range(r0, a.rounds + 1):
        t0 = time.time()
        z = ev.draw(prior, eps)
        rb = np.random.default_rng([a.seed, 11, r])  # bootstrap stream (keeps rng as in active.run)
        if r == 1 and init == "sobol":
            pool, sc, info = sobol_pool(p, a.batch, a.seed), None, {"design": "sobol"}
            pick = list(range(len(pool["ics"])))
        else:
            pool = p.pool(np.random.default_rng([a.seed, r]), a.pool, set())
            members = ([(eqx.combine(ws[j], bfrozen), z[j::J][: a.K_score // J]) for j in range(J)] if J and r > 1
                       else [(fno, z[: a.K_score])])
            inp = ev.prep(pool["ics"], pool["st"], p.T, p.L)
            Gp, msp = grams(members, inp, tq)
            del inp
            Gr, msr = grams(members, inp_R, tq)
            if a.norm == "abs":  # active.score's scale: per field over the pool
                sp = sr = np.broadcast_to(msp.mean(0), msp.shape[1:])
            else:  # relative to each IC's own predicted magnitude, like the relL2 loss and the raw_* test metric
                sp, sr = np.maximum(msp, 1e-3 * msp.mean(0)), np.maximum(msr, 1e-3 * msp.mean(0))
            sp, sr = (np.where(x > 0, x, 1.0)[..., None, None] for x in (sp, sr))
            Gc, MR = (Gp / sp).sum(1), (Gr / sr).sum(1).mean(0)
            pick, at, sc, s2 = acquire(Gc, MR, min(a.batch, len(Gc)), crit, batch, pool["feat"], rng, a.noise_frac)
            var = np.trace(Gc, axis1=1, axis2=2)
            info = {"members": int(Gc.shape[1]), "ensemble": "adapters" if J and r > 1 else "z", "noise_s2": s2,
                    "picked_var": [float(var[k]) for k in pick], "pool_var_median": float(np.median(var)),
                    "picked_var_pct": [round(float((var < var[k]).mean()), 3) for k in pick], "ref_var": float(np.trace(MR)),
                    "picked_epig_frac": None if crit != "epig" else [round(x / float(np.trace(MR)), 4) for x in at]}
        n = len(pool["ics"])
        t_score = time.time() - t0
        traj = p.simulate(pool, np.array(pick))
        ok = np.isfinite(traj).all((1, 2, 3, 4))
        for i, k in enumerate(pick):
            if ok[i]:
                acq["ics"].append(pool["ics"][k]), acq["st"].append(None if pool["st"] is None else pool["st"][k])
                acq["traj"].append(traj[i]), acq["id"].append(pool["id"][k])
        t_sim = time.time() - t0 - t_score
        stt = None if acq["st"][0] is None else np.stack(acq["st"])
        trj = np.stack(acq["traj"])
        data = ev.fit_data(ev.prep(np.stack(acq["ics"]), stt, p.T, p.L), trj)
        npairs = trj.shape[0] * (trj.shape[1] - 1)
        eps, dl = fit(eps, prior, data, a.steps0 if r == 1 else a.steps, a.M, rng, float(npairs), 1.0, eps0)
        t_fit = time.time() - t0 - t_score - t_sim
        z = ev.draw(prior, eps)
        e = al.test_error(fno, p, z)
        if a.ft_steps:
            t1 = time.time()
            ftw, ft = ft_update(ftw, data, z, rng)
            m, zt = ft_model(ftw)
            ft = {"test": al.test_error(m, p, z[:8] if zt is None else zt), **ft}
            eqx.tree_serialise_leaves(f"{out}/ft.tmp.eqx", ftw)
            os.replace(f"{out}/ft.tmp.eqx", f"{out}/ft.eqx")
            ft["time_s"] = time.time() - t1
        t_test = time.time() - t0 - t_score - t_sim - t_fit
        if J:  # bootstrap adapters for the next round's scoring
            (d, nd), fresh = data, len(counts) == 0
            counts = np.concatenate([counts, rb.poisson(1.0, (nd - len(counts), J))])
            for j in range(J):
                c = counts[:, j]
                pj, zj = (c / c.sum() if c.sum() else None), z[j::J]
                for _ in range(a.boot_steps0 if fresh else a.boot_steps):
                    si, qi = rb.choice(nd, a.boot_batch, p=pj), rb.integers(0, d["y"].shape[1], a.boot_batch)
                    ws[j], b_ost[j], _ = b_step(ws[j], b_ost[j], b0, bfrozen, d, si, qi, zj[rb.integers(0, len(zj), a.boot_batch)])
            eqx.tree_serialise_leaves(f"{out}/boot.tmp.eqx", ws)
            os.replace(f"{out}/boot.tmp.eqx", f"{out}/boot.eqx")
        del data
        rec = {"round": r, "n_sims": len(acq["traj"]), "test": e, "fit_relL2": float(dl.mean()), "nonfinite_sims": int((~ok).sum()),
               "eps_spread": float(eps.std(0).mean()), "eps_norm": float(np.linalg.norm(eps, axis=1).mean()),
               "picked_scores": None if sc is None else at, "pool_score_median": None if sc is None else float(np.median(sc)),
               "pool": n, "acq": info,
               "time": {"score": t_score, "sim": t_sim, "fit": t_fit, "test": t_test, "boot": time.time() - t0 - t_score - t_sim - t_fit - t_test}}
        if a.ft_steps:
            rec["ft"] = ft
        st = {"eps": eps, "eps0": eps0, "round": r, "traj": trj, "ics": np.stack(acq["ics"]), "id": np.stack(acq["id"]),
              "st": np.array(acq["st"], dtype=object) if stt is None else stt, "boot": counts}
        np.savez(sf + ".tmp.npz", **st)
        os.replace(sf + ".tmp.npz", sf)
        open(hist, "a").write(json.dumps(rec) + "\n")
        log(f"round {r} sims {rec['n_sims']}", ev.brief({"post": e}), f"fit {rec['fit_relL2']:.4f}",
            {k: round(v, 1) for k, v in rec["time"].items()}, "| ft" if a.ft_steps else "",
            (ev.brief({"ft": ft["test"]}), round(ft["train_rel"], 4)) if a.ft_steps else "",
            {k: info[k] for k in ("picked_var_pct", "picked_epig_frac") if k in info})
    if a.method == "active_v2" and a.ft_steps and a.export:
        export(a)
    open(f"{out}/DONE", "w").write("")


# ---------------------------------------------------------------- final adapted model for demos
def export(a):
    """Final fine-tuned model of a run -> <results>/<run>_s<step>/final_models/<target>/: fno.eqx (full model), z.npy
    (the particles it is tested with), acquired_params.npy (knobs of the acquired sims), meta.json (model config, units,
    n_sims, the run's final errors); then checks that load_adapted reproduces the run's final "ft" test error."""
    cfg, S, (fno, enc, Z), path = ev.load(a.run, a.ckpt)
    rname = f"{os.path.basename(a.run.rstrip('/'))}_s{int(os.path.basename(path)[5:13])}"
    out = a.out or f"/workspace/results/{rname}/active/{a.target}_{a.method}"
    dest = a.dest or f"/workspace/results/{rname}/final_models/{a.target}"
    p = al.Target(a.target)
    prior = get_prior(a, enc, Z, p)
    st = np.load(f"{out}/state.npz", allow_pickle=True)
    ftw, _, ft_model = updater(a, ev.adapt_statics(fno, p.test.items[0][0].ns), prior, out)
    model, zt = ft_model(ftw)
    H = [json.loads(x) for x in open(f"{out}/history.jsonl")]
    spec = json.load(open(f"{ev.TARGETS}/{a.target}/spec.json"))
    os.makedirs(dest, exist_ok=True)
    eqx.tree_serialise_leaves(f"{dest}/fno.eqx", model)
    np.save(f"{dest}/z.npy", ev.draw(prior, st["eps"])[:8] if zt is None else zt)
    np.save(f"{dest}/acquired_params.npy", st["id"])
    meta = {"target": a.target, "method": a.method, "stage": a.stage, "ckpt": path, "run_dir": out, "model": cfg["model"],
            "T": p.T, "L": p.L, "frames": p.frames, "fields": spec["fields"], "static": spec["static"], "knobs": spec["knobs"],
            "n_sims": int(len(st["traj"])), "rounds": int(st["round"]), "test_ft": H[-1]["ft"]["test"], "test_latent": H[-1]["test"],
            "units": "swe: x km, t ks, eta / b m, u / v m/s, sponge 1/ks (targets.SWE_MIX); others as in targets.py",
            "usage": "pred = active_v2.load_adapted(dir); pred(ics, statics, times) with ics, statics as targets.make_ics "
                     "gives them, times = frame / (frames - 1) in [0, 1] -> particle-mean prediction (n, len(times), nf, N, N)"}
    json.dump(meta, open(f"{dest}/meta.json", "w"), indent=1)
    meta["check"] = check_loaded(dest, a.target)
    json.dump(meta, open(f"{dest}/meta.json", "w"), indent=1)
    d = abs(meta["check"]["raw_slot0"] - meta["test_ft"]["raw_slot0"])
    print(f"exported {dest}: n_sims {meta['n_sims']}, loader raw_slot0 {meta['check']['raw_slot0']:.4f} vs run "
          f"{meta['test_ft']['raw_slot0']:.4f}", flush=True)
    assert d < 1e-3 * meta["test_ft"]["raw_slot0"], d


def load_adapted(d):
    """final_models/<target> dir -> predict(ics, statics, times): ICs (n, nf, N, N) and statics (n, ns, N, N) in the
    target's units (as targets.make_ics gives them), times in [0, 1] (frame / (frames - 1)) -> the particle-mean
    prediction (n, len(times), nf, N, N) in physical units.  predict.meta = meta.json."""
    from models.fno import CondFNO

    m = json.load(open(f"{d}/meta.json"))
    fno = eqx.tree_deserialise_leaves(f"{d}/fno.eqx", CondFNO(jax.random.PRNGKey(0), **m["model"]))
    z, nf = np.load(f"{d}/z.npy"), len(m["fields"])

    def predict(ics, statics, times, c=None):  # c: ICs per device call (eval.predict's default: ~64 forwards)
        inp = ev.prep(np.asarray(ics, np.float32), None if statics is None else np.asarray(statics, np.float32), m["T"], m["L"])
        return np.concatenate([mean[:, :, :nf] for _, mean, _ in ev.predict(fno, inp, z, np.asarray(times, np.float32), c)])

    predict.meta = m
    return predict


def check_loaded(d, target):
    """load_adapted on the fixed test set (raw ICs and statics, frames 1..F-1) -> eval.summarize of its error."""
    pred, s = load_adapted(d), ev.target_set(target)[0]
    E, pres = [], []
    for i in range(0, len(s.items), 25):
        idx = np.arange(i, min(i + 25, len(s.items)))
        inp, y, pr = s.load(idx)
        it = [s.items[k] for k in idx]
        ics = np.stack([e.frames_raw(w, np.array([0]))[0] for e, w, _ in it])
        st = np.stack([e.statics(w) for e, w, _ in it]) if it[0][0].ns else None
        P = np.zeros_like(y)
        P[:, :, : ics.shape[1]] = pred(ics, st, s.t)
        E.append(ev.errors(P, y, inp)), pres.append(pr)
    return ev.summarize(ev.concat(E), np.concatenate(pres))


# ---------------------------------------------------------------- summary: all methods vs scratch
def summary(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key, methods = a.metric, ["active", "random", "generic", *METHODS]
    colors = dict(zip(methods, ["#1b7f6b", "#7aa6d6", "#e0a030", "#c0392b", "#8e44ad", "#d35400", "#6d4c41"]))
    tg = [t for t in a.targets if any(os.path.exists(f"{a.results}/active/{t}_{m}/history.jsonl") for m in methods)]
    fig, axes = plt.subplots(1, len(tg), figsize=(5 * len(tg), 4), squeeze=False)
    table = {}
    for ax, t in zip(axes[0], tg):
        sp = f"{a.scratch}/{t}.json"
        sc = al.scratch_curve(sp, key) if os.path.exists(sp) else None
        refs = {int(N): float(v) for N, v in zip(*sc)} if sc is not None else {}
        target = float(sc[1][np.argmin(np.abs(sc[0] - a.target_n))]) if refs else None
        row = {"target_error": target, "scratch_curve": refs}
        if refs:
            ax.plot(sc[0], sc[1], "o-", color="#555555", label="FNO from scratch", ms=3)
        for m in methods:
            h = f"{a.results}/active/{t}_{m}/history.jsonl"
            if not os.path.exists(h):
                continue
            row[m] = {}
            for br, ls in (("ft", "o-"), ("latent", "s--")):
                n, e = al.curve(h, key, br)
                if br == "ft":
                    ax.plot(np.maximum(n, 1), e, ls, color=colors[m], label=f"{m} (fine-tuned)", ms=3)
                row[m][br] = {"final": float(e[-1]), "n_final": int(n[-1]), "mean_after_0": float(e[1:].mean()) if len(e) > 1 else None,
                              "sims_to_target": al.sims_to(n, e, target) if target is not None else None,
                              "sims_to_scratch_at": {N: al.sims_to(n, e, v) for N, v in refs.items()},
                              "curve": [[int(x), float(y)] for x, y in zip(n, e)]}
        table[t] = row
        ax.set(xscale="log", yscale="log", xlabel="target simulations (0 plotted at 1)", ylabel=f"test {key}", title=t)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{a.results}/headline_v2_{key}.png", dpi=150)
    json.dump(table, open(f"{a.results}/sims_to_target_v2_{key}.json", "w"), indent=1)
    for t, row in table.items():  # compact text table: fine-tuned error at matched sims, sims to scratch errors
        print(f"\n{t}  target {row['target_error']}")
        for m in methods:
            if m in row:
                for br in ("ft", "latent"):
                    c = dict(row[m][br]["curve"])
                    print(f"  {m:12s} {br:6s}", " ".join(f"{N}:{c[N]:.3f}" for N in (0, 5, 10, 25, 50, 75, 100) if N in c),
                          "| sims to scratch@", {N: None if v is None else round(v, 1) for N, v in row[m][br]["sims_to_scratch_at"].items()})


# ---------------------------------------------------------------- self-check of the weight-space formulas
def check():
    r = np.random.default_rng(0)
    E, n, nR, s2 = 6, 40, 30, 0.3
    b, bR = r.standard_normal((n, E)), r.standard_normal((nR, E))  # scalar outputs: B(x) is a row
    Gc, MR = b[:, :, None] * b[:, None], (bR[:, :, None] * bR[:, None]).mean(0)
    ref = ((b @ bR.T) ** 2).sum(1) / ((b**2).sum(1) + s2) / nR  # EPIG: sum Cov^2 / (Var + s2) / |R|
    assert np.allclose(gain(Gc / s2, MR, np.eye(E), "epig"), ref)
    pick, at, _, s2b = acquire(Gc, MR, 4, "logdet", "greedy", None, r, 0.5)
    K = b[pick] @ b[pick].T  # greedy log det in weight space = log det(I + K_S / s2) over the picked set (Sylvester)
    assert np.isclose(sum(at), np.linalg.slogdet(np.eye(4) + K / s2b)[1]) and len(set(pick)) == 4
    pick, at, first, _ = acquire(Gc, MR, 4, "epig", "greedy", None, r, 0.5)
    assert len(set(pick)) == 4 and pick[0] == int(np.argmax(first)), (pick, at)
    P = sobol_pool(type("P", (), {"base": "ap", "fu": np.ones((3, 1, 1)), "su": np.ones((2, 1, 1))})(), 5, 0)["id"]
    assert P.shape == (5, len(targets.RANGES["ap"])) and (P[:, targets.KNOBS["ap"].index("front0_on")] == 1).all()
    print("active_v2 check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary", "check", "export"])
    ap.add_argument("--run", default="/workspace/runs/pretrain_A")
    ap.add_argument("--ckpt")
    ap.add_argument("--target", default="swe")
    ap.add_argument("--method", default="active_v2", choices=list(METHODS))
    ap.add_argument("--out")
    ap.add_argument("--dest", help="export: final model dir (default <results>/<run>_s<step>/final_models/<target>)")
    ap.add_argument("--export", type=int, default=1, help="run: export the final model of an active_v2 run")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--K_score", type=int, default=8, help="ensemble members for scoring (adapters: K_score / J each)")
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--ref", type=int, default=500, help="reference set R (fresh ICs) for EPIG")
    ap.add_argument("--norm", default="rel", choices=["rel", "abs"],
                    help="ensemble deviations relative to each IC's predicted magnitude (rel) or a pool-wide field scale (abs: "
                         "picks sat at the 99.9th variance percentile on AP)")
    ap.add_argument("--noise_frac", type=float, default=1.0, help="s2 = noise_frac * median candidate variance (0.1 saturated: pick 1 took 86% of the reference variance)")
    ap.add_argument("--J", type=int, default=4, help="active_v2: bootstrap adapters")
    ap.add_argument("--boot_steps0", type=int, default=300, help="Adam steps per adapter the first time")
    ap.add_argument("--boot_steps", type=int, default=100, help="Adam steps per adapter per later round")
    ap.add_argument("--boot_batch", type=int, default=8)
    ap.add_argument("--score_frames", type=int, default=3)
    ap.add_argument("--steps0", type=int, default=200)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    # reported fine-tuning branch: defaults = the A-checkpoint runs of active.py (adapter stage B); finals pass theirs
    ap.add_argument("--stage", default="b", choices=["b", "c"])
    ap.add_argument("--ft_steps", type=int, default=300)
    ap.add_argument("--ft_batch", type=int, default=16)
    ap.add_argument("--ft_lr", type=float, default=2e-4)
    ap.add_argument("--ft_pull", type=float, default=1e-3)
    ap.add_argument("--ft_full", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--c_lr", type=float, default=2e-4)
    ap.add_argument("--c_pull", type=float, default=1e-3)
    ap.add_argument("--c_steps", type=int, default=2000)
    ap.add_argument("--tau", type=float, default=1.0, help="graph prior temperature: sigma -> tau sigma")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", help="summary: results dir of one checkpoint")
    ap.add_argument("--targets", nargs="*", default=["swe", "ap"])
    ap.add_argument("--scratch", default="/workspace/results/scratch")
    ap.add_argument("--metric", default="raw_slot0")
    ap.add_argument("--target_n", type=int, default=512)
    a = ap.parse_args()
    {"run": run, "summary": summary, "check": check, "export": export}[a.cmd](*([] if a.cmd == "check" else [a]))


if __name__ == "__main__":
    main()

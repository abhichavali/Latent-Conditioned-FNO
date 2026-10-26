"""v3 few-shot on the targets (swe, ap, fire) and k-shot on rehearsal equations, for a train_ctx.py checkpoint.

  python fewshot_ctx.py run --run /workspace/runs/ctx --ckpt CK --target swe|ap|fire --method active|v1random
  python fewshot_ctx.py rehearsal --run /workspace/runs/ctx --ckpt CK
  python fewshot_ctx.py summary --results /workspace/results/ctx_s<step>

run: 20 rounds x 5 sims.  Round r: candidate pool exactly as active.py (Target.pool with seed (seed, r), 2000), then
  active    S samples z ~ q(z | G, sims so far) (round 1: the zero-shot q(z | G)); score = active.score (per-field
            variance at score_frames times / scale^2) on the whole pool; active.choose (k-means, 5 clusters, top
            of each); the picks, best score first, are simulated (targets.simulate) and join the context
  v1random  the 5 next sims of v1's A u B random run (its state.npz, in acquisition order): same sims as v1's curve
Branches (test raw_slot0 etc. on the fixed 100-IC test set), at N in NS:
  a  amortized: posterior mean of q(z | G, all N sims) in one forward pass (N = 100: 400 sim tokens; training saw
     at most k_max = 64)
  b  semi-amortized: (a), then z refined on the N sims (eval.make_fitter: whitened eps around q, |eps|^2 anchor,
     data weight = (sim, frame) pairs, as v1's particles; one particle)
  c  (b), then active.make_ft (v1 stage B: all weights, lr 5e-4, 300 steps per round warm-started, pull 1e-4) with
     z = z_b fixed.  The chain runs every round (5, 10, ..., 100); N = 1, 2 are side branches from the pretrained
     weights (300 steps each), so N >= 5 follows v1's schedule exactly.
Acquisition only ever uses the amortized posterior.  state.npz + history.jsonl every round; a rerun resumes.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NS = (0, 1, 2, 5, 10, 25, 50, 100)
EVAL_ROUNDS = (1, 2, 5, 10, 20)
V1 = "/workspace/results/pretrain_AB_s200000"


def load(run, ckpt=None):
    import equinox as eqx
    import jax

    import train
    import train_ctx

    cfg = yaml.safe_load(open(f"{run}/config.yaml"))
    params = train_ctx.build(cfg, jax.random.PRNGKey(0))
    path = ckpt or train.latest(run)[0]
    opt = train_ctx.optimizer(cfg)
    params, _ = eqx.tree_deserialise_leaves(path, (params, opt.init(eqx.filter(params, eqx.is_array))))
    return cfg, params, path


def rname(run, path):
    return f"{os.path.basename(run.rstrip('/'))}_s{int(os.path.basename(path)[5:13])}"


class Post:
    """q(z | G, sims) for one equation; context features are added as sims arrive."""

    def __init__(self, enc, graph, T, L, pres, K):
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        from data import canon

        self.enc, self.K, self.pres = enc, K, jnp.asarray(pres)
        self.g = jax.tree_util.tree_map(lambda a: jnp.asarray(a[0]), canon.pad([graph]))
        self.lt, self.ll = jnp.float32(np.log(T)), jnp.float32(np.log(L))
        self.f = []
        self.fn = eqx.filter_jit(lambda e, g, cx, cst, ca, pr, lt, ll, m: e(g, cx, cst, ca, pr, lt, ll, m[None]))

    def add(self, traj, sts=None):
        from data.canon import F_MAX, S_MAX
        from models.context import CTX_FRAMES, ctx_features

        for i in range(len(traj)):
            f = ctx_features(traj[i][list(CTX_FRAMES)], None if sts is None else sts[i])
            x = np.zeros((len(CTX_FRAMES), F_MAX, *f["x"].shape[-2:]), np.float16)
            x[:, : f["x"].shape[1]] = f["x"]
            self.f.append((x, f["st"], f["aux"]))

    def add_cached(self, X, ST, AUX, ws):
        from data.canon import F_MAX, S_MAX

        for w in ws:
            x = np.zeros((X.shape[1], F_MAX, *X.shape[-2:]), np.float16)
            st = np.zeros((S_MAX, *X.shape[-2:]), np.float16)
            x[:, : X.shape[2]], st[: ST.shape[1]] = X[w], ST[w]
            self.f.append((x, st, AUX[w]))

    def one(self, idx, S):
        import jax.numpy as jnp

        f = [self.f[i] for i in idx]
        n = len(f)
        pad = lambda j, shape, dt: np.stack([v[j] for v in f] + [np.zeros(shape, dt)] * (S - n)) if n else np.zeros((S, *shape), dt)
        x0, s0, a0 = (self.f[0] if self.f else (np.zeros((7, 3, 32, 32), np.float16), np.zeros((2, 32, 32), np.float16), np.zeros(8, np.float32)))
        cx, cst, ca = pad(0, x0.shape, np.float16), pad(1, s0.shape, np.float16), pad(2, a0.shape, np.float32)
        mu, ls = self.fn(self.enc, self.g, jnp.asarray(cx), jnp.asarray(cst), jnp.asarray(ca), self.pres, self.lt, self.ll,
                         jnp.asarray(np.arange(S) < n))
        return np.asarray(mu[0]), np.exp(np.asarray(ls[0]))

    def q(self, n, rng=None):
        """Posterior over all of the first n sims, one pass (slots padded to a multiple of 16: few recompiles)."""
        return self.one(range(n), max(16, int(np.ceil(n / 16)) * 16))


def run(a):
    import equinox as eqx
    import jax.numpy as jnp

    import active
    import eval as ev
    from models.context import adapt_statics_ctx

    cfg, (fno, enc), path = load(a.run, a.ckpt)
    out = a.out or f"/workspace/results/{rname(a.run, path)}/active/{a.target}_{a.method}"
    os.makedirs(out, exist_ok=True)
    log = lambda *x: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)
    if os.path.exists(f"{out}/DONE"):  # finished (ft.eqx removed): nothing to resume
        return log(f"{out} is done")
    p = active.Target(a.target)
    ns = p.test.items[0][0].ns
    fno, enc = ev.adapt_statics(fno, ns), adapt_statics_ctx(enc, ns)
    inp0 = p.test.load([0])[0]
    post = Post(enc, p.graph, p.T, p.L, inp0["pres"][0], cfg["data"]["k_max"])
    fit = ev.make_fitter(fno, lr=a.b_lr)
    w0, frozen, ft_opt, ft_step = active.make_ft(fno, a.ft_lr, a.ft_pull)
    tq = np.linspace(0, p.frames - 1, a.score_frames + 1)[1:].round() / (p.frames - 1)
    test = lambda f, z: active.test_error(f, p, np.asarray(z, np.float32)[None])
    hist, sf = f"{out}/history.jsonl", f"{out}/state.npz"
    v1 = np.load(f"{V1}/active/{a.target}_random/state.npz", allow_pickle=True) if a.method == "v1random" else None

    def data_of(n):
        stt = None if acq["st"][0] is None else np.stack(acq["st"][:n])
        return ev.fit_data(ev.prep(np.stack(acq["ics"][:n]), stt, p.T, p.L), np.stack(acq["traj"][:n]))

    def refine(mu, sd, data, n, rng):
        prior = {"mu": mu.astype(np.float32), "L": np.diag(sd).astype(np.float32)}
        eps, dl = fit(np.zeros((1, len(mu)), np.float32), prior, data, a.b_steps, a.b_M, rng, float(n * (p.frames - 1)), 1.0)
        return ev.draw(prior, eps)[0], float(dl.mean())

    def finetune(w, ost, data, z, rng):
        d, n = data
        Z = jnp.asarray(np.repeat(z[None], a.ft_batch, 0))
        ls = []
        for _ in range(a.ft_steps):
            si, qi = rng.integers(0, n, a.ft_batch), rng.integers(0, d["y"].shape[1], a.ft_batch)
            w, ost, l = ft_step(w, ost, w0, frozen, d, si, qi, Z)
            ls.append(l)
        return w, ost, float(np.mean(np.asarray(ls[-20:])))

    def branches(n, rng, w=None, ost=None):
        """(a), (b) on the first n sims; (c) continues the chain (w, ost) or, if w is None, a side branch from w0."""
        t0 = time.time()
        mu, sd = post.q(n, rng)
        rec = {"n_sims": n, "a": test(fno, mu), "sigma_mean": float(sd.mean())}
        data = data_of(n)
        zb, rec["b_fit_relL2"] = refine(mu, sd, data, n, rng)
        rec["b"], rec["b_shift_sigma"] = test(fno, zb), float(np.sqrt((((zb - mu) / sd) ** 2).mean()))
        if not a.skip_c:
            w, ost = (w0, ft_opt.init(w0)) if w is None else (w, ost)
            w, ost, rec["c_train_rel"] = finetune(w, ost, data, zb, rng)
            rec["c"] = test(eqx.combine(w, frozen), zb)
        rec["time_s"] = time.time() - t0
        return rec, w, ost, zb, data

    if os.path.exists(sf):
        st = dict(np.load(sf, allow_pickle=True))
        r0 = int(st["round"]) + 1
        acq = {k: list(st[k]) for k in ("ics", "st", "traj", "id")}
        post.add(np.stack(acq["traj"]), None if acq["st"][0] is None else np.stack(acq["st"]))
        w = w0 if a.skip_c else eqx.tree_deserialise_leaves(f"{out}/ft.eqx", w0)
        ost = None if a.skip_c else ft_opt.init(w)
        log(f"resumed at round {r0}, {len(acq['traj'])} sims")
    else:
        r0, acq, w = 1, {"ics": [], "st": [], "traj": [], "id": []}, w0
        ost = None if a.skip_c else ft_opt.init(w0)
        mu, sd = post.q(0, None)
        e0 = test(fno, mu)
        rec = {"round": 0, "n_sims": 0, "a": e0, "b": e0, **({} if a.skip_c else {"c": e0}), "sigma_mean": float(sd.mean()), "ckpt": path,
               "target": a.target, "method": a.method, "args": vars(a)}
        open(hist, "w").write(json.dumps(rec) + "\n")
        log("round 0", ev.brief({"zero_shot": e0}))
    for r in range(r0, a.rounds + 1):
        t0 = time.time()
        rng = np.random.default_rng([a.seed, 7, r])
        n0 = len(acq["traj"])
        mu, sd = post.q(n0, rng)
        sc = None
        if a.method == "v1random":
            sl = slice(a.batch * (r - 1), a.batch * r)
            new = {"ics": v1["ics"][sl], "st": list(v1["st"][sl]) if v1["st"].dtype != object else [None] * len(v1["ics"][sl]),
                   "traj": v1["traj"][sl], "id": v1["id"][sl]}
            if not len(new["traj"]):
                break
        else:
            pool = p.pool(np.random.default_rng([a.seed, r]), a.pool, set())
            Z = (mu + sd * rng.standard_normal((a.S, len(mu)))).astype(np.float32)
            inp = ev.prep(pool["ics"], pool["st"], p.T, p.L)
            sc = active.score(fno, inp, Z, tq)
            del inp
            pick = sorted(active.choose(sc, pool["feat"], a.batch, rng), key=lambda i: -sc[i])
            traj = p.simulate(pool, np.array(pick))
            ok = np.isfinite(traj).all((1, 2, 3, 4))
            pick = [k for k, o in zip(pick, ok) if o]
            new = {"ics": pool["ics"][pick], "st": [None if pool["st"] is None else pool["st"][k] for k in pick],
                   "traj": traj[ok], "id": pool["id"][pick]}
        t_acq = time.time() - t0
        for k in acq:
            acq[k] += list(new[k])
        post.add(np.asarray(new["traj"]), None if acq["st"][0] is None else np.stack(new["st"]))
        N = len(acq["traj"])
        recs = []
        if r == 1:  # N = 1, 2: side branches from the pretrained weights
            for n in (1, 2):
                if n < N:
                    rec, *_ = branches(n, rng)
                    recs.append({"round": 1, "sub": True, **rec})
        if r in EVAL_ROUNDS or r == a.rounds:
            rec, w, ost, zb, _ = branches(N, rng, w, ost)
        elif a.skip_c:
            rec = {"n_sims": N}
        else:  # chain only: (b) for z, then the fine-tuning round
            mu, sd = post.q(N, rng)
            data = data_of(N)
            zb, fr = refine(mu, sd, data, N, rng)
            w, ost, ctr = finetune(w, ost, data, zb, rng)
            rec = {"n_sims": N, "b_fit_relL2": fr, "c_train_rel": ctr, "sigma_mean": float(sd.mean())}
        rec.update(round=r, t_acq=t_acq, time_round=time.time() - t0,
                   picked_scores=None if sc is None else [float(sc[k]) for k in pick],
                   pool_score_median=None if sc is None else float(np.median(sc)))
        recs.append(rec)
        if not a.skip_c:
            eqx.tree_serialise_leaves(f"{out}/ft.tmp.eqx", w)
            os.replace(f"{out}/ft.tmp.eqx", f"{out}/ft.eqx")
        st = {"round": r, "traj": np.stack(acq["traj"]), "ics": np.stack(acq["ics"]), "id": np.stack(acq["id"]),
              "st": np.array(acq["st"], dtype=object) if acq["st"][0] is None else np.stack(acq["st"])}
        np.savez(sf + ".tmp.npz", **st)
        os.replace(sf + ".tmp.npz", sf)
        with open(hist, "a") as f:
            for x in recs:
                f.write(json.dumps(x) + "\n")
        for x in recs:
            log(f"round {r} N {x['n_sims']}", ev.brief({k: x[k] for k in ("a", "a_all", "b", "c") if k in x}),
                f"{x['time_s'] if 'time_s' in x else x['time_round']:.0f}s")
    if os.path.exists(f"{out}/ft.eqx"):
        os.remove(f"{out}/ft.eqx")  # only needed to resume; 2 GB each
    open(f"{out}/DONE", "w").write("")


def rehearsal(a):
    """k-shot on held-out-structure equations: per equation, test windows = up to 20 windows of ~30% of its IC
    groups; context = the first k windows of a fixed random order of the other groups' windows."""
    import jax

    import eval as ev
    from data import adapter, canon, ctx

    cfg, (fno, enc), path = load(a.run, a.ckpt)
    S = json.load(open(f"{a.run}/splits.json"))
    roots = {r["tag"]: r["path"] for r in cfg["data"]["roots"]}
    rng = np.random.default_rng(0)
    gids = [S["rehearsal"][i] for i in sorted(rng.choice(len(S["rehearsal"]), min(a.n_eq, len(S["rehearsal"])), replace=False))]
    eqs = [adapter.Eq(g.split("/")[0], f"{roots[g.split('/')[0]]}/{g.split('/')[1]}") for g in gids]
    for tag in roots:
        adapter.drop_nonfinite([e for e in eqs if e.gid.startswith(tag + "/")], f"{cfg['cache_dir']}/nonfinite_{tag}.json")
    cache = ctx.load_cache(eqs, cfg["ctx_cache"])
    ks = (0, 1, 2, 5, 10, 16, 32, 64)  # k > available context windows: all of them (n_ctx in the json)
    items, Z, n_ctx = [], {k: [] for k in ks}, []
    for e in eqs:
        ug = np.unique(e.groups)
        tg = set(rng.choice(ug, max(1, int(round(0.3 * len(ug)))), replace=False).tolist())
        tw = [int(w) for w, g in zip(e.windows, e.groups) if g in tg][:20]
        cw = rng.permutation([int(w) for w, g in zip(e.windows, e.groups) if g not in tg])
        pres = np.zeros(5, np.float32)
        pres[: e.nf], pres[3 : 3 + e.ns] = 1, 1
        post = Post(enc, canon.graph(e.spec), e.T, e.L, pres, cfg["data"]["k_max"])
        post.add_cached(*cache[e.gid], cw[: max(ks)])
        for k in ks:
            mu, _ = post.q(min(k, len(cw)), rng)
            Z[k] += [mu] * len(tw)
        items += [(e, w, -1) for w in tw]
        n_ctx.append(len(cw))
    s = ev.Set(items)
    modes = {f"k{k}": np.asarray(Z[k], np.float32)[None] for k in ks}
    modes["persistence"] = "persistence"
    res = ev.evaluate(fno, s, modes)
    out = a.out or f"/workspace/results/{rname(a.run, path)}"
    os.makedirs(out, exist_ok=True)
    json.dump({"n_eq": len(eqs), "n_windows": len(items), "n_ctx": n_ctx, "ckpt": path, "results": res}, open(f"{out}/rehearsal_kshot.json", "w"), indent=1)
    print(json.dumps(ev.brief(res)), flush=True)


def summary(a):
    """deck_raw_slot0.json (v1 deck schema) + headline plot: ctx branches next to v1 (A u B finals) and scratch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from active import sims_to

    key = a.metric
    v1 = json.load(open(a.v1_deck))
    deck = {"metric": key, "metric_note": v1["metric_note"] + "; ctx = v3 context model", "targets": {}}
    names = {"c": "", "b": "_semi", "a": "_amortized", "a_all": "_amortized_all"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, t in zip(axes, ("swe", "ap", "fire")):
        v = v1["targets"].get(t, {})
        dt = {"scratch": v.get("scratch", []), "methods": {}, "fewer": {}}
        for m in ("random", "active"):
            if m in v.get("methods", {}):
                dt["methods"][f"v1_{m}"] = v["methods"][m]
        for h in sorted(glob.glob(f"{a.results}/active/{t}_*/history.jsonl")):
            method = os.path.basename(os.path.dirname(h))[len(t) + 1 :]
            H = [json.loads(l) for l in open(h)]
            for br, suf in names.items():
                pts = sorted({x["n_sims"]: x[br][key] for x in H if br in x}.items())
                if pts:
                    dt["methods"][f"ctx_{method}{suf}"] = [{"sims": n, "error": e} for n, e in pts]
        sc = {int(x["sims"]): x["error"] for x in dt["scratch"]}
        for name, pts in dt["methods"].items():
            n, e = np.array([x["sims"] for x in pts], float), np.array([x["error"] for x in pts])
            dt["fewer"][name] = {str(N): {"scratch_error": se, "sims": s, "x_fewer": N / s if s else None}
                                 for N, se in sc.items() for s in [sims_to(n, e, se)] if s is not None}
        deck["targets"][t] = dt
        if sc:
            ax.plot(list(sc), list(sc.values()), "o-", color="#555555", label="FNO from scratch", ms=3)
        for name, pts in dt["methods"].items():
            n = np.maximum([x["sims"] for x in pts], 0.5)
            ax.plot(n, [x["error"] for x in pts], "o-" if name.startswith("ctx") else "s--", ms=3,
                    lw=1.6 if name.startswith("ctx") else 1.0, label=name)
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_title(t), ax.set_xlabel("target simulations (0 plotted at 0.5)")
        ax.set_ylabel(f"test {key}"), ax.spines[["top", "right"]].set_visible(False), ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{a.results}/headline_{key}.png", dpi=150)
    json.dump(deck, open(f"{a.results}/deck_{key}.json", "w"), indent=1)
    print(json.dumps({t: {m: [round(x["error"], 3) for x in v] for m, v in d["methods"].items()} for t, d in deck["targets"].items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "rehearsal", "summary"])
    ap.add_argument("--run", default="/workspace/runs/ctx")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--target", default="swe")
    ap.add_argument("--method", default="active", choices=["active", "v1random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--S", type=int, default=8, help="posterior samples for scoring")
    ap.add_argument("--score_frames", type=int, default=3)
    ap.add_argument("--b_steps", type=int, default=200)
    ap.add_argument("--b_M", type=int, default=16)
    ap.add_argument("--b_lr", type=float, default=0.05)
    ap.add_argument("--ft_steps", type=int, default=300)
    ap.add_argument("--skip_c", action="store_true", help="branches (a), (b) only (no 515M fine-tune)")
    ap.add_argument("--ft_batch", type=int, default=16)
    ap.add_argument("--ft_lr", type=float, default=5e-4)
    ap.add_argument("--ft_pull", type=float, default=1e-4)
    ap.add_argument("--n_eq", type=int, default=60, help="rehearsal equations")
    ap.add_argument("--results")
    ap.add_argument("--metric", default="raw_slot0")
    ap.add_argument("--v1_deck", default=f"{V1}/deck_raw_slot0.json")
    a = ap.parse_args()
    {"run": run, "rehearsal": rehearsal, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()

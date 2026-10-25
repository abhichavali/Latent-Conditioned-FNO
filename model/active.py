"""Phase 4: few-shot active learning on a new equation, theta and phi frozen: K particles over the latent z.

  python active.py run --run RUN --target swe|ap|swe_m|reh:A/eq_0123 --method active|random|generic [--rounds 20]
  python active.py summary --results /workspace/results/<run>      # headline plot + sims-to-target-error table

Round r (pool seed (seed, r), so all methods see the same pools):
  1 fresh pool of candidate ICs: targets.sample (rehearsal: the equation's unused windows)
  2 score s_i = sum_{f, t, x} Var_k[pred_i] / scale_f^2 with K_score particles at a few frame times
    (scale_f^2 = mean square of the particle-mean prediction of field f over the pool)
  3 k-means into B clusters on the knob features (rehearsal: low Fourier modes of the IC); the top-scoring candidate of
    each cluster is simulated with the real solver (rehearsal: its stored window)
  4 every particle is refit, warm-started (round 1: from its prior sample), with Adam on whitened eps (z = mu + L eps):
        min  sum_{sims, frames} relL2(G(.; z_k), sim) + |eps_k|^2       (relL2: the training loss, normalized space)
  5 test error of the posterior-mean prediction (mean over particles) on the fixed test set
Methods: active (graph prior p_phi(z | G)), random (graph prior, random candidates), generic (N(mean, cov of the
training z_i), active).  State is saved every round (state.npz, history.jsonl); a rerun resumes.
SWE is in km (targets.SWE_KM); swe_m is the same target in metres (diagnostic: the IC normalization's absolute floors
are unit-dependent).
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval as ev  # noqa: E402
import targets  # noqa: E402
from data import adapter  # noqa: E402
from data.canon import F_MAX  # noqa: E402

TARGETS = ev.TARGETS


class Target:
    """A target equation: knob sampler + real solver.  swe_m = SWE in metres (fields, b times KM)."""

    def __init__(self, name):
        self.name, self.base = name, "swe" if name == "swe_m" else name
        spec = json.load(open(f"{TARGETS}/{name}/spec.json"))
        self.graph, self.T, self.L, self.frames = spec["canonical"], spec["T"], spec["L"], spec["frames"]
        self.test = ev.target_set(name)[0]
        u = targets.KM if name == "swe_m" else 1.0
        self.fu, self.su = np.array([u, u, u], np.float32)[:, None, None], np.array([u, 1.0], np.float32)[:, None, None]

    def pool(self, rng, n, used):
        P = targets.sample_params(self.base, n, rng)
        ics, st = targets.make_ics(self.base, P)
        nf, ns = ics.shape[1], st.shape[1]
        return {"feat": targets.features(self.base, P), "ics": ics * self.fu[:nf], "st": st * self.su[:ns], "id": P}

    def simulate(self, pool, idx):
        ics, st = pool["ics"][idx], pool["st"][idx]
        nf, ns = ics.shape[1], st.shape[1]
        return targets.simulate(self.base, ics / self.fu[:nf], st / self.su[:ns]) * self.fu[:nf]


class Rehearsal:
    """A held-out-structure equation of the pretraining data: "simulate" = look up a stored window.  Test set: the
    windows of ~30% of its IC groups (at most 100); candidates: the windows of the other groups, not yet acquired."""

    def __init__(self, e, rng):
        self.e, self.graph, self.T, self.L, self.frames = e, None, e.T, e.L, e.frames
        g = np.unique(e.groups)
        tg = set(rng.choice(g, max(1, int(round(0.3 * len(g)))), replace=False).tolist())
        tw = [int(w) for w, gg in zip(e.windows, e.groups) if gg in tg][:100]
        self.cand = [int(w) for w, gg in zip(e.windows, e.groups) if gg not in tg]
        self.test = ev.Set([(e, w, -1) for w in tw])

    def pool(self, rng, n, used):
        w = np.array([x for x in self.cand if x not in used])
        w = w[rng.permutation(len(w))[:n]]
        ics = np.stack([self.e.frames_raw(x, np.array([0]))[0] for x in w])
        st = np.stack([self.e.statics(x) for x in w]) if self.e.ns else None
        f = np.log1p(np.abs(np.fft.rfft2(ics))[:, :, :4, :4]).reshape(len(w), -1)  # low Fourier modes of the IC
        return {"feat": (f - f.mean(0)) / (f.std(0) + 1e-6), "ics": ics, "st": st, "id": w}

    def simulate(self, pool, idx):
        return np.stack([self.e.frames_raw(int(x), np.arange(self.e.frames)) for x in pool["id"][idx]])


def problem(target, cfg, S, rng):
    if not target.startswith("reh:"):
        return Target(target)
    gid = target[4:]
    assert gid in S["rehearsal"], f"{gid} is not a rehearsal equation of this run"
    tag = gid.split("/")[0]
    root = [r for r in cfg["data"]["roots"] if r["tag"] == tag][0]
    e = adapter.Eq(tag, f"{root['path']}/{gid.split('/')[1]}")
    adapter.drop_nonfinite([e], f"{cfg['cache_dir']}/nonfinite_{tag}.json")
    p = Rehearsal(e, rng)
    from data import canon

    p.graph = canon.graph(e.spec)
    return p


def score(fno, inp, z, t):
    """Particle disagreement per candidate: sum_{f, t, x} Var_k / scale_f^2, scale_f^2 = mean square of the particle mean."""
    vs, ms = [], np.zeros(F_MAX)
    for j, mean, var in ev.predict(fno, inp, z, t):
        vs.append(var.sum((1, 3, 4)))
        ms += (mean**2).mean((1, 3, 4)).sum(0)
    ms /= len(inp["u0"])
    return (np.concatenate(vs) / np.where(ms > 0, ms, 1.0)).sum(1)


def choose(scores, feat, B, rng):
    """Top-scoring candidate of each of B k-means clusters of the features (filled by the global ranking)."""
    from scipy.cluster.vq import kmeans2

    _, lab = kmeans2(feat.astype(np.float64), B, minit="++", rng=np.random.default_rng(int(rng.integers(1 << 31))))
    pick = [int(np.flatnonzero(lab == c)[np.argmax(scores[lab == c])]) for c in np.unique(lab)]
    for i in np.argsort(-scores):
        if len(pick) >= B:
            break
        if int(i) not in pick:
            pick.append(int(i))
    return pick


def test_error(fno, p, z):
    n = len(p.test.items)
    return ev.evaluate(fno, p.test, {"post": np.broadcast_to(z[:, None], (len(z), n, z.shape[1]))})["post"]


def run(a):
    cfg, S, (fno, enc, Z), path = ev.load(a.run, a.ckpt)
    rname = f"{os.path.basename(a.run.rstrip('/'))}_s{int(os.path.basename(path)[5:13])}"
    out = a.out or f"/workspace/results/{rname}/active/{a.target.replace('/', '_').replace(':', '-')}_{a.method}"
    os.makedirs(out, exist_ok=True)
    log = lambda *x: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)
    p = problem(a.target, cfg, S, np.random.default_rng(a.seed + 100))
    prior = ev.generic_prior(Z) if a.method == "generic" else ev.graph_prior(enc, p.graph)
    fit = ev.make_fitter(fno, lr=a.lr)
    rng = np.random.default_rng([a.seed, 7])
    hist = f"{out}/history.jsonl"
    sf = f"{out}/state.npz"
    tq = np.linspace(0, p.frames - 1, a.score_frames + 1)[1:].round() / (p.frames - 1)  # scoring times
    if os.path.exists(sf):
        st = dict(np.load(sf, allow_pickle=True))
        eps, r0 = st["eps"], int(st["round"]) + 1
        acq = {k: list(st[k]) for k in ("ics", "st", "traj", "id")}
        rng = np.random.default_rng([a.seed, 7, r0])
        log(f"resumed at round {r0} ({len(acq['traj'])} sims)")
    else:
        eps, r0 = np.random.default_rng(a.seed).standard_normal((a.K, len(prior["mu"]))).astype(np.float32), 1
        acq = {"ics": [], "st": [], "traj": [], "id": []}
        t0 = time.time()
        e0 = test_error(fno, p, ev.draw(prior, eps))
        emu = test_error(fno, p, prior["mu"][None])
        rec = {"round": 0, "n_sims": 0, "test": e0, "test_prior_mean": emu, "time_s": time.time() - t0,
               "ckpt": path, "target": a.target, "method": a.method, "args": vars(a), "prior_sigma_mean": float(np.diag(prior["L"]).mean())}
        open(hist, "w").write(json.dumps(rec) + "\n")
        log("round 0", ev.brief({"post": e0, "mu": emu}))
    for r in range(r0, a.rounds + 1):
        t0 = time.time()
        pool = p.pool(np.random.default_rng([a.seed, r]), a.pool, set(int(x) for x in acq["id"]) if isinstance(p, Rehearsal) else set())
        n = len(pool["ics"])
        if n == 0:
            log("candidate pool exhausted")
            break
        z = ev.draw(prior, eps)
        if a.method == "random":
            pick, sc = [int(i) for i in rng.choice(n, min(a.batch, n), replace=False)], None
        else:
            inp = ev.prep(pool["ics"], pool["st"], p.T, p.L)
            sc = score(fno, inp, z[: a.K_score], tq)
            pick = choose(sc, pool["feat"], min(a.batch, n), rng)
            del inp
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
        eps, dl = fit(eps, prior, data, a.steps0 if r == 1 else a.steps, a.M, rng, float(npairs), 1.0)
        del data
        t_fit = time.time() - t0 - t_score - t_sim
        z = ev.draw(prior, eps)
        e = test_error(fno, p, z)
        rec = {"round": r, "n_sims": len(acq["traj"]), "test": e, "fit_relL2": float(dl.mean()), "nonfinite_sims": int((~ok).sum()),
               "eps_spread": float(eps.std(0).mean()), "eps_norm": float(np.linalg.norm(eps, axis=1).mean()),
               "picked_scores": None if sc is None else [float(sc[k]) for k in pick],
               "pool_score_median": None if sc is None else float(np.median(sc)), "pool": n,
               "time": {"score": t_score, "sim": t_sim, "fit": t_fit, "test": time.time() - t0 - t_score - t_sim - t_fit}}
        st = {"eps": eps, "round": r, "traj": trj, "ics": np.stack(acq["ics"]), "id": np.stack(acq["id"]),
              "st": np.array(acq["st"], dtype=object) if stt is None else stt}
        np.savez(sf + ".tmp.npz", **st)
        os.replace(sf + ".tmp.npz", sf)
        open(hist, "a").write(json.dumps(rec) + "\n")
        log(f"round {r} sims {rec['n_sims']}", ev.brief({"post": e}), f"fit {rec['fit_relL2']:.4f} spread {rec['eps_spread']:.3f}",
            {k: round(v, 1) for k, v in rec["time"].items()})
    open(f"{out}/DONE", "w").write("")


# ---------------------------------------------------------------- summary: headline plot, sims to target error
def curve(path, key):
    H = [json.loads(l) for l in open(path)]
    return np.array([h["n_sims"] for h in H]), np.array([h["test"][key] for h in H])


def scratch_curve(path, key):
    """baselines.py summary json {N: {metric: {mean, std}}} -> (N, mean error)."""
    pts = sorted((int(N), v[key]["mean"]) for N, v in json.load(open(path)).items())
    return np.array([x for x, _ in pts]), np.array([y for _, y in pts])


def sims_to(n, err, target):
    """Fewest simulations reaching err <= target (linear interpolation in log N between rounds), or None."""
    for i in range(len(n)):
        if err[i] <= target:
            if i == 0 or n[i - 1] <= 0:
                return float(n[i])
            f = (np.log(err[i - 1]) - np.log(target)) / (np.log(err[i - 1]) - np.log(err[i]))
            return float(np.exp(np.log(n[i - 1]) + f * (np.log(n[i]) - np.log(n[i - 1]))))
    return None


def summary(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = a.metric
    names = {"active": "pretrained + active", "random": "pretrained + random", "generic": "pretrained + generic prior"}
    colors = {"active": "#1b7f6b", "random": "#7aa6d6", "generic": "#e0a030", "scratch": "#555555"}
    tg = sorted({os.path.basename(d).rsplit("_", 1)[0] for d in glob.glob(f"{a.results}/active/*_*") if os.path.isdir(d)})
    tg = [t for t in tg if not t.startswith("reh-")] + [t for t in tg if t.startswith("reh-")]
    fig, axes = plt.subplots(1, len(tg), figsize=(4.6 * len(tg), 3.8), squeeze=False)
    table = {}
    for ax, t in zip(axes[0], tg):
        sc_path = f"{a.scratch}/{t}.json"
        sc = scratch_curve(sc_path, key) if os.path.exists(sc_path) else None
        target = None
        if sc is not None and len(sc[0]):
            k = np.argmin(np.abs(sc[0] - a.target_n))
            target = float(sc[1][k])
            ax.plot(sc[0], sc[1], "o-", color=colors["scratch"], label="FNO from scratch", ms=3)
            ax.axhline(target, color=colors["scratch"], ls=":", lw=0.8)
        row = {"target_error": target, "target_from": f"scratch at N={int(sc[0][k])}" if target is not None else None}
        for m in ("active", "random", "generic"):
            h = f"{a.results}/active/{t}_{m}/history.jsonl"
            if not os.path.exists(h):
                continue
            n, e = curve(h, key)
            ax.plot(np.maximum(n, 1), e, "o-", color=colors[m], label=names[m], ms=3)
            row[m] = {"zero_shot": float(e[0]), "final": float(e[-1]), "n_final": int(n[-1]),
                      "sims_to_target": sims_to(np.maximum(n, 1e-9), e, target) if target is not None else None}
        if sc is not None and target is not None:
            row["scratch"] = {"sims_to_target": sims_to(sc[0].astype(float), sc[1], target)}
        table[t] = row
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("target simulations (0 plotted at 1)")
        ax.set_ylabel(f"test {key}")
        ax.set_title(t)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{a.results}/headline_{key}.png", dpi=150)
    json.dump(table, open(f"{a.results}/sims_to_target_{key}.json", "w"), indent=1)
    print(json.dumps(table, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--run", default="/workspace/runs/pretrain_A")
    ap.add_argument("--ckpt")
    ap.add_argument("--target", default="swe")
    ap.add_argument("--method", default="active", choices=["active", "random", "generic"])
    ap.add_argument("--out")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--K_score", type=int, default=8, help="particles used for scoring the pool")
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--score_frames", type=int, default=3)
    ap.add_argument("--steps0", type=int, default=200, help="Adam steps in round 1 (from the prior samples)")
    ap.add_argument("--steps", type=int, default=100, help="Adam steps in later rounds (warm start)")
    ap.add_argument("--M", type=int, default=2, help="(sim, frame) pairs per particle per step (K * M = 64 grads ~ 3.5 GB)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", help="summary: results dir of one checkpoint")
    ap.add_argument("--scratch", default="/workspace/results/scratch")
    ap.add_argument("--metric", default="raw_slot0")
    ap.add_argument("--target_n", type=int, default=512, help="summary: target error = scratch error at this N")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    main()

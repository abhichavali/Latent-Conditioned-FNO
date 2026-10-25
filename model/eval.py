"""Phase 3: evaluation of a pretraining checkpoint (frozen theta, phi).

  1 latent collapse   val-IC error with the table latents vs latents swapped across equations (ratio >> 1: z is used)
  2 zero-shot         G(.; mu_phi(G)) on val-IC (reference: table latents), rehearsal, SWE, AP; persistence baseline
  3 graph vs generic  p_phi(z | G) vs N(mean, cov of the training latents): error at the prior mean and of the
                      prior-predictive mean over K samples (= round 0 of active.py)
  4 calibration       rehearsal: z* fitted to all of an equation's windows; fraction of (z* - mu) / sigma inside 1 / 2
                      sigma, and the temperature tau = rms of it (sigma -> tau sigma calibrates)

  python eval.py --run /workspace/runs/pretrain_A [--ckpt .../step_X.eqx] [--out /workspace/results/<run>]

Metric: relL2 over frames 1..F-1 from the IC at frame 0, averaged over trajectories; "slot0" = primary field, "all" =
mean over present fields (see summarize): traj = per (trajectory, field) over (frames, x, y) in physical units;
raw = per (trajectory, frame, field) in physical units, averaged over frames (= baselines.py raw_*, unit-free; the
headline metric); rel = the training loss's relL2 in normalized space (= baselines.py rel_*).  Saved as results.json
(rewritten after every section) and zero_shot.png.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import train  # noqa: E402
from data import adapter, canon  # noqa: E402
from data.canon import F_MAX, S_MAX  # noqa: E402
from models.fno import C_IN  # noqa: E402

TARGETS = "/workspace/targets"
KEYS = ("u0", "st", "pres", "aux", "logT", "logL")


# ---------------------------------------------------------------- model, priors
def brief(r, keys=("raw_slot0", "rel_slot0", "rel_all")):
    return {m: [round(v[k], 4) for k in keys] for m, v in r.items() if isinstance(v, dict) and keys[0] in v}


def load(run, ckpt=None, data_dir="/workspace/data"):
    """Run dir -> cfg, splits, (fno, encoder, latent table), checkpoint path.  Data roots missing on this box are
    remapped to data_dir/<tag> (the A run was trained on another box)."""
    cfg = yaml.safe_load(open(f"{run}/config.yaml"))
    for r in cfg["data"]["roots"]:
        if not os.path.isdir(r["path"]):
            r["path"] = f"{data_dir}/{r['tag']}"
    S = json.load(open(f"{run}/splits.json"))
    params = train.build(cfg, len(S["train"]), jax.random.PRNGKey(0))
    opt = train.optimizer(cfg, params)
    path = ckpt or train.latest(run)[0]
    assert path, f"no checkpoint in {run}"
    params, _ = eqx.tree_deserialise_leaves(path, (params, opt.init(eqx.filter(params, eqx.is_array))))
    return cfg, S, params, path


def static_channels(k):
    """Lift input channels of static slot k (models/fno.py input order: fields, statics, presence (fields, statics),
    chi, x, y, aux (tanh m/s, log10 s per field slot, log10 rms per static slot)): map, presence flag, log-rms aux."""
    aux0 = F_MAX + S_MAX + (F_MAX + S_MAX) + 1 + 2
    assert aux0 + 2 * F_MAX + S_MAX == C_IN
    return [F_MAX + k, F_MAX + S_MAX + F_MAX + k, aux0 + 2 * F_MAX + k]


def adapt_statics(fno, ns, mode="copy"):
    """No pretraining equation has a second static, so the lift weights of static slot 1 are untrained (random init),
    yet a target with two statics (SWE: b, sponge s) feeds them.  "copy": slot 0's weights (map, presence, aux) into
    slot 1 (the default for SWE everywhere: zero-shot, particles, stage B init); "zero": ignore slot 1; "raw": as is."""
    if ns < 2 or mode == "raw":
        return fno
    W = fno.lift.weight
    W = W.at[:, static_channels(1)].set(W[:, static_channels(0)] if mode == "copy" else 0.0)
    return eqx.tree_at(lambda m: m.lift.weight, fno, W)


def encode(enc, graphs):
    """[canonical graph] -> mu, sigma (E, dz) numpy."""
    mu, ls = jax.jit(jax.vmap(enc))(jax.tree_util.tree_map(jnp.asarray, canon.pad(graphs)))
    return np.asarray(mu), np.exp(np.asarray(ls))


def graph_prior(enc, g, tau=1.0):
    """p_phi(z | G) = N(mu, diag (tau sigma)^2); tau = the rehearsal calibration temperature (1 = as trained)."""
    mu, sd = encode(enc, [g])
    return {"mu": mu[0], "L": np.diag(tau * sd[0])}


def generic_prior(Z):
    """N(mean, cov) of the training latents."""
    Z = np.asarray(Z, np.float64)
    return {"mu": Z.mean(0).astype(np.float32),
            "L": np.linalg.cholesky(np.cov(Z.T) + 1e-6 * np.eye(Z.shape[1])).astype(np.float32)}


def draw(prior, eps):
    """eps (K, dz) standard normal -> z = mu + L eps."""
    return (prior["mu"] + eps @ prior["L"].T).astype(np.float32)


# ---------------------------------------------------------------- inputs, prediction
def prep(ics, statics, T, L):
    """Raw ICs (n, nf, N, N), statics (n, ns, N, N) or None, scalars or per-IC arrays T, L -> model inputs (numpy)
    plus the per-IC normalization m, s padded to F_MAX (m = 0, s = 1 in absent slots)."""
    n = len(ics)
    out = [adapter.normalize(ic, None, None if statics is None else statics[i]) for i, ic in enumerate(ics)]
    d = {k: np.stack([o[k] for o in out]) for k in ("u0", "st", "pres", "aux")}
    d["m"], d["s"] = np.zeros((n, F_MAX), np.float32), np.ones((n, F_MAX), np.float32)
    for i, o in enumerate(out):
        d["m"][i, : len(o["m"])], d["s"][i, : len(o["s"])] = o["m"], o["s"]
    lt, ll = np.broadcast_to(np.log(np.asarray(T, np.float32)), (n,)), np.broadcast_to(np.log(np.asarray(L, np.float32)), (n,))
    d["logT"], d["logL"] = lt.astype(np.float32), ll.astype(np.float32)
    return d


def concat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def _fwd(fno, b, z, t):
    """One IC's inputs b, latent z (dz,), times t (Q,) -> (Q, F_MAX, N, N) normalized."""
    return jax.vmap(lambda tq: fno(b["u0"], b["st"], b["pres"], b["aux"], tq, z, b["logT"], b["logL"]))(t)


@eqx.filter_jit
def _moments(fno, b, Z, t):
    """b: chunk of c ICs, Z (K, c, dz), t (Q,) -> mean and variance over K of the normalized predictions (c, Q, F, N, N),
    accumulated by a scan over K (memory ~ c * Q forwards)."""
    def body(acc, z):
        p = jax.vmap(lambda bb, zz: _fwd(fno, bb, zz, t))(b, z)
        return (acc[0] + p, acc[1] + p * p), None

    shape = (b["u0"].shape[0], t.shape[0], *b["u0"].shape[1:])
    (s1, s2), _ = jax.lax.scan(body, (jnp.zeros(shape), jnp.zeros(shape)), Z)
    mean = s1 / Z.shape[0]
    return mean, jnp.maximum(s2 / Z.shape[0] - mean * mean, 0)


def predict(fno, inp, Z, t, c=None):
    """Yields (idx, mean, var) per chunk: physical units, fields padded to F_MAX.  Z (K, dz) shared by all ICs, or
    (K, n, dz) per IC.  Chunks of c ICs (default: ~64 forwards per scan step, ~1 GB); the last chunk is padded."""
    n, t = len(inp["u0"]), np.asarray(t, np.float32)
    c = c or max(1, 64 // len(t))
    for i in range(0, n, c):
        j = np.arange(i, min(i + c, n))
        jj = np.pad(j, (0, c - len(j)), mode="edge")
        z = np.broadcast_to(Z[:, None], (Z.shape[0], c, Z.shape[-1])) if Z.ndim == 2 else Z[:, jj]
        mean, var = _moments(fno, {k: jnp.asarray(inp[k][jj]) for k in KEYS}, jnp.asarray(z), jnp.asarray(t))
        s, m = inp["s"][j][:, None, :, None, None], inp["m"][j][:, None, :, None, None]
        yield j, np.asarray(mean)[: len(j)] * s + m, np.asarray(var)[: len(j)] * s * s


def errors(pred, true, inp):
    """(n, Q, F, N, N) physical (true zero-padded) -> per (trajectory, frame, field) mean squares (n, Q, F): squared
    error e2, truth y2, and the truth in the model's normalized space, IC mean removed, yc2 (for the training metric)."""
    m, s = inp["m"][:, None, :, None, None], inp["s"][:, None, :, None, None]
    return {"e2": ((pred - true) ** 2).mean((-2, -1)), "y2": (true**2).mean((-2, -1)), "yc2": (((true - m) / s) ** 2).mean((-2, -1)),
            "s": np.broadcast_to(inp["s"][:, None], true.shape[:3])}


def summarize(E, pres):
    """E: errors() concatenated over trajectories, pres (n, F) -> metrics, absent slots excluded.
      traj  |pred - true| / |true| per (trajectory, field) over (frames, x, y), physical (the phase-3 metric)
      raw   the same per (trajectory, frame, field), averaged over frames (baselines.py raw_*: physical, unit-free)
      rel   train.rel_l2 uncapped per (trajectory, frame, field) in normalized space (baselines.py rel_*: the loss)"""
    msk = np.where(pres, 1.0, np.nan)
    raw = np.sqrt(E["e2"] / np.maximum(E["y2"], 1e-30)) * msk[:, None]
    rel = np.sqrt(E["e2"]) / E["s"] / np.maximum(np.sqrt(E["yc2"]), 1e-2) * msk[:, None]
    traj = np.sqrt(E["e2"].sum(1) / np.maximum(E["y2"].sum(1), 1e-30)) * msk
    out = {"n": len(pres), "per_field_raw": np.nanmean(raw, (0, 1)).tolist()}
    for k, v in (("traj", traj), ("raw", raw.mean(1)), ("rel", rel.mean(1))):  # (n, F) per trajectory
        out[f"{k}_slot0"], out[f"{k}_all"] = float(np.mean(v[:, 0])), float(np.mean(np.nanmean(v, 1)))
    out["raw_slot0_median"] = float(np.median(raw.mean(1)[:, 0]))
    return out


# ---------------------------------------------------------------- evaluation sets
class Set:
    """Trajectories (n, frames, nf, N, N) raw + statics of one or more equations, loaded lazily per chunk.
    items: [(Eq, sample index, latent row or -1)]."""

    def __init__(self, items):
        self.items = items
        fr = {e.frames for e, _, _ in items}
        assert len(fr) == 1, f"mixed frame counts {fr}"
        self.frames = fr.pop()
        self.t = np.arange(1, self.frames) / (self.frames - 1)

    def load(self, idx):
        """-> inputs (prep), raw targets (len, frames-1, F_MAX, N, N) with NaN-free zero padding, presence (len, F_MAX)."""
        ins, ys, pres = [], [], np.zeros((len(idx), F_MAX), bool)
        for k, i in enumerate(idx):
            e, w, _ = self.items[i]
            raw = e.frames_raw(w, np.arange(e.frames))
            st = e.statics(w)
            ins.append(prep(raw[:1], None if st is None else st[None], e.T, e.L))
            y = np.zeros((e.frames - 1, F_MAX, *raw.shape[-2:]), np.float32)
            y[:, : e.nf] = raw[1:]
            ys.append(y)
            pres[k, : e.nf] = True
        return concat(ins), np.stack(ys), pres


def evaluate(fno, s, modes, chunk=4):
    """modes {name: Z (K, n, dz) per item, or "persistence"} -> {name: summary}, over all items of Set s."""
    n = len(s.items)
    acc, pres = {m: [] for m in modes}, []
    for i in range(0, n, chunk):
        idx = np.arange(i, min(i + chunk, n))
        inp, y, pr = s.load(idx)
        pres.append(pr)
        for name, Z in modes.items():
            if isinstance(Z, str):  # persistence: u(t) = u(0)
                ic = inp["u0"] * inp["s"][:, :, None, None] + inp["m"][:, :, None, None]
                pred = np.broadcast_to(ic[:, None], y.shape)
            else:
                pred = next(predict(fno, inp, Z[:, idx], s.t, c=len(idx)))[1]
            acc[name].append(errors(pred, y, inp))
    pres = np.concatenate(pres)
    return {m: summarize(concat(v), pres) for m, v in acc.items()}


def target_set(name):
    e = adapter.Eq(name.upper(), f"{TARGETS}/{name}")
    spec = json.load(open(f"{TARGETS}/{name}/spec.json"))
    return Set([(e, w, -1) for w in range(len(e.windows))]), spec["canonical"]


def pick(e, windows, k, rng):
    """Up to k windows of distinct ICs (pool IC groups), seeded."""
    w = np.asarray(windows)
    if not len(w):
        return []
    g = {int(x): i for i, x in enumerate(e.windows)}
    groups = np.array([e.groups[g[int(x)]] for x in w])
    first = [w[groups == u][0] for u in rng.permutation(np.unique(groups))]
    return [int(x) for x in first[:k]]


# ---------------------------------------------------------------- latent fitting (shared with active.py)
def make_fitter(fno, lr=0.05, cap=4.0):
    """Adam on whitened particles eps (K, dz), z = mu + L eps:
        loss_k = w_data * mean over M (sim, frame) pairs of relL2 (normalized space, as in training)
                 + w_prior * |eps_k - anchor_k|^2
    anchor = each particle's own prior draw (randomized MAP: keeps the ensemble spread; anchoring all at 0 = mu
    collapses it onto one MAP point).  data on device: u0 st pres aux logT logL (n, ...), y (n, Q, F, N, N)
    normalized, t (Q,).  Each particle draws its own M pairs per step."""
    adam = optax.adam(lr)

    @eqx.filter_jit
    def step(eps, ost, data, si, qi, mu, L, anchor, w_data, w_prior):
        def loss(eps):
            z = mu + eps @ L.T

            def one(zk, s, q):
                b = {k: data[k][s] for k in KEYS}
                f = jax.checkpoint(lambda bb, tq: fno(bb["u0"], bb["st"], bb["pres"], bb["aux"], tq, zk, bb["logT"], bb["logL"]))
                pred = jax.vmap(f)(b, data["t"][q])  # remat: K * M backward passes fit in ~6 GB
                rel, _ = train.rel_l2(pred[:, None], data["y"][s, q][:, None], b["pres"][:, :F_MAX], cap=cap)
                return rel

            d = jax.lax.map(lambda a: one(*a), (z, si, qi), batch_size=8)  # 8 particles at a time: memory / 4
            return (w_data * d + w_prior * ((eps - anchor) ** 2).sum(-1)).sum(), d

        (_, d), g = jax.value_and_grad(loss, has_aux=True)(eps)
        upd, ost = adam.update(g, ost)
        return eps + upd, ost, d

    def fit(eps, prior, data, steps, M, rng, w_data, w_prior=1.0, anchor=None):
        """data = fit_data(...) -> fitted eps (K, dz), last per-particle data relL2 (K,)."""
        data, n = data
        anchor = jnp.zeros_like(jnp.asarray(eps)) if anchor is None else jnp.asarray(anchor)
        eps = jnp.asarray(eps)
        ost = adam.init(eps)
        Q = data["y"].shape[1]
        mu, L = jnp.asarray(prior["mu"]), jnp.asarray(prior["L"])
        for _ in range(steps):
            si, qi = rng.integers(0, n, (len(eps), M)), rng.integers(0, Q, (len(eps), M))
            eps, ost, d = step(eps, ost, data, jnp.asarray(si), jnp.asarray(qi), mu, L, anchor, jnp.float32(w_data),
                               jnp.float32(w_prior))
        return np.asarray(eps), np.asarray(d)

    return fit


def fit_data(inp, traj, pad=100):
    """Inputs (prep of the ICs) + raw trajectories (n, frames, nf, N, N) -> device dict for the fitter, padded (by
    repetition) to a multiple of pad sims (100 = the whole budget: one compile per run; a mid-run recompile on a full
    GPU can fail to load its kernels); "n" = the real count."""
    n, fr, nf = traj.shape[:3]
    y = np.zeros((n, fr - 1, F_MAX, *traj.shape[-2:]), np.float32)
    y[:, :, :nf] = (traj[:, 1:] - inp["m"][:, None, :nf, None, None]) / inp["s"][:, None, :nf, None, None]
    j = np.arange(-(-n // pad) * pad) % n
    d = {k: jnp.asarray(inp[k][j]) for k in KEYS}
    d["y"], d["t"] = jnp.asarray(y[j]), jnp.asarray(np.arange(1, fr) / (fr - 1), jnp.float32)
    return d, n


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/workspace/runs/pretrain_A")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--val_per_eq", type=int, default=2)
    ap.add_argument("--reh_per_eq", type=int, default=8)
    ap.add_argument("--targets", nargs="*", default=["swe", "ap", "swe_m"])
    ap.add_argument("--calib_steps", type=int, default=300)
    ap.add_argument("--calib_eqs", type=int, default=0, help="calibrate on this many rehearsal equations (seeded; 0 = all)")
    ap.add_argument("--tau", type=float, default=0.0, help="also evaluate the graph prior with sigma -> tau sigma")
    ap.add_argument("--skip", nargs="*", default=[], help="sections to skip: data (all below) valic rehearsal calib")
    a = ap.parse_args()
    cfg, S, (fno, enc, Z), path = load(a.run, a.ckpt)
    step = int(os.path.basename(path)[5:13])
    out = a.out or f"/workspace/results/{os.path.basename(a.run.rstrip('/'))}_s{step}"
    os.makedirs(out, exist_ok=True)
    R = {"run": a.run, "ckpt": path, "step": step, "K": a.K, "metric": __doc__.split("Metric: ")[1].split("Saved")[0].strip()}
    if os.path.exists(f"{out}/results.json"):  # sections run again replace their entries, the rest is kept
        R = {**json.load(open(f"{out}/results.json")), **R}
    save = lambda: json.dump(R, open(f"{out}/results.json", "w"), indent=1)
    log = lambda *x: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)
    log(f"loaded {path} (step {step}), {len(S['train'])} train latents")
    gen = generic_prior(Z)
    eps = np.random.default_rng(0).standard_normal((a.K, Z.shape[1])).astype(np.float32)
    Zn = np.asarray(Z)

    # targets: zero-shot with graph / generic prior, at the mean and as the K-sample prior-predictive mean
    for name in a.targets:
        if not os.path.exists(f"{TARGETS}/{name}/spec.json"):
            continue
        s, g = target_set(name)
        gp = graph_prior(enc, g)
        n = len(s.items)
        ns = s.items[0][0].ns
        fno_t = adapt_statics(fno, ns)
        modes = {"graph_mu": np.broadcast_to(gp["mu"], (1, n, len(gp["mu"]))),
                 "generic_mu": np.broadcast_to(gen["mu"], (1, n, len(gen["mu"]))),
                 f"graph_K{a.K}": np.broadcast_to(draw(gp, eps)[:, None], (a.K, n, Zn.shape[1])),
                 f"generic_K{a.K}": np.broadcast_to(draw(gen, eps)[:, None], (a.K, n, Zn.shape[1])),
                 "persistence": "persistence"}
        if a.tau:
            modes[f"graph_tau_K{a.K}"] = np.broadcast_to(draw(graph_prior(enc, g, a.tau), eps)[:, None], (a.K, n, Zn.shape[1]))
        t0 = time.time()
        R[f"target/{name}"] = evaluate(fno_t, s, modes)
        R[f"target/{name}"]["prior_sigma_mean"] = float(np.diag(gp["L"]).mean())
        if ns >= 2:  # the untrained static slot 1: as trained (random) / ignored / copied from slot 0 (default above)
            R[f"target/{name}"]["static_slot1_graph_mu"] = {m: evaluate(adapt_statics(fno, ns, m), s, {"graph_mu": modes["graph_mu"]})["graph_mu"]
                                                            for m in ("raw", "zero")}
        log(name, f"{time.time() - t0:.0f}s", brief(R[f"target/{name}"]))
        save()

    roots_ok = all(os.path.exists(r["path"]) and os.listdir(r["path"]) for r in cfg["data"]["roots"])
    if "data" in a.skip or not roots_ok:
        log("data roots missing or skipped: val-IC / rehearsal / calibration not run")
        R["note"] = "val-IC / rehearsal skipped (data roots not on this box)"
        save()
        plot(R, out)
        return
    eqs = {e.gid: e for e in adapter.load_equations(cfg["data"]["roots"])}
    adapter.drop_nonfinite(list(eqs.values()), f"{cfg['cache_dir']}/nonfinite_{'+'.join(r['tag'] for r in cfg['data']['roots'])}.json")
    row = {g: i for i, g in enumerate(S["train"])}
    tags = "+".join(r["tag"] for r in cfg["data"]["roots"])
    graphs = canon.cached_graphs([(g, eqs[g].spec) for g in S["train"] + S["rehearsal"]], f"{cfg['cache_dir']}/graphs_{tags}.pkl")

    # 1 + 2: val-IC (collapse test, zero-shot reference)
    if "valic" not in a.skip:
        rng = np.random.default_rng(0)
        items = [(eqs[g], w, row[g]) for g in S["train"] for w in pick(eqs[g], S["val_ic"].get(g, []), a.val_per_eq, rng)]
        vs = Set(items)
        eq_of = np.array([it[2] for it in items])
        ue = np.unique(eq_of)
        nxt = {u: ue[(k + 1) % len(ue)] for k, u in enumerate(ue)}  # swap: the latent of the next equation
        mu_tr, _ = encode(enc, [graphs[S["train"][r]] for r in ue])
        mu_of = {u: mu_tr[k] for k, u in enumerate(ue)}
        modes = {"z_table": Zn[eq_of][None], "swap": Zn[[nxt[u] for u in eq_of]][None],
                 "graph_mu": np.stack([mu_of[u] for u in eq_of])[None],
                 "generic_mu": np.broadcast_to(gen["mu"], (1, len(items), Zn.shape[1])), "persistence": "persistence"}
        t0 = time.time()
        R["val_ic"] = evaluate(fno, vs, modes)
        for k in ("raw_slot0", "raw_all", "rel_all", "traj_all"):
            R["val_ic"][f"swap_ratio_{k}"] = R["val_ic"]["swap"][k] / R["val_ic"]["z_table"][k]
        R["val_ic"]["equations"] = len(ue)
        log("val_ic", f"{time.time() - t0:.0f}s", brief(R["val_ic"]),
            "swap ratio (rel_all)", round(R["val_ic"]["swap_ratio_rel_all"], 3))
        save()

    # 2 + 3: rehearsal (held-out structures)
    reh = S["rehearsal"]
    mu_r, sd_r = encode(enc, [graphs[g] for g in reh])
    if "rehearsal" not in a.skip:
        rng = np.random.default_rng(1)
        items = [(eqs[g], w, j) for j, g in enumerate(reh) for w in pick(eqs[g], eqs[g].windows, a.reh_per_eq, rng)]
        rs = Set(items)
        j_of = np.array([it[2] for it in items])
        n = len(items)
        zg = np.stack([draw({"mu": mu_r[j], "L": np.diag(sd_r[j])}, eps) for j in range(len(reh))])  # (E, K, dz)
        modes = {"graph_mu": mu_r[j_of][None], "generic_mu": np.broadcast_to(gen["mu"], (1, n, Zn.shape[1])),
                 f"graph_K{a.K}": np.moveaxis(zg[j_of], 1, 0),
                 f"generic_K{a.K}": np.broadcast_to(draw(gen, eps)[:, None], (a.K, n, Zn.shape[1])), "persistence": "persistence"}
        if a.tau:
            zt = np.stack([draw({"mu": mu_r[j], "L": np.diag(a.tau * sd_r[j])}, eps) for j in range(len(reh))])
            modes[f"graph_tau_K{a.K}"] = np.moveaxis(zt[j_of], 1, 0)
        t0 = time.time()
        R["rehearsal"] = evaluate(fno, rs, modes)
        R["rehearsal"]["equations"] = len(reh)
        log("rehearsal", f"{time.time() - t0:.0f}s", brief(R["rehearsal"]))
        save()

    # 4: calibration on rehearsal equations: z* fitted (no prior) to all windows, starting at mu
    if "calib" not in a.skip:
        fit = make_fitter(fno)
        t0, res, fit_err = time.time(), [], []
        sel = sorted(np.random.default_rng(2).choice(len(reh), a.calib_eqs, replace=False)) if 0 < a.calib_eqs < len(reh) else range(len(reh))
        for j in sel:
            g = reh[j]
            e = eqs[g]
            ws = list(e.windows)
            traj = np.stack([e.frames_raw(w, np.arange(e.frames)) for w in ws])
            st = np.stack([e.statics(w) for w in ws]) if e.ns else None
            d = fit_data(prep(traj[:, 0], st, e.T, e.L), traj)
            ep, dl = fit(np.zeros((1, Zn.shape[1]), np.float32), {"mu": mu_r[j], "L": np.diag(sd_r[j])}, d, a.calib_steps, 32,
                         np.random.default_rng(j), 1.0, 0.0)
            res.append(ep[0])
            fit_err.append(float(dl[0]))
        r = np.abs(np.stack(res))  # whitened residuals (z* - mu) / sigma
        tau = float(np.sqrt((r**2).mean()))
        R["calibration"] = {"equations": len(res), "steps": a.calib_steps, "inside_1sigma": float((r < 1).mean()),
                            "inside_2sigma": float((r < 2).mean()), "expected": [0.683, 0.954], "tau": tau,
                            "inside_1tau_sigma": float((r < tau).mean()), "inside_2tau_sigma": float((r < 2 * tau).mean()),
                            "median_abs_residual": float(np.median(r)), "fit_relL2_last_batch": float(np.mean(fit_err)),
                            "note": "dims the backbone ignores stay at mu (residual 0), which biases toward 'inside'"}
        log("calibration", f"{time.time() - t0:.0f}s", R["calibration"])
        save()
    plot(R, out)


def plot(R, out):
    """Zero-shot bars: data sets (val-IC, rehearsal) in the training metric rel_all (their physical relL2 is dominated by
    trajectories whose truth is ~0), targets in raw_slot0 (physical, = the scratch baseline's metric)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ["z_table", "swap", "graph_mu", "generic_mu", "graph_K32", "generic_K32", "persistence"]
    colors = {"z_table": "#4c72b0", "swap": "#c44e52", "graph_mu": "#2a9d8f", "generic_mu": "#e9c46a", "graph_K32": "#264653",
              "generic_K32": "#f4a261", "persistence": "#999999"}
    groups = [([k for k in ("val_ic", "rehearsal") if k in R], "rel_all", "relL2 (training metric), all fields"),
              ([k for k in R if k.startswith("target/")], "raw_slot0", "relL2 per frame, primary field (physical)")]
    groups = [g for g in groups if g[0]]
    fig, axes = plt.subplots(1, len(groups), figsize=(9, 3.6), squeeze=False)
    w = 0.8 / len(modes)
    for ax, (sets, key, ylab) in zip(axes[0], groups):
        for k, m in enumerate(modes):
            xs = [i for i, s in enumerate(sets) if m in R[s]]
            if xs:
                ax.bar([x + (k - len(modes) / 2 + 0.5) * w for x in xs], [R[sets[x]][m][key] for x in xs], w, label=m, color=colors[m])
        ax.set_xticks(range(len(sets)), [s.replace("target/", "") for s in sets])
        ax.set_ylabel(ylab, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(f"zero-shot, step {R['step']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/zero_shot.png", dpi=150)


if __name__ == "__main__":
    main()

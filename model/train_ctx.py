"""v3 pretraining: Neural-Process objective, no latent table (models/context.py, data/ctx.py, config/ctx.yaml).

Per batch element: a training equation, k context windows C (k = 0 w.p. p0, else log-uniform on 1..K_max) and Q
target windows T from other IC groups (train windows only, pretrain_AB's frozen splits).
  L = relL2(G_theta(target IC, t; z), target frames) + beta * KL(q(z | G, C u T) || q(z | G, C)),
  z ~ q(z | G, C) for the first Q // 2 targets of each equation (the test-time posterior, trained directly, so it is
  anchored without the KL), z ~ q(z | G, C u T) for the rest.
beta = beta * max(beta_floor, linear ramp over beta_ramp of the steps after beta_delay of them).  Why: the decoder's z
path is zero at init (dRel/dz = 0), so any KL there is the encoder's only gradient and Adam makes it a full-size step:
the first v3 run collapsed q to context-invariance by step ~1100 (KL 127 -> 0.05; rehearsal k0 = k64 at step 18k).
Graph-term dropout (g_drop): G carries every coefficient (canon: sign, log10|c|), so on training equations C is redundant
and q(z | G, C) ~ q(z | G) (second run, step 6k: k0..k64 and k16ct equal to 4 decimals, dmu 0.08 on |z| = 11).  For a
g_drop fraction of equations each graph term is dropped with probability rho ~ U(0, 1), so z must come from C.
Non-finite updates are skipped; a log window
with only non-finite steps exits without a checkpoint (run.sh resumes from the last finite one).

  python train_ctx.py --config config/ctx.yaml [--steps N] [--set model.width=128 --set name=ctx_smoke]

Validation (every val_minutes and at every start): fixed rehearsal (held-out structures) and val-IC batches, error with
z = posterior mean of q(z | G, first k context windows) for k in val_k, plus persistence and k=16 with C u T.
Run dir, checkpoints (params = (fno, context encoder)), resume and SIGTERM as in train.py.
"""

import argparse
import json
import os
import resource
import shutil
import signal
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
from data import ctx  # noqa: E402
from data.canon import F_MAX  # noqa: E402
from models.context import ContextEncoder, CtxFNO, kl_diag  # noqa: E402
from models.fno import batched  # noqa: E402


def build(cfg, key):
    k1, k2 = jax.random.split(key)
    return CtxFNO(k1, **cfg["model"]), ContextEncoder(k2, dz=cfg["model"]["dz"], **cfg["encoder"])


def optimizer(cfg):
    t = cfg["train"]
    sched = optax.warmup_cosine_decay_schedule(0.0, t["lr"], t["warmup"], t["steps"], end_value=t["lr"] * t["lr_end"])
    return optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(t["clip"]), optax.adamw(sched, weight_decay=t["wd"])), 10**9)  # never apply a non-finite update


def flat(b):
    """(E, Q, ...) targets -> v1's (E * Q, ...) batch for models/fno.batched."""
    E, Q = b["u0"].shape[:2]
    out = {k: b[k].reshape(E * Q, *b[k].shape[2:]) for k in ("u0", "st", "pres", "aux", "y", "t")}
    out["logT"], out["logL"] = jnp.repeat(b["logT"], Q), jnp.repeat(b["logL"], Q)
    return out, Q


def posterior(enc, b, G, masks, tkeep=None):
    """masks (E, P, S), tkeep (E, T) graph terms kept (None: all) -> mu, log sigma (E, P, dz)."""
    g = jax.tree_util.tree_map(lambda a: a[b["eq"]], G)
    if tkeep is not None:
        g = {**g, "term_mask": g["term_mask"] * tkeep}
    return jax.vmap(enc)(g, b["cx"], b["cst"], b["caux"], b["cpres"], b["logT"], b["logL"], masks)


def make_steps(opt, cap, g_drop=0.0):
    @eqx.filter_jit
    def train_step(params, opt_state, b, G, key, beta):
        def loss_fn(params):
            fno, enc = params
            k1, k2, k3, key_z = jax.random.split(key, 4)
            E, T = b["eq"].shape[0], G["term_mask"].shape[1]
            rho = jnp.where(jax.random.uniform(k1, (E,)) < g_drop, jax.random.uniform(k2, (E,)), 0.0)
            tkeep = (jax.random.uniform(k3, (E, T)) >= rho[:, None]).astype(jnp.float32)
            mu, ls = posterior(enc, b, G, jnp.stack([b["cmask"], b["ctmask"]], 1), tkeep)
            z = mu + jnp.exp(ls) * jax.random.normal(key_z, mu.shape)  # (E, 2, dz): C, C u T
            fb, Q = flat(b)
            pc = (jnp.arange(z.shape[0] * Q) % Q < Q // 2)[:, None]  # rows decoded from q(z | G, C)
            zz = jnp.where(pc, jnp.repeat(z[:, 0], Q, 0), jnp.repeat(z[:, 1], Q, 0))
            rel, r = train.rel_l2(batched(fno, fb, zz), fb["y"], fb["pres"][:, :F_MAX], cap=cap)
            mc = jnp.broadcast_to((fb["pres"][:, None, :F_MAX] > 0) & pc[:, None], r.shape).astype(jnp.float32)
            kl = kl_diag(mu[:, 1], ls[:, 1], mu[:, 0], ls[:, 0])
            k0 = (b["k"] == 0).astype(jnp.float32)
            avg = lambda v, w: (v * w).sum() / jnp.maximum(w.sum(), 1)
            aux = {"rel": rel, "rel_c": (r * mc).sum() / jnp.maximum(mc.sum(), 1), "kl": kl.mean(), "kl_k0": avg(kl, k0), "kl_kpos": avg(kl, 1 - k0),
                   "sig_c": jnp.exp(ls[:, 0]).mean(), "sig_ct": jnp.exp(ls[:, 1]).mean(),
                   "dmu": jnp.sqrt(((mu[:, 1] - mu[:, 0]) ** 2).sum(-1)).mean(), "k_mean": b["k"].mean()}
            loss = rel + beta * kl.mean()
            return loss, {"loss": loss, **aux}

        (_, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
        upd, opt_state = opt.update(grads, opt_state, eqx.filter(params, eqx.is_array))
        aux["gnorm"] = optax.global_norm(grads)
        return eqx.apply_updates(params, upd), opt_state, aux

    @eqx.filter_jit
    def eval_step(params, b, G, mask, nog=False):
        """z = posterior mean under mask (E, S) (nog: no graph terms) -> uncapped per-term relL2 (E * Q, nq, F), presence."""
        fno, enc = params
        mu, _ = posterior(enc, b, G, mask[:, None], jnp.zeros(G["term_mask"].shape[1:])[None] if nog else None)
        fb, Q = flat(b)
        _, r = train.rel_l2(batched(fno, fb, jnp.repeat(mu[:, 0], Q, 0)), fb["y"], fb["pres"][:, :F_MAX])
        return r, fb["pres"][:, :F_MAX]

    return train_step, eval_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, help="override total steps (also on resume)")
    ap.add_argument("--set", action="append", default=[], help="key.sub=value (yaml) override at first start, e.g. model.width=128")
    a = ap.parse_args()
    cfg = train.load_config(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        d = cfg
        for p in k.split(".")[:-1]:
            d = d[p]
        d[k.split(".")[-1]] = yaml.safe_load(v)
    run_dir = os.path.join(cfg["run_root"], cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    cf = f"{run_dir}/config.yaml"
    if os.path.exists(cf):
        cfg = yaml.safe_load(open(cf))
    if a.steps:
        cfg["train"]["steps"] = a.steps
    yaml.safe_dump(cfg, open(cf, "w"))
    if not os.path.exists(f"{run_dir}/splits.json"):
        shutil.copy(cfg["split"]["copy_from"], f"{run_dir}/splits.json")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    jax.config.update("jax_compilation_cache_dir", cfg.get("jax_cache", "/workspace/jaxcache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 2)

    t, d = cfg["train"], cfg["data"]
    S, tr, reh, G = train.setup(cfg, run_dir)
    t0 = time.time()
    cache = ctx.load_cache(tr + reh, cfg["ctx_cache"])
    print(f"ctx cache loaded in {time.time() - t0:.0f}s: {sum(v[0].nbytes for v in cache.values()) / 1e9:.1f} GB", flush=True)
    Gd = jax.device_put(G)
    params = build(cfg, jax.random.PRNGKey(t["seed"]))
    opt = optimizer(cfg)
    opt_state = opt.init(eqx.filter(params, eqx.is_array))
    path, step = train.latest(run_dir)
    train_s0 = 0.0
    if path:
        params, opt_state = eqx.tree_deserialise_leaves(path, (params, opt_state))
        side = path[:-4] + ".json"
        train_s0 = json.load(open(side))["train_s"] if os.path.exists(side) else 0.0
        print(f"resumed from {path} at step {step}", flush=True)
    train_step, eval_step = make_steps(opt, t.get("rel_cap"), t.get("g_drop", 0.0))
    npar = lambda m: sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array)))
    print(f"params: fno {npar(params[0]) / 1e6:.1f}M, context encoder {npar(params[1]) / 1e6:.2f}M; train eqs {len(tr)}, "
          f"rehearsal eqs {len(reh)}", flush=True)

    K, E, Q, nq = d["k_max"], d["eqs"], d["targets"], d["queries"]
    rows = [(e, i) for i, e in enumerate(tr)]
    stream = ctx.CtxStream(rows, [S["train_windows"][e.gid] for e in tr], cache, E, Q, nq, K, d["p0"], t["seed"],
                           d["eq_weight_power"], d["workers"], d["prefetch"])
    nb = t["val_batches"]
    vi = [r for r in rows if len(S["val_ic"][r[0].gid])]
    val = {"val_ic": ctx.fixed_batches(vi, [S["val_ic"][e.gid] for e, _ in vi], [S["train_windows"][e.gid] for e, _ in vi],
                                       cache, E, Q, nq, K, 10**6, nb)}
    if reh:
        rr = [(e, len(tr) + j) for j, e in enumerate(reh)]
        val["rehearsal"] = ctx.fixed_batches(rr, [e.windows for e in reh], [e.windows for e in reh], cache, E, Q, nq, K, 10**6 + 1, nb)

    from tensorboardX import SummaryWriter

    tb = SummaryWriter(f"{run_dir}/tb")
    logf = open(f"{run_dir}/log.jsonl", "a")
    sf = f"{run_dir}/start_time"
    if not os.path.exists(sf):
        open(sf, "w").write(str(time.time()))
    t_run, t_proc = float(open(sf).read()), time.time()

    def log(rec):
        now = time.time()
        rec = {"step": step, "time": now, "elapsed_h": (now - t_run) / 3600, "train_h": (train_s0 + now - t_proc) / 3600, **rec}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        for k, v in rec.items():
            if k not in ("step", "time") and isinstance(v, (int, float)) and np.isfinite(v):
                tb.add_scalar(k, v, step)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "time"}), flush=True)

    cap = t.get("rel_cap") or np.inf
    c = lambda v: cap * np.tanh(v / cap)

    def summ(out):
        r = np.concatenate([x.reshape(-1) for x, _ in out])
        m = np.concatenate([np.broadcast_to(mk[:, None, :], x.shape).reshape(-1) for x, mk in out]) > 0
        r0 = np.concatenate([x[:, :, 0].reshape(-1) for x, _ in out])
        return {"rel": float(c(r[m]).mean()), "rel_median": float(np.median(r[m])), "rel_slot0": float(c(r0).mean())}

    def run_eval(name, k, ct=False, nog=False):
        out = []
        for b in val[name]:
            S_ = b["cmask"].shape[1]
            mask = np.arange(S_)[None] < np.minimum(k, b["k"])[:, None]
            if ct:
                mask = mask | (np.arange(S_)[None] >= K)
            out.append(jax.device_get(eval_step(params, jax.device_put(b), Gd, jnp.asarray(mask), nog)))
        return summ(out)

    def persistence(name):
        out = []
        for b in val[name]:
            E_, Q_ = b["u0"].shape[:2]
            y = b["y"].reshape(E_ * Q_, *b["y"].shape[2:])
            u0 = b["u0"].reshape(E_ * Q_, *b["u0"].shape[2:])
            pres = b["pres"].reshape(E_ * Q_, -1)[:, :F_MAX]
            _, r = train.rel_l2(jnp.broadcast_to(u0[:, None], y.shape), jnp.asarray(y), jnp.asarray(pres))
            out.append((np.asarray(r), pres))
        return summ(out)

    persist = {name: persistence(name) for name in val}

    def validate():
        out = {}
        for name in val:
            out.update({f"{name}/persistence/{k}": v for k, v in persist[name].items()})
            for k in t["val_k"]:
                out.update({f"{name}/k{k}/{m}": v for m, v in run_eval(name, k).items()})
            out.update({f"{name}/k16ct/{m}": v for m, v in run_eval(name, 16, ct=True).items()})
            for k in (0, 16):  # no graph: z from the context alone (does the encoder read C?)
                out.update({f"{name}/nog_k{k}/{m}": v for m, v in run_eval(name, k, nog=True).items()})
        return out

    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    base_key = jax.random.PRNGKey(t["seed"] + 1)
    total = t["steps"]
    ramp, delay = max(int(t["beta_ramp"] * total), 1), int(t.get("beta_delay", 0) * total)
    it = stream.iterate(step)
    acc, t_ck, t_log, s_log = [], time.time(), time.time(), step
    log({"event": "start", **validate()})
    t_val = time.time()
    while step < total:
        beta = t["beta"] * max(t["beta_floor"], min(max(step - delay, 0) / ramp, 1.0))
        b = jax.device_put(next(it))
        params, opt_state, aux = train_step(params, opt_state, b, Gd, jax.random.fold_in(base_key, step), jnp.float32(beta))
        acc.append(aux)
        step += 1
        if step % t["log_every"] == 0 or step == total:
            m = jax.device_get(acc)
            rec = {f"train/{k}": float(np.nanmean([x[k] for x in m])) for k in m[0]}
            rec["train/nonfinite"] = int(sum(not np.isfinite(x["loss"]) for x in m))
            rec.update({"sched/beta": beta, "steps_per_s": (step - s_log) / (time.time() - t_log)})
            acc, t_log, s_log = [], time.time(), step
            if rec["train/nonfinite"] == len(m):
                log({**rec, "event": "every step non-finite: exit without checkpoint"})
                os._exit(3)
            if time.time() - t_val > 60 * t["val_minutes"] or step == total:
                rec.update(validate())
                t_val = time.time()
            log(rec)
        if time.time() - t_ck > 60 * t["ckpt_minutes"] or step == total or stop:
            train.save(run_dir, step, params, opt_state, t["keep"], train_s0 + time.time() - t_proc)
            t_ck = time.time()
            log({"ckpt": f"{run_dir}/ckpt/step_{step:08d}.eqx"})
        if stop:
            log({"event": "stopped by SIGTERM"})
            os._exit(0)
    tb.close()
    open(f"{run_dir}/DONE", "w").write(str(step))
    print("done", step, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

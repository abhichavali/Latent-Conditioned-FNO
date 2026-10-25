"""Pretraining: conditioned FNO + per-equation latents z_i + graph encoder prior p_phi(z | G).

  L = relL2(G_theta(inputs, t; z~_i), target) + lam * -log N(z_i; mu_phi(G_i), sigma_phi(G_i)^2) + beta * |z_i|^2
  z~_i = z_i, or with probability p a reparameterized sample from p_phi(. | G_i).  lam = p = 0 for the first
  warm_frac of the steps, then both ramp linearly (over ramp_frac) to their targets.

  python train.py --config config/A.yaml            # run dir = run_root/name; resumes from its latest checkpoint

The run dir freezes config.yaml and splits.json on first start; log.jsonl + tb/ (TensorBoard) + ckpt/.
Data order and per-step RNG are functions of (seed, step), so a resumed run continues exactly.
"""

import argparse
import glob
import json
import os
import resource
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

from data import adapter, canon, splits  # noqa: E402
from data.canon import F_MAX  # noqa: E402
from models.encoder import GraphEncoder  # noqa: E402
from models.fno import CondFNO, batched  # noqa: E402


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def load_config(path):
    cfg = yaml.safe_load(open(path))
    if "base" in cfg:
        cfg = merge(load_config(os.path.join(os.path.dirname(path), cfg.pop("base"))), cfg)
    return cfg


def rel_l2(pred, y, mask, floor=1e-2, cap=None):
    """Relative L2 per (sample, query time, field slot), averaged over present slots.  (B, Q, F, N, N), mask (B, F).
    cap: soft cap cap * tanh(r / cap) per term, so a few near-constant ICs whose fields later grow by 100-1000x
    (IC-std normalization cannot know that) do not dominate the gradient."""
    err = jnp.sqrt(((pred - y) ** 2).mean((-2, -1)))
    r = err / jnp.maximum(jnp.sqrt((y**2).mean((-2, -1))), floor)
    if cap:
        r = cap * jnp.tanh(r / cap)
    m = jnp.broadcast_to(mask[:, None, :], r.shape)
    return (r * m).sum() / m.sum(), r


def build(cfg, n_train, key):
    k1, k2, k3 = jax.random.split(key, 3)
    dz = cfg["model"]["dz"]
    fno = CondFNO(k1, **cfg["model"])
    enc = GraphEncoder(k2, dz=dz, **cfg["encoder"])
    Z = 0.01 * jax.random.normal(k3, (n_train, dz))
    return fno, enc, Z


def optimizer(cfg, params):
    t = cfg["train"]
    sched = lambda lr: optax.warmup_cosine_decay_schedule(0.0, lr, t["warmup"], t["steps"], end_value=lr * t["lr_end"])
    tx = {"fno": optax.adamw(sched(t["lr"]), weight_decay=t["wd"]),
          "enc": optax.adamw(sched(t["lr"] * t["enc_lr_mult"]), weight_decay=t["wd"]),
          "z": optax.adam(sched(t["lr"] * t["z_lr_mult"]))}
    arrays = eqx.filter(params, eqx.is_array)
    labels = tuple(jax.tree_util.tree_map(lambda _, n=n: n, p) for p, n in zip(arrays, ("fno", "enc", "z")))
    return optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(t["clip"]), optax.multi_transform(tx, labels)), 50)


def make_steps(opt, cap):
    def forward(params, b, G, zt):
        return batched(params[0], b, zt)

    @eqx.filter_jit
    def train_step(params, opt_state, b, G, key, lam, p_enc, beta):
        def loss_fn(params):
            fno, enc, Z = params
            g = jax.tree_util.tree_map(lambda a: a[b["eq"]], G)
            mu, ls = jax.vmap(enc)(g)
            zi = Z[b["eq"]]
            k1, k2 = jax.random.split(key)
            use = jax.random.uniform(k2, (zi.shape[0],)) < p_enc
            zt = jnp.where(use[:, None], mu + jnp.exp(ls) * jax.random.normal(k1, zi.shape), zi)
            rel, _ = rel_l2(forward(params, b, G, zt), b["y"], b["pres"][:, :F_MAX], cap=cap)
            nll = 0.5 * (((zi - mu) / jnp.exp(ls)) ** 2 + 2 * ls + jnp.log(2 * jnp.pi)).mean(-1).mean()
            loss = rel + lam * nll + beta * (zi**2).sum(-1).mean()
            return loss, {"loss": loss, "rel": rel, "nll": nll, "sigma": jnp.exp(ls).mean(), "enc_frac": use.mean()}

        (_, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
        upd, opt_state = opt.update(grads, opt_state, eqx.filter(params, eqx.is_array))
        aux["gnorm"] = optax.global_norm(grads)
        return eqx.apply_updates(params, upd), opt_state, aux

    @eqx.filter_jit
    def eval_step(params, b, G, mode):
        """mode: "z_table" (latent table), "prior_mu" (z = mu_phi(G), prior only), "swap" (table latents permuted
        across the batch's equations: does the backbone use z?), "persistence" (u(t) = u(0)).
        Returns per-term relL2 (B, Q, F), uncapped, and the presence mask (B, F)."""
        fno, enc, Z = params
        if mode == "prior_mu":
            z, _ = jax.vmap(enc)(jax.tree_util.tree_map(lambda a: a[b["eq"]], G))
        else:
            z = Z[b["eq"]]
        if mode == "swap":
            z = jnp.roll(z, 1, axis=0)
        pred = jnp.broadcast_to(b["u0"][:, None], b["y"].shape) if mode == "persistence" else forward(params, b, G, z)
        _, r = rel_l2(pred, b["y"], b["pres"][:, :F_MAX])
        return r, b["pres"][:, :F_MAX]

    return train_step, eval_step


def save(run_dir, step, params, opt_state, keep):
    d = f"{run_dir}/ckpt"
    os.makedirs(d, exist_ok=True)
    tmp = f"{d}/.tmp.eqx"
    eqx.tree_serialise_leaves(tmp, (params, opt_state))
    os.replace(tmp, f"{d}/step_{step:08d}.eqx")
    for old in sorted(glob.glob(f"{d}/step_*.eqx"))[:-keep]:
        os.remove(old)


def latest(run_dir):
    c = sorted(glob.glob(f"{run_dir}/ckpt/step_*.eqx"))
    return (c[-1], int(os.path.basename(c[-1])[5:13])) if c else (None, 0)


def setup(cfg, run_dir):
    """Equations, frozen splits, graph table (train rows first, then rehearsal rows)."""
    eqs = adapter.load_equations(cfg["data"]["roots"])
    tags = "+".join(r["tag"] for r in cfg["data"]["roots"])
    bad = adapter.drop_nonfinite(eqs, f"{cfg['cache_dir']}/nonfinite_{tags}.json")
    print(f"non-finite windows dropped: {sum(map(len, bad.values()))} in {len(bad)} equations", flush=True)
    sf = f"{run_dir}/splits.json"
    if os.path.exists(sf):
        S = json.load(open(sf))
    else:
        S = splits.make(eqs, cfg["split"])
        json.dump(S, open(sf, "w"))
    by = {e.gid: e for e in eqs}
    missing = [g for g in S["train"] + S["rehearsal"] if g not in by]
    assert not missing, f"split equations missing on disk: {missing[:5]}"
    train, reh = [by[g] for g in S["train"]], [by[g] for g in S["rehearsal"]]
    for k in ("train_windows", "val_ic"):  # frozen splits stay valid if the non-finite scan changes
        S[k] = {g: [w for w in v if w not in by[g].bad] for g, v in S[k].items()}
    graphs = canon.cached_graphs([(e.gid, e.spec) for e in train + reh], f"{cfg['cache_dir']}/graphs_{tags}.pkl")
    G = canon.pad([graphs[e.gid] for e in train + reh])
    return S, train, reh, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, help="override total steps (also on resume)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run_dir = os.path.join(cfg["run_root"], cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    cf = f"{run_dir}/config.yaml"
    if os.path.exists(cf):  # the run's frozen config wins on restart
        cfg = yaml.safe_load(open(cf))
    if a.steps:
        cfg["train"]["steps"] = a.steps
    yaml.safe_dump(cfg, open(cf, "w"))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))  # one mmap per equation file
    jax.config.update("jax_compilation_cache_dir", cfg.get("jax_cache", "/workspace/jaxcache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 2)

    t, d = cfg["train"], cfg["data"]
    S, train, reh, G = setup(cfg, run_dir)
    print(json.dumps({"counts": S["counts"], "structures": S["structures"]}), flush=True)
    Gd = jax.device_put(G)
    params = build(cfg, len(train), jax.random.PRNGKey(t["seed"]))
    opt = optimizer(cfg, params)
    opt_state = opt.init(eqx.filter(params, eqx.is_array))
    path, step = latest(run_dir)
    if path:
        params, opt_state = eqx.tree_deserialise_leaves(path, (params, opt_state))
        print(f"resumed from {path} at step {step}", flush=True)
    train_step, eval_step = make_steps(opt, t.get("rel_cap"))
    n_par = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(params[0], eqx.is_array)))
    print(f"backbone params {n_par / 1e6:.1f}M, train eqs {len(train)}, rehearsal eqs {len(reh)}", flush=True)

    rows = [(e, i) for i, e in enumerate(train)]
    stream = adapter.Stream(rows, [np.array(S["train_windows"][e.gid]) for e in train], d["batch"], d["queries"],
                            t["seed"], d["eq_weight_power"], d["workers"], d["prefetch"])
    vi = [(r, np.array(S["val_ic"][r[0].gid])) for r in rows if S["val_ic"][r[0].gid]]
    nb = t["val_batches"]
    val = {"val_ic": adapter.fixed_batches([r for r, _ in vi], [w for _, w in vi], d["batch"], d["queries"], 10**6, nb)}
    if reh:
        val["rehearsal"] = adapter.fixed_batches([(e, len(train) + j) for j, e in enumerate(reh)],
                                                 [e.windows for e in reh], d["batch"], d["queries"], 10**6 + 1, nb)

    from tensorboardX import SummaryWriter

    tb = SummaryWriter(f"{run_dir}/tb")
    logf = open(f"{run_dir}/log.jsonl", "a")
    sf = f"{run_dir}/start_time"
    if not os.path.exists(sf):
        open(sf, "w").write(str(time.time()))
    t_run, t_proc = float(open(sf).read()), time.time()

    def log(rec):
        now = time.time()
        rec = {"step": step, "time": now, "elapsed_h": (now - t_run) / 3600, "proc_elapsed_s": now - t_proc, **rec}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        for k, v in rec.items():
            if k not in ("step", "time") and isinstance(v, (int, float)) and np.isfinite(v):
                tb.add_scalar(k, v, step)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items() if k != "time"}), flush=True)

    cap = t.get("rel_cap") or np.inf

    def run_eval(name, mode):
        """rel = soft-capped mean (same quantity as train/rel), plus uncapped mean, median, slot-0 capped mean."""
        out = jax.device_get([eval_step(params, jax.device_put(b), Gd, mode) for b in val[name]])
        r = np.concatenate([x.reshape(-1) for x, _ in out])
        m = np.concatenate([np.broadcast_to(mk[:, None, :], x.shape).reshape(-1) for x, mk in out]) > 0
        r0 = np.concatenate([x[:, :, 0].reshape(-1) for x, _ in out])
        c = lambda v: cap * np.tanh(v / cap) if np.isfinite(cap) else v
        return {"rel": float(c(r[m]).mean()), "rel_uncapped": float(r[m].mean()), "rel_median": float(np.median(r[m])),
                "rel_slot0": float(c(r0).mean())}

    persist = {name: run_eval(name, "persistence") for name in val}  # constant: computed once

    def validate():
        out = {}
        for name in val:
            p = persist[name]
            out.update({f"{name}/persistence/{k}": v for k, v in p.items()})
            for mode in (["z_table", "prior_mu", "swap"] if name == "val_ic" else ["prior_mu"]):
                r = run_eval(name, mode)
                out.update({f"{name}/{mode}/{k}": v for k, v in r.items()})
                out[f"{name}/{mode}/ratio_to_persistence"] = r["rel"] / p["rel"]
                out[f"{name}/{mode}/median_ratio_to_persistence"] = r["rel_median"] / p["rel_median"]
        out["val_ic/swap_ratio"] = out["val_ic/swap/rel"] / out["val_ic/z_table/rel"]
        out["val_ic/swap_ratio_median"] = out["val_ic/swap/rel_median"] / out["val_ic/z_table/rel_median"]
        return out

    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))  # clean stop: checkpoint, then exit (no DONE)

    base_key = jax.random.PRNGKey(t["seed"] + 1)
    total, warm, ramp = t["steps"], int(t["warm_frac"] * t["steps"]), max(int(t["ramp_frac"] * t["steps"]), 1)
    it = stream.iterate(step)
    acc, t_ck, t_log, s_log = [], time.time(), time.time(), step
    log({"event": "start", **validate()})  # also on every resume
    t_val = time.time()
    while step < total:
        f = float(np.clip((step - warm) / ramp, 0, 1))
        b = jax.device_put(next(it))
        params, opt_state, aux = train_step(params, opt_state, b, Gd, jax.random.fold_in(base_key, step),
                                            jnp.float32(t["lam"] * f), jnp.float32(t["p_enc"] * f), jnp.float32(t["beta"]))
        acc.append(aux)
        step += 1
        if step % t["log_every"] == 0 or step == total:
            m = jax.device_get(acc)
            rec = {f"train/{k}": float(np.nanmean([x[k] for x in m])) for k in m[0]}
            rec["train/nonfinite"] = int(sum(not np.isfinite(x["loss"]) for x in m))
            zn = np.linalg.norm(np.asarray(params[2]), axis=-1)
            rec.update({"z/norm_mean": float(zn.mean()), "z/norm_max": float(zn.max()), "z/norm_min": float(zn.min()),
                        "sched/lam": t["lam"] * f, "sched/p_enc": t["p_enc"] * f,
                        "steps_per_s": (step - s_log) / (time.time() - t_log)})
            acc, t_log, s_log = [], time.time(), step
            if time.time() - t_val > 60 * t["val_minutes"] or step == total:
                rec.update(validate())
                t_val = time.time()
            log(rec)
        if time.time() - t_ck > 60 * t["ckpt_minutes"] or step == total or stop:
            save(run_dir, step, params, opt_state, t["keep"])
            t_ck = time.time()
            log({"ckpt": f"{run_dir}/ckpt/step_{step:08d}.eqx"})
        if stop:
            log({"event": "stopped by SIGTERM"})
            os._exit(0)
    tb.close()
    open(f"{run_dir}/DONE", "w").write(str(step))
    print("done", step, flush=True)
    os._exit(0)  # prefetch threads are daemonless; skip their shutdown


if __name__ == "__main__":
    main()

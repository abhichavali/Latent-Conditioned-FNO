"""FNO from scratch on N simulations of one target: the "no prior" curve of test error vs target simulations.

Same backbone and input / output interface as the pretrained model (models/fno.py, data/adapter.py normalization),
with model.dz = 0, so the conditioning MLP sees only [Fourier features of t, log T / 4, log L / 4]: no latent, no
encoder.  Settings: config/scratch.yaml (scratch: section; optimizer, batch, queries, rel_cap from A.yaml).

  python baselines.py gen     --config config/scratch.yaml [--only swe ap]   # 512 training sims per target
  python baselines.py sweep   --config config/scratch.yaml --target ap [--sizes 8 16] [--seeds 0]
  python baselines.py summary --config config/scratch.yaml                   # {swe,ap}.json, meta.json, plot

Data: <out>/data/{swe,ap}_train/ in the test-set layout (targets.write_test, seed data_seed): spec.json, u.npy,
static.npy, params.npy (knobs).  Runs: <out>/runs/<target>_N<N>_s<seed>.json + .npz (per-term test errors) when
finished, skipped on rerun; .ckpt.eqx + .ckpt.json while running, resumed.  Logs: <out>/logs/.

Each run holds out max(val_min, val_frac * N) of its N sims (seeded) for early stopping, trains on the rest with a
cosine schedule over clip(steps_per_sim * N, steps_min, steps_max) steps, evaluates val every eval_every steps, stops
after `patience` evals without a min_delta relative improvement, and is tested at its best-val step.

Test metric (fixed test sets, 100 ICs, frames 1..14 from the IC at frame 0), as eval.py summarize: *_slot0 = primary
field, *_all = also averaged over the present fields.  rel = train.rel_l2 uncapped per (IC, frame, field) in the
model's normalized space (the loss), averaged over ICs and frames; raw = |pred - true| / |true| per (IC, frame, field)
in physical units, averaged over frames and ICs (unit-free; eval.py's headline); traj = the same per (IC, field) over
(frames, x, y), averaged over ICs.
"""

import argparse
import glob
import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import targets  # noqa: E402
from data import adapter  # noqa: E402
from data.canon import F_MAX  # noqa: E402
from models.fno import CondFNO, batched  # noqa: E402
from train import load_config, rel_l2  # noqa: E402

KEYS = ("u0", "st", "pres", "aux", "y", "m", "s")
UNITS = ("fields", "static", "L", "frame_dt", "frames", "rhs")  # train and test data must agree on these


def load(name, d):
    """Every trajectory of a dataset dir as model inputs: u0 st pres aux (n, ...), y (n, frames - 1, F_MAX, N, N),
    m s (n, nf) normalization; logT, logL scalars; plus the Eq."""
    e = adapter.Eq(name.upper(), d)
    items = []
    for w in e.windows:
        raw = e.frames_raw(w, np.arange(e.frames))
        items.append(adapter.normalize(raw[0], raw[1:], e.statics(w)))
    D = {k: np.stack([it[k] for it in items]) for k in KEYS}
    D["logT"], D["logL"] = adapter.scalars(e.T, e.L)
    return D, e


def _scalars(D, n):
    return {"logT": np.full(n, D["logT"]), "logL": np.full(n, D["logL"])}


def train_batch(D, idx, B, Q, seed, step):
    """B trajectories of idx (with replacement), Q distinct query frames each; a function of (seed, step) only."""
    r = np.random.default_rng([seed, step])
    i, nq = r.choice(idx, B), D["y"].shape[1]
    q = np.sort(np.stack([r.choice(nq, Q, replace=False) for _ in range(B)]), 1)
    b = {k: D[k][i] for k in ("u0", "st", "pres", "aux")}
    return {**b, "y": D["y"][i[:, None], q], "t": ((q + 1) / nq).astype(np.float32), **_scalars(D, B)}


def fwd(model, b):
    return batched(model, b, jnp.zeros((b["u0"].shape[0], 0)))  # dz = 0: no latent


@eqx.filter_jit
def errors(model, b, persist=False):
    """Per (sample, frame) terms: rel (B, Q, F_MAX) = train.rel_l2 uncapped in normalized space; e2, y2 (B, Q, nf) =
    mean squared error and mean squared truth in physical units.  persist: predict u(t) = u(0)."""
    pred = jnp.broadcast_to(b["u0"][:, None], b["y"].shape) if persist else fwd(model, b)
    _, r = rel_l2(pred, b["y"], b["pres"][:, :F_MAX])
    nf = b["m"].shape[1]
    s, m = b["s"][:, None, :, None, None], b["m"][:, None, :, None, None]
    pr, yr = pred[:, :, :nf] * s + m, b["y"][:, :, :nf] * s + m
    return r, ((pr - yr) ** 2).mean((-2, -1)), (yr**2).mean((-2, -1))


def evaluate(model, D, idx, persist=False, bs=4):
    """All frames of trajectories idx -> rel, e2, y2 (n, Q, nf).  Padded to full batches (one compile)."""
    nf = D["m"].shape[1]
    j = np.concatenate([idx, np.repeat(idx[-1:], -len(idx) % bs)])
    t = np.arange(1, D["y"].shape[1] + 1, dtype=np.float32) / D["y"].shape[1]
    out = []
    for k in range(0, len(j), bs):
        i = j[k : k + bs]
        b = {**{key: D[key][i] for key in KEYS}, "t": np.broadcast_to(t, (len(i), len(t))).copy(), **_scalars(D, len(i))}
        out.append(jax.device_get(errors(model, jax.device_put(b), persist)))
    r, e2, y2 = (np.concatenate([o[x] for o in out])[: len(idx)] for x in (0, 1, 2))
    return r[..., :nf], e2, y2


def stats(r, e2, y2):
    raw = np.sqrt(e2 / np.maximum(y2, 1e-30))
    traj = np.sqrt(e2.sum(1) / np.maximum(y2.sum(1), 1e-30))  # (n, nf)
    out = {f"{k}_{a}": float(v[..., 0].mean() if a == "slot0" else v.mean()) for k, v in (("rel", r), ("raw", raw), ("traj", traj))
           for a in ("slot0", "all")}
    return {**out, "rel_per_field": r.mean((0, 1)).tolist(), "raw_per_field": raw.mean((0, 1)).tolist(),
            "traj_per_field": traj.mean(0).tolist(), "raw_slot0_median": float(np.median(raw[..., 0].mean(1))),
            "rel_slot0_median": float(np.median(r[..., 0].mean(1)))}


def make_step(opt, cap):
    @eqx.filter_jit
    def step(model, opt_state, b):
        loss_fn = lambda m: rel_l2(fwd(m, b), b["y"], b["pres"][:, :F_MAX], cap=cap)[0]
        loss, g = eqx.filter_value_and_grad(loss_fn)(model)
        upd, opt_state = opt.update(g, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, upd), opt_state, loss

    return step


def run(cfg, name, D, T, N, seed):
    """One (target, N, seed) run; resumes from its checkpoint, skipped once its json exists."""
    sc, t, d = cfg["scratch"], cfg["train"], cfg["data"]
    tag = f"{sc['out']}/runs/{name}_N{N:03d}_s{seed}"
    if os.path.exists(tag + ".json"):
        print(f"skip {tag}: done", flush=True)
        return
    nval = max(sc["val_min"], round(sc["val_frac"] * N))
    perm = np.random.default_rng(seed).permutation(N)
    vi, ti = np.sort(perm[:nval]), np.sort(perm[nval:])
    total = int(np.clip(sc["steps_per_sim"] * N, sc["steps_min"], sc["steps_max"]))
    sched = optax.warmup_cosine_decay_schedule(0.0, t["lr"], t["warmup"], total, end_value=t["lr"] * t["lr_end"])
    opt = optax.chain(optax.clip_by_global_norm(t["clip"]), optax.adamw(sched, weight_decay=t["wd"]))
    model = CondFNO(jax.random.PRNGKey(seed), **cfg["model"])
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    best = model
    st = {"step": 0, "best": float("inf"), "best_step": 0, "ref": float("inf"), "bad": 0, "curve": [], "train_s": 0.0}
    if os.path.exists(tag + ".ckpt.json"):
        model, opt_state, best = eqx.tree_deserialise_leaves(tag + ".ckpt.eqx", (model, opt_state, best))
        st = json.load(open(tag + ".ckpt.json"))
        print(f"resumed {tag} at step {st['step']}", flush=True)
    step_fn = make_step(opt, t.get("rel_cap"))
    B, Q = d["batch"], d["queries"]
    pool = ThreadPoolExecutor(1)  # next batch is gathered while the GPU runs this one
    nxt = pool.submit(train_batch, D, ti, B, Q, seed, st["step"])
    t0, t_ck, s0, losses, train_s0 = time.time(), time.time(), st["step"], [], st["train_s"]
    print(f"run {name} N {N} seed {seed}: train {len(ti)} val {nval} max steps {total}", flush=True)
    while st["step"] < total:
        b = nxt.result()
        nxt = pool.submit(train_batch, D, ti, B, Q, seed, st["step"] + 1)
        model, opt_state, loss = step_fn(model, opt_state, jax.device_put(b))
        losses.append(loss)
        st["step"] += 1
        stop = False
        if st["step"] % sc["eval_every"] == 0 or st["step"] == total:
            v = float(evaluate(model, D, vi)[0].mean())
            tr = float(np.mean(jax.device_get(losses)))
            losses = []
            st["curve"].append([st["step"], round(tr, 6), round(v, 6)])
            if v < st["best"]:
                st["best"], st["best_step"], best = v, st["step"], model
            if v < st["ref"] * (1 - sc["min_delta"]):
                st["ref"], st["bad"] = v, 0
            else:
                st["bad"] += 1
            stop = st["bad"] >= sc["patience"]
            print(json.dumps({"N": N, "seed": seed, "step": st["step"], "train": tr, "val": v, "best": st["best"],
                              "bad": st["bad"], "steps_per_s": (st["step"] - s0) / (time.time() - t0),
                              "min": (train_s0 + time.time() - t0) / 60}), flush=True)
        st["train_s"] = train_s0 + time.time() - t0
        if stop:
            break
        if time.time() - t_ck > 60 * sc["ckpt_minutes"]:
            eqx.tree_serialise_leaves(tag + ".ckpt.tmp", (model, opt_state, best))
            os.replace(tag + ".ckpt.tmp", tag + ".ckpt.eqx")
            json.dump(st, open(tag + ".ckpt.json", "w"))
            t_ck = time.time()
    pool.shutdown()
    r, e2, y2 = evaluate(best, T, np.arange(len(T["u0"])))
    np.savez(tag + ".npz", rel=r, e2=e2, y2=y2)
    rec = {"target": name, "N": N, "seed": seed, "n_train": len(ti), "n_val": nval, "val_sims": vi.tolist(),
           "max_steps": total, "steps": st["step"], "best_step": st["best_step"], "early_stop": st["step"] < total,
           "best_val": st["best"], "train_s": st["train_s"], "steps_per_s": (st["step"] - s0) / (time.time() - t0),
           "host": socket.gethostname(), "gpu": jax.devices()[0].device_kind, "test": stats(r, e2, y2),
           "val_curve": st["curve"]}
    json.dump(rec, open(tag + ".json", "w"))
    for f in glob.glob(tag + ".ckpt.*"):
        os.remove(f)
    print(json.dumps({k: rec[k] for k in ("target", "N", "seed", "steps", "best_step", "train_s", "test")}), flush=True)


def sweep(cfg, name, sizes, seeds):
    sc = cfg["scratch"]
    D, e = load(name, f"{sc['out']}/data/{name}_train")
    T, te = load(name, f"{sc['targets']}/{name}")
    bad = [k for k in UNITS if e.spec[k] != te.spec[k]]
    assert not bad, f"{name}: training data and test set differ in {bad} (units?): regenerate with `gen`"
    assert len(D["u0"]) >= max(sizes), f"{name}: {len(D['u0'])} training sims < {max(sizes)}"
    os.makedirs(f"{sc['out']}/runs", exist_ok=True)
    pf = f"{sc['out']}/runs/{name}_persistence.json"
    if not os.path.exists(pf):
        json.dump(stats(*evaluate(None, T, np.arange(len(T["u0"])), persist=True)), open(pf, "w"))
    for seed in seeds:
        for N in sizes:
            if seed == seeds[0] or N <= sc["seed2_max_n"]:
                run(cfg, name, D, T, N, seed)


def summary(cfg):
    sc = cfg["scratch"]
    out, meta = sc["out"], {"targets": {}}
    for name in targets.SPECS:
        runs = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/runs/{name}_N*_s*.json")) if ".ckpt" not in f]
        if not runs:
            continue
        by, res = {}, {}
        for r in runs:
            by.setdefault(r["N"], []).append(r)
        for N in sorted(by):
            rs = by[N]
            vals = lambda k: [r["test"][k] for r in rs if k in r["test"]]  # traj_* exists only in later runs
            ms = lambda k: {"mean": float(np.mean(vals(k))), "std": float(np.std(vals(k), 0)), "n": len(vals(k))} if vals(k) else None
            res[str(N)] = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs], **{f"{k}_{a}": ms(f"{k}_{a}") for k in
                           ("rel", "raw", "traj") for a in ("slot0", "all")},
                           **{f"{k}_per_field": np.mean(vals(f"{k}_per_field"), 0).tolist() if vals(f"{k}_per_field") else None
                              for k in ("rel", "raw", "traj")},
                           "n_train": rs[0]["n_train"], "n_val": rs[0]["n_val"],
                           **{k: [r[k] for r in rs] for k in ("steps", "max_steps", "best_step", "early_stop", "train_s", "gpu")}}
        json.dump(res, open(f"{out}/{name}.json", "w"), indent=1)
        spec = json.load(open(f"{sc['targets']}/{name}/spec.json"))
        pf = f"{out}/runs/{name}_persistence.json"
        meta["targets"][name] = {"fields": spec["fields"], "L": spec["L"], "frame_dt": spec["frame_dt"], "rhs": spec["rhs"],
                                 "persistence": json.load(open(pf)) if os.path.exists(pf) else None}
    meta.update({"metric": __doc__.split("Test metric")[1].strip(), "std": "over seeds, ddof 0 (0 for one seed); n = runs with that metric",
                 "model": cfg["model"], "train": cfg["train"], "data": cfg["data"], "scratch": sc})
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    plot(out, meta)


def plot(out, meta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    names = [n for n in targets.SPECS if os.path.exists(f"{out}/{n}.json")]
    rows = (("raw", "physical relL2 per frame (headline)"), ("rel", "normalized relL2 (training loss)"))
    fig, axs = plt.subplots(len(rows), len(names), figsize=(5.2 * len(names), 4 * len(rows)), squeeze=False)
    title = {"swe": "Shallow water (tsunami)", "ap": "Aliev-Panfilov (scar)", "fire": "Wildfire (firebreaks)"}
    for (metric, ylab), row in zip(rows, axs):
        for ax, n in zip(row, names):
            res = json.load(open(f"{out}/{n}.json"))
            N = np.array([int(k) for k in res])
            f = meta["targets"][n]["fields"]
            for a, lab, c in (("slot0", f"primary field ({f[0]})", "#2a78d6"), ("all", f"all fields ({', '.join(f)})", "#eb6834")):
                m = np.array([res[k][f"{metric}_{a}"]["mean"] for k in res])
                s = np.array([res[k][f"{metric}_{a}"]["std"] for k in res])
                ax.errorbar(N, m, yerr=s, color=c, lw=2, marker="o", ms=5, capsize=3, label=lab)
                ax.annotate(f"{m[-1]:.3f}", (N[-1], m[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8,
                            color="#52514e")
            p = meta["targets"][n]["persistence"]
            if p:
                ax.axhline(p[f"{metric}_slot0"], color="#8a8985", lw=1, ls="--", label=f"persistence u(t) = u(0), {f[0]}")
            ax.set(xscale="log", yscale="log", xlabel="target simulations N", ylabel=f"test {ylab}", title=title[n])
            ax.set_xticks(N, [str(x) for x in N])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.grid(True, which="major", color="#e4e3df", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("FNO from scratch: test error (100 ICs x 14 frames) vs number of target simulations", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/scratch_vs_N.png", dpi=150)
    print(f"wrote {out}/scratch_vs_N.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "sweep", "summary"])
    ap.add_argument("--config", default="config/scratch.yaml")
    ap.add_argument("--only", nargs="+", default=["swe", "ap"])
    ap.add_argument("--target", choices=list(targets.SPECS))
    ap.add_argument("--sizes", nargs="+", type=int)
    ap.add_argument("--seeds", nargs="+", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    sc = cfg["scratch"]
    jax.config.update("jax_compilation_cache_dir", cfg.get("jax_cache", "/workspace/jaxcache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 2)
    if a.cmd == "gen":
        for n in a.only:
            d = f"{sc['out']}/data/{n}_train"
            if not os.path.exists(f"{d}/spec.json"):  # write_test writes spec.json last
                targets.write_test(n, d, n=sc["n_sims"], seed=sc["data_seed"])
    elif a.cmd == "sweep":
        sweep(cfg, a.target, a.sizes or sc["sizes"], a.seeds or sc["seeds"])
    else:
        summary(cfg)


if __name__ == "__main__":
    main()

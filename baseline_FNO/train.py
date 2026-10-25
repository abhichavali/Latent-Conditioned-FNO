"""FNO baseline (no prior, no active learning): learn the one-step map u(t) -> u(t + dt) from N random-IC
rollouts, sweep N, test on held-out ICs (one-step and 19-step autoregressive relative L2).

  python train.py --data /root/baseline_data --out results

Writes results/results.json, results/error_vs_data.png, RESULTS.md (next to this file) and truth-vs-FNO
rollouts for the viz pages: results/viz/{swe,ap}_{truth,fno}/.
"""

import argparse
import json
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_collection"))
import pde  # noqa: E402
from data import EQS  # noqa: E402

SIZES = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
PRIMARY = {"swe": "eta (surface)", "ap": "u (potential)"}


class SpectralConv(eqx.Module):
    wr: jax.Array
    wi: jax.Array
    m: int = eqx.field(static=True)

    def __init__(self, c, m, key):
        k1, k2 = jax.random.split(key)
        self.wr = jax.random.uniform(k1, (2, c, c, m, m)) / (c * c)
        self.wi = jax.random.uniform(k2, (2, c, c, m, m)) / (c * c)
        self.m = m

    def __call__(self, x):  # (C, N, N), lowest m x m modes (both signs of ky)
        m, n = self.m, x.shape[-1]
        xh, w = jnp.fft.rfft2(x), self.wr + 1j * self.wi
        out = jnp.zeros((w.shape[2], n, n // 2 + 1), xh.dtype)
        out = out.at[:, :m, :m].set(jnp.einsum("ixy,ioxy->oxy", xh[:, :m, :m], w[0]))
        out = out.at[:, -m:, :m].set(jnp.einsum("ixy,ioxy->oxy", xh[:, -m:, :m], w[1]))
        return jnp.fft.irfft2(out, s=(n, n))


class FNO(eqx.Module):
    """FNO2d (Li et al. 2021): 1x1 lift, L Fourier layers (spectral conv + 1x1 skip, GELU), 2-layer 1x1 head."""

    lift: eqx.nn.Conv2d
    spec: list
    skip: list
    p1: eqx.nn.Conv2d
    p2: eqx.nn.Conv2d

    def __init__(self, cin, cout, key, width=64, modes=16, layers=4):
        k = jax.random.split(key, 2 * layers + 3)
        self.lift = eqx.nn.Conv2d(cin, width, 1, key=k[0])
        self.spec = [SpectralConv(width, modes, k[1 + i]) for i in range(layers)]
        self.skip = [eqx.nn.Conv2d(width, width, 1, key=k[1 + layers + i]) for i in range(layers)]
        self.p1, self.p2 = eqx.nn.Conv2d(width, 128, 1, key=k[-2]), eqx.nn.Conv2d(128, cout, 1, key=k[-1])

    def __call__(self, x):
        h = self.lift(x)
        for i, (s, w) in enumerate(zip(self.spec, self.skip)):
            h = s(h) + w(h)
            h = jax.nn.gelu(h) if i < len(self.spec) - 1 else h
        return self.p2(jax.nn.gelu(self.p1(h)))


def step_fn(model, x, static):  # normalized state (C, N, N) -> next normalized state (residual)
    return x + model(jnp.concatenate([x, static]))


@eqx.filter_jit
def train_step(model, opt_state, opt, traj, static, key, bs):
    k1, k2 = jax.random.split(key)
    i = jax.random.randint(k1, (bs,), 0, traj.shape[0])
    t = jax.random.randint(k2, (bs,), 0, traj.shape[1] - 1)
    x, y = traj[i, t], traj[i, t + 1]

    def loss(m):
        return jnp.mean((jax.vmap(lambda a: step_fn(m, a, static))(x) - y) ** 2)

    l, g = eqx.filter_value_and_grad(loss)(model)
    upd, opt_state = opt.update(g, opt_state, eqx.filter(model, eqx.is_array))
    return eqx.apply_updates(model, upd), opt_state, l


@eqx.filter_jit
def rollout(model, x0, static, steps):  # x0 (B, C, N, N) -> (B, steps + 1, C, N, N)
    def body(x, _):
        x = jax.vmap(lambda a: step_fn(model, a, static))(x)
        return x, x

    _, xs = jax.lax.scan(body, x0, None, length=steps)
    return jnp.concatenate([x0[:, None], jnp.moveaxis(xs, 0, 1)], 1)


@eqx.filter_jit
def one_step(model, x, static):
    return jax.vmap(lambda a: step_fn(model, a, static))(x)


def nrmse(p, y, sig):  # (..., C, N, N), per-field scale (C,) -> (..., C).  Relative L2 blows up when the truth
    return jnp.sqrt(((p - y) ** 2).mean((-2, -1))) / sig  # decays to ~0 (absorbed tsunami, extinct excitation)


def evaluate(model, test, mu, sd, static, bs=16):
    """Normalized RMSE per field (RMSE / test std of that field): one-step (avg over ICs and t), autoregressive."""
    F, os_, ro = test.shape[1], [], []
    sig = test.std((0, 1, 3, 4))
    for i in range(0, len(test), bs):
        y = test[i : i + bs]
        yn = (y - mu) / sd
        p1 = one_step(model, yn[:, :-1].reshape(-1, *yn.shape[2:]), static).reshape(yn[:, 1:].shape) * sd + mu
        os_.append(np.asarray(nrmse(p1, y[:, 1:], sig)))
        pr = rollout(model, yn[:, 0], static, F - 1) * sd + mu
        ro.append(np.asarray(nrmse(pr[:, 1:], y[:, 1:], sig)))
    os_, ro = np.concatenate(os_), np.concatenate(ro)  # (n, F-1, C)
    return {"one_step": os_.mean((0, 1)).tolist(), "rollout_mean": ro.mean((0, 1)).tolist(),
            "rollout_final": ro[:, -1].mean(0).tolist(), "rollout_curve": ro.mean(0)[:, 0].tolist()}


def fit(traj_n, static, steps, key, bs=16, lr=1e-3):
    k1, k2 = jax.random.split(key)
    C = traj_n.shape[2]
    model = FNO(C + static.shape[0], C, k1)
    opt = optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    for s, k in enumerate(jax.random.split(k2, steps)):
        model, opt_state, l = train_step(model, opt_state, opt, traj_n, static, k, bs)
    return model, float(l)


FINE = {"swe": 4, "ap": 5}  # demo playback: frames per FNO step (2 s and 0.5 t.u.)


def viz(name, model, test, mu, sd, static_n, static_raw, data_dir, out, steps=57):
    """Held-out IC rolled out 3x the training horizon: solver truth at the fine playback rate vs FNO steps
    upsampled by motion-compensated interpolation (interp.py), in the tsunami / heart page formats."""
    import interp

    spec, k = EQS[name], FINE[name]
    # SWE: the strongest source; AP: the most excited tissue at the end of the test window (sustained re-entry)
    i = int(np.argmax(test[:, 0, 0].max((-2, -1)) if name == "swe" else (test[:, -1, 0] > 0.5).mean((-2, -1))))
    x0 = test[i, 0]
    u0 = np.concatenate([x0, static_raw]) if name == "swe" else x0
    sub = spec["substeps"] // k
    truth = np.asarray(pde.rollout(pde.solver_key(spec), jnp.asarray(u0)[None], spec["dt"], sub, steps * k + 1))[0, :, : x0.shape[0]]
    coarse = np.asarray(rollout(model, jnp.asarray((x0 - mu[0]) / sd[0])[None], static_n, steps))[0] * sd[0] + mu[0]
    fno = interp.upsample(coarse, k)
    dt = spec["dt"] * sub
    b_vis = np.load(f"{data_dir}/swe_bvis.npy") if name == "swe" else None
    for tag, tr in (("truth", truth), ("fno", fno)):
        d = f"{out}/viz/{name}_{tag}"
        os.makedirs(d, exist_ok=True)
        if name == "swe":
            np.save(f"{d}/h.npy", np.maximum(tr[:, 0] - b_vis, 0).astype(np.float32))
            np.save(f"{d}/b.npy", b_vis)
            meta = {"dt": dt, "time_unit": "s", "lx": spec["L"], "ly": spec["L"], "vexag": 4, "wexag": 6, "title": f"Coastal tsunami ({tag})"}
        else:
            np.save(f"{d}/u.npy", np.clip(tr[:, 0], -0.5, 1.5).astype(np.float32))
            meta = {"dt": dt * 12.9, "time_unit": "ms", "geometry": "sheet", "range": [0, 1], "title": f"Aliev-Panfilov ({tag})"}
        json.dump(meta, open(f"{d}/meta.json", "w"))
    err = np.asarray(nrmse(jnp.asarray(coarse[1:]), jnp.asarray(truth[k::k]), test.std((0, 1, 3, 4))))[:, 0]
    return {"ic": i, "steps": steps, "rel_l2_at": {str(t): float(err[t - 1]) for t in (1, 10, 19, 38, 57)},
            "interp_check": interp.check(truth[: 30 * k + 1, :1], k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/baseline_data")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--eqs", nargs="+", default=list(EQS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res_f = f"{a.out}/results.json"
    res = json.load(open(res_f)) if os.path.exists(res_f) else {}
    for name in a.eqs:
        train_all = np.load(f"{a.data}/{name}_train.npy")
        test = np.load(f"{a.data}/{name}_test.npy")
        static_raw = np.load(f"{a.data}/swe_static.npy") if name == "swe" else None  # (2, N, N)
        res.setdefault(name, {})
        model = None
        for N in a.sizes:
            for seed in range(3 if N <= 64 else 1):
                tag = f"{N}/{seed}"
                if tag in res[name] and not (N == a.sizes[-1] and seed == 0):
                    continue  # resume; the largest model is retrained for the viz
                sub = train_all[np.random.default_rng(seed).permutation(len(train_all))[:N]]
                mu = sub.mean((0, 1, 3, 4), keepdims=True)[0]
                sd = sub.std((0, 1, 3, 4), keepdims=True)[0] + 1e-8
                static_n = jnp.zeros((0, *sub.shape[-2:]))
                if static_raw is not None:
                    m_, s_ = static_raw.mean((1, 2), keepdims=True), static_raw.std((1, 2), keepdims=True) + 1e-12
                    static_n = jnp.asarray((static_raw - m_) / s_)
                steps = int(np.clip(15 * N * (sub.shape[1] - 1) / 16, min(1500, a.max_steps), a.max_steps))
                t = time.time()
                model, loss = fit(jnp.asarray((sub - mu) / sd), static_n, steps, jax.random.PRNGKey(seed))
                ev = evaluate(model, test, mu, sd, static_n)
                res[name][tag] = {"N": N, "seed": seed, "steps": steps, "train_loss": loss, "time_s": round(time.time() - t, 1), **ev}
                print(name, tag, {k: (round(v[0], 4) if isinstance(v, list) else v) for k, v in res[name][tag].items() if k != "rollout_curve"}, flush=True)
                json.dump(res, open(res_f, "w"), indent=1)
        res[name]["viz"] = viz(name, model, test, mu, sd, static_n, static_raw, a.data, a.out)
        json.dump(res, open(res_f, "w"), indent=1)
    import report

    report.write(res, a.out)


if __name__ == "__main__":
    main()

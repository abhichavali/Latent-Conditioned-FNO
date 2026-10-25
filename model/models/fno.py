"""Conditioned, time-conditioned FNO2d (periodic, no padding), adapted from baseline_FNO/train.py.

Conditioning c = [z (d_z), FourierFeatures(t) (geometric freqs 1..max_freq), log T / 4, log L / 4].  A small MLP maps
c to per-layer modulation: FiLM h <- (1 + gamma) * h + beta before the activation, and a softmax mixture of M spectral
filter banks R_l(k; c) = sum_m alpha_{l,m}(c) R_{l,m}(k).  The MLP's last layer is zero-initialised, so the network
starts as a plain FNO (gamma = beta = 0, uniform alpha).  Output: all F_MAX field slots at time t, residual on u0.

Inputs per sample (see data/adapter.py): u0 (F_MAX, N, N), st (S_MAX, N, N), pres (F_MAX + S_MAX,), aux (AUX_DIM,);
input channels = fields, statics, presence maps, the reserved mask chi (zeros), coordinates (x, y) in [0, 1)^2,
and aux maps.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from data.adapter import AUX_DIM
from data.canon import F_MAX, S_MAX

C_IN = F_MAX + S_MAX + (F_MAX + S_MAX) + 1 + 2 + AUX_DIM


class SpectralMix(eqx.Module):
    """M filter banks on the lowest m x m modes (both signs of ky); outputs mixed by alpha (M,)."""

    wr: jax.Array
    wi: jax.Array
    m: int = eqx.field(static=True)

    def __init__(self, c, m, M, key):
        k1, k2 = jax.random.split(key)
        self.wr = jax.random.uniform(k1, (M, 2, c, c, m, m)) / (c * c)
        self.wi = jax.random.uniform(k2, (M, 2, c, c, m, m)) / (c * c)
        self.m = m

    def __call__(self, x, alpha):  # x (C, N, N) float32
        m, n = self.m, x.shape[-1]
        xh, w = jnp.fft.rfft2(x), self.wr + 1j * self.wi
        lo = jnp.einsum("ixy,mioxy,m->oxy", xh[:, :m, :m], w[:, 0], alpha)
        hi = jnp.einsum("ixy,mioxy,m->oxy", xh[:, -m:, :m], w[:, 1], alpha)
        out = jnp.zeros((w.shape[3], n, n // 2 + 1), xh.dtype)
        out = out.at[:, :m, :m].set(lo).at[:, -m:, :m].set(hi)
        return jnp.fft.irfft2(out, s=(n, n))


def fourier_features(t, n, fmax):
    f = jnp.geomspace(1.0, fmax, n)
    return jnp.concatenate([jnp.sin(math.pi * f * t), jnp.cos(math.pi * f * t)])


class CondFNO(eqx.Module):
    lift: eqx.nn.Conv2d
    spec: list
    skip: list
    p1: eqx.nn.Conv2d
    p2: eqx.nn.Conv2d
    cmlp: eqx.nn.MLP
    width: int = eqx.field(static=True)
    banks: int = eqx.field(static=True)
    n_freq: int = eqx.field(static=True)
    max_freq: float = eqx.field(static=True)

    def __init__(self, key, width=64, layers=4, modes=24, banks=4, dz=32, n_freq=16, max_freq=64.0,
                 cond_hidden=256, head=128):
        k = jax.random.split(key, 2 * layers + 4)
        self.width, self.banks, self.n_freq, self.max_freq = width, banks, n_freq, float(max_freq)
        self.lift = eqx.nn.Conv2d(C_IN, width, 1, key=k[0])
        self.spec = [SpectralMix(width, modes, banks, k[1 + i]) for i in range(layers)]
        self.skip = [eqx.nn.Conv2d(width, width, 1, key=k[1 + layers + i]) for i in range(layers)]
        self.p1, self.p2 = eqx.nn.Conv2d(width, head, 1, key=k[-3]), eqx.nn.Conv2d(head, F_MAX, 1, key=k[-2])
        mlp = eqx.nn.MLP(dz + 2 * n_freq + 2, layers * (2 * width + banks), cond_hidden, 2, activation=jax.nn.gelu, key=k[-1])
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: (l.weight, l.bias), last, (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)))
        self.cmlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)

    def __call__(self, u0, st, pres, aux, t, z, logT, logL):
        """One sample, one query time t in [0, 1] -> (F_MAX, N, N) normalized fields."""
        N = u0.shape[-1]
        g = jnp.arange(N, dtype=jnp.float32) / N
        X, Y = jnp.meshgrid(g, g)  # X varies along the last axis
        maps = lambda v: jnp.broadcast_to(v[:, None, None], (v.shape[0], N, N))
        x = jnp.concatenate([u0, st, maps(pres), jnp.zeros((1, N, N)), X[None], Y[None], maps(aux)])
        c = jnp.concatenate([z, fourier_features(t, self.n_freq, self.max_freq), jnp.stack([logT, logL]) / 4])
        mod = self.cmlp(c).reshape(len(self.spec), 2 * self.width + self.banks)
        h = self.lift(x)
        for i, (s, w) in enumerate(zip(self.spec, self.skip)):
            gam, bet, a = mod[i, : self.width], mod[i, self.width : 2 * self.width], mod[i, 2 * self.width :]
            h = s(h, jax.nn.softmax(a)) + w(h)
            h = (1 + gam)[:, None, None] * h + bet[:, None, None]
            h = jax.nn.gelu(h) if i < len(self.spec) - 1 else h
        return u0 + self.p2(jax.nn.gelu(self.p1(h)))


def batched(model, b, z):
    """b: batch dict with u0 st pres aux (B, ...), t (B, Q), logT logL (B,); z (B, dz) -> (B, Q, F_MAX, N, N)."""
    per_t = jax.vmap(model, in_axes=(None, None, None, None, 0, None, None, None))
    return jax.vmap(per_t)(b["u0"], b["st"], b["pres"], b["aux"], b["t"], z, b["logT"], b["logL"])

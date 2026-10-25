"""v3: amortized context-conditioned latent q(z | G, C) and a wider conditioned FNO (train_ctx.py).

ContextEncoder: one transformer over
  graph tokens   GraphEncoder.terms: psi(term) + field-slot embedding per additive term (before v1's sum), -> d
  sim tokens     each context window (frames CTX_FRAMES, IC-normalized as in data/adapter.normalize, 4x4-mean-pooled
                 to 32^2, asinh-squashed) -> 64 tubelet patch tokens (all frames, fields and statics of a 4x4 patch),
                 + spatial position + an embedding of its normalization stats (aux: log10 scale etc., presence,
                 log T, log L); 2 self-attention blocks, then `sim_tokens` learned queries cross-attend (Perceiver)
  readout        one learned query; its output -> mu (LayerNormed, as v1), log sigma (soft-clamped)
No ordering embedding across simulations, so q is permutation invariant over C; masks select C (any subset).

CtxFNO: v1's CondFNO input layout (C_IN, so eval.adapt_statics applies) with a wider conditioning MLP c -> per layer
FiLM + bank mixture + a rank-r channel-mixing delta  h += tanh A(c) (tanh B(c) h) / sqrt(width).  FiLM, mixture and A start at
zero (plain FNO at init); B keeps its random init so A gets gradients.  remat: jax.checkpoint per layer.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from data import adapter
from data.adapter import AUX_DIM
from data.canon import F_MAX, S_MAX
from models.encoder import GraphEncoder, _ln
from models.fno import C_IN, SpectralMix, fourier_features

CTX_FRAMES = (0, 1, 2, 4, 7, 10, 14)  # of a 15-frame window; frame 0 = the IC
CTX_N, PATCH = 32, 4
NPATCH = (CTX_N // PATCH) ** 2
P_IN = (len(CTX_FRAMES) * F_MAX + S_MAX) * PATCH * PATCH
STAT_IN = AUX_DIM + F_MAX + S_MAX + 2


def ctx_features(raw, statics=None):
    """raw (len(CTX_FRAMES), nf, N, N) physical frames (first = IC), statics (ns, N, N) or None -> encoder inputs:
    x (frames, F_MAX, 32, 32) f16 normalized by the IC (adapter.normalize), st (S_MAX, 32, 32) f16, aux, pres."""
    assert raw.shape[0] == len(CTX_FRAMES)
    n = adapter.normalize(raw[0], raw[1:], statics)
    x = np.concatenate([n["u0"][None], n["y"]])
    p = x.shape[-1] // CTX_N
    pool = lambda a: a.reshape(*a.shape[:-2], CTX_N, p, CTX_N, p).mean((-3, -1))
    x = np.nan_to_num(np.clip(pool(x), -6e4, 6e4))
    return {"x": x.astype(np.float16), "st": np.nan_to_num(pool(n["st"])).astype(np.float16), "aux": n["aux"], "pres": n["pres"]}


def kl_diag(mu1, ls1, mu2, ls2):
    """KL(N(mu1, e^{2 ls1}) || N(mu2, e^{2 ls2})) summed over the last axis."""
    return (ls2 - ls1 + (jnp.exp(2 * ls1) + (mu1 - mu2) ** 2) / (2 * jnp.exp(2 * ls2)) - 0.5).sum(-1)


class Block(eqx.Module):
    """Pre-LN transformer block (LayerNorm without affine); kv = None: self-attention."""

    att: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP

    def __init__(self, d, heads, key):
        k1, k2 = jax.random.split(key)
        self.att = eqx.nn.MultiheadAttention(heads, d, key=k1)
        self.mlp = eqx.nn.MLP(d, d, 4 * d, 1, activation=jax.nn.gelu, key=k2)

    def __call__(self, x, kv=None, mask=None):
        h = _ln(x)
        c = h if kv is None else _ln(kv)
        x = x + self.att(h, c, c, mask=mask)
        return x + jax.vmap(self.mlp)(_ln(x))


class ContextEncoder(eqx.Module):
    graph: GraphEncoder
    term: eqx.nn.Linear
    patch: eqx.nn.Linear
    stat: eqx.nn.Linear
    pos: jax.Array
    queries: jax.Array
    sim_blocks: list
    cross: Block
    typ: jax.Array
    readout: jax.Array
    agg: list
    head: eqx.nn.Linear
    dz: int = eqx.field(static=True)
    lo: float = eqx.field(static=True)
    hi: float = eqx.field(static=True)

    def __init__(self, key, dz=128, hidden=64, rounds=3, d=256, heads=8, sim_layers=2, agg_layers=4, sim_tokens=4,
                 logsig=(-4.0, 0.5)):
        k = jax.random.split(key, 10 + sim_layers + agg_layers)
        self.dz, self.lo, self.hi = dz, float(logsig[0]), float(logsig[1])
        self.graph = GraphEncoder(k[0], dz=dz, hidden=hidden, rounds=rounds, logsig=logsig)  # its rho is unused
        self.term = eqx.nn.Linear(hidden, d, key=k[1])
        self.patch = eqx.nn.Linear(P_IN, d, key=k[2])
        self.stat = eqx.nn.Linear(STAT_IN, d, key=k[3])
        self.pos = 0.02 * jax.random.normal(k[4], (NPATCH, d))
        self.queries = 0.02 * jax.random.normal(k[5], (sim_tokens, d))
        self.typ = 0.02 * jax.random.normal(k[6], (2, d))
        self.readout = 0.02 * jax.random.normal(k[7], (d,))
        self.cross = Block(d, heads, k[8])
        self.head = eqx.nn.Linear(d, 2 * dz, key=k[9])
        self.sim_blocks = [Block(d, heads, k[10 + i]) for i in range(sim_layers)]
        self.agg = [Block(d, heads, k[10 + sim_layers + i]) for i in range(agg_layers)]

    def sims(self, x, st, aux, pres, logT, logL):
        """One context window -> (sim_tokens, d)."""
        a = jnp.arcsinh(jnp.concatenate([x.reshape(-1, CTX_N, CTX_N), st]).astype(jnp.float32))
        g = CTX_N // PATCH
        a = a.reshape(a.shape[0], g, PATCH, g, PATCH).transpose(1, 3, 0, 2, 4).reshape(g * g, -1)
        s = self.stat(jnp.concatenate([aux, pres, jnp.stack([logT, logL]) / 4]))
        h = jax.vmap(self.patch)(a) + self.pos + s
        for b in self.sim_blocks:
            h = b(h)
        return self.cross(self.queries + s, h)

    def posterior(self, terms, tmask, toks, smask):
        """terms (T, d), toks (S, sim_tokens, d), masks -> mu, log sigma (dz,)."""
        tok = jnp.concatenate([self.readout[None], terms + self.typ[0], toks.reshape(-1, toks.shape[-1]) + self.typ[1]])
        m = jnp.concatenate([jnp.ones(1, bool), tmask > 0, jnp.repeat(smask > 0, toks.shape[1])])
        m = jnp.broadcast_to(m[None], (tok.shape[0], tok.shape[0]))
        for b in self.agg:
            tok = b(tok, mask=m)
        out = self.head(_ln(tok[0]))
        return _ln(out[: self.dz]), self.lo + (self.hi - self.lo) * jax.nn.sigmoid(out[self.dz :])

    def __call__(self, g, cx, cst, caux, pres, logT, logL, masks):
        """One equation: graph g, S context slots cx (S, frames, F_MAX, 32, 32), cst (S, S_MAX, 32, 32), caux (S, AUX),
        pres (F_MAX + S_MAX,), masks (P, S) -> mu, log sigma (P, dz): one posterior per mask row."""
        terms = jax.vmap(self.term)(self.graph.terms(g))
        toks = jax.vmap(lambda x, s, a: self.sims(x, s, a, pres, logT, logL))(cx, cst, caux)
        return jax.vmap(lambda m: self.posterior(terms, g["term_mask"], toks, m))(masks)


def adapt_statics_ctx(enc, ns):
    """Like eval.adapt_statics for the sim tokens: static slot 1 is never present in pretraining, so its patch and
    stat columns are untrained; copy slot 0's into them (SWE)."""
    if ns < 2:
        return enc
    pp, c0 = PATCH * PATCH, len(CTX_FRAMES) * F_MAX
    W = enc.patch.weight
    W = W.at[:, (c0 + 1) * pp : (c0 + 2) * pp].set(W[:, c0 * pp : (c0 + 1) * pp])
    Ws = enc.stat.weight
    for j0, j1 in ((2 * F_MAX, 2 * F_MAX + 1), (AUX_DIM + F_MAX, AUX_DIM + F_MAX + 1)):
        Ws = Ws.at[:, j1].set(Ws[:, j0])
    return eqx.tree_at(lambda e: (e.patch.weight, e.stat.weight), enc, (W, Ws))


class CtxFNO(eqx.Module):
    lift: eqx.nn.Conv2d
    spec: list
    skip: list
    p1: eqx.nn.Conv2d
    p2: eqx.nn.Conv2d
    cmlp: eqx.nn.MLP
    width: int = eqx.field(static=True)
    banks: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    n_freq: int = eqx.field(static=True)
    max_freq: float = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __init__(self, key, width=96, layers=6, modes=24, banks=4, dz=128, n_freq=16, max_freq=64.0, cond_hidden=512,
                 head=128, rank=8, remat=True):
        k = jax.random.split(key, 2 * layers + 4)
        self.width, self.banks, self.rank, self.n_freq, self.max_freq = width, banks, rank, n_freq, float(max_freq)
        self.remat = bool(remat)
        self.lift = eqx.nn.Conv2d(C_IN, width, 1, key=k[0])
        self.spec = [SpectralMix(width, modes, banks, k[1 + i]) for i in range(layers)]
        self.skip = [eqx.nn.Conv2d(width, width, 1, key=k[1 + layers + i]) for i in range(layers)]
        self.p1, self.p2 = eqx.nn.Conv2d(width, head, 1, key=k[-3]), eqx.nn.Conv2d(head, F_MAX, 1, key=k[-2])
        per = 2 * width + banks + 2 * width * rank
        mlp = eqx.nn.MLP(dz + 2 * n_freq + 2, layers * per, cond_hidden, 2, activation=jax.nn.gelu, key=k[-1])
        keep = jnp.arange(layers * per) % per >= 2 * width + banks + width * rank  # only B(c) keeps its random init
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: (l.weight, l.bias), last, (last.weight * keep[:, None], last.bias * keep))
        self.cmlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)

    def __call__(self, u0, st, pres, aux, t, z, logT, logL):
        """One sample, one query time t in [0, 1] -> (F_MAX, N, N) normalized fields (same signature as CondFNO)."""
        N, W, M, r = u0.shape[-1], self.width, self.banks, self.rank
        g = jnp.arange(N, dtype=jnp.float32) / N
        X, Y = jnp.meshgrid(g, g)
        maps = lambda v: jnp.broadcast_to(v[:, None, None], (v.shape[0], N, N))
        x = jnp.concatenate([u0, st, maps(pres), jnp.zeros((1, N, N)), X[None], Y[None], maps(aux)])
        c = jnp.concatenate([z, fourier_features(t, self.n_freq, self.max_freq), jnp.stack([logT, logL]) / 4])
        mod = self.cmlp(c).reshape(len(self.spec), -1)
        h = self.lift(x)
        last = len(self.spec) - 1
        for i, (s, w) in enumerate(zip(self.spec, self.skip)):
            def layer(h, m, s=s, w=w, i=i):
                gam, bet, a = m[:W], m[W : 2 * W], m[2 * W : 2 * W + M]
                A = jnp.tanh(m[2 * W + M : 2 * W + M + W * r]).reshape(W, r)  # bounded factors: the product of two
                B = jnp.tanh(m[2 * W + M + W * r :]).reshape(r, W)  # unbounded MLP outputs diverged (NaN at step 2700)
                h = s(h, jax.nn.softmax(a)) + w(h) + jnp.einsum("or,rxy->oxy", A, jnp.einsum("rc,cxy->rxy", B, h)) / math.sqrt(W)
                h = (1 + gam)[:, None, None] * h + bet[:, None, None]
                return jax.nn.gelu(h) if i < last else h

            h = jax.checkpoint(layer)(h, mod[i]) if self.remat else layer(h, mod[i])
        return u0 + self.p2(jax.nn.gelu(self.p1(h)))

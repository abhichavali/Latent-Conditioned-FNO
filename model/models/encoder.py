"""Graph encoder p_phi(z | G) = N(mu, diag sigma^2) over the canonical equation graph (data/canon.py).

Each additive term's expression tree gets R rounds of message passing (child -> parent with the pow argument
position, parent -> child), its nodes are sum-pooled to psi(term), a target-field-slot embedding is added, all terms
of all fields are SUMMED, and an MLP rho gives (mu, log sigma).  mu is LayerNormed (no affine, so it cannot shrink
to a collapsed point); log sigma is soft-clamped to [lo, hi].
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from data.canon import F_MAX, NODE_DIM


def _ln(x, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + eps)


class GraphEncoder(eqx.Module):
    emb: eqx.nn.Linear
    up: list
    down: list
    upd: list
    pos: jax.Array
    term: eqx.nn.MLP
    slot: jax.Array
    rho: eqx.nn.MLP
    dz: int = eqx.field(static=True)
    lo: float = eqx.field(static=True)
    hi: float = eqx.field(static=True)

    def __init__(self, key, dz=32, hidden=64, rounds=3, logsig=(-4.0, 0.5)):
        k = jax.random.split(key, 3 * rounds + 5)
        H = hidden
        self.dz, self.lo, self.hi = dz, float(logsig[0]), float(logsig[1])
        self.emb = eqx.nn.Linear(NODE_DIM, H, key=k[0])
        self.up = [eqx.nn.Linear(H, H, key=k[1 + i]) for i in range(rounds)]
        self.down = [eqx.nn.Linear(H, H, key=k[1 + rounds + i]) for i in range(rounds)]
        self.upd = [eqx.nn.MLP(3 * H, H, H, 1, activation=jax.nn.gelu, key=k[1 + 2 * rounds + i]) for i in range(rounds)]
        self.pos = 0.1 * jax.random.normal(k[-4], (2, H))
        self.term = eqx.nn.MLP(H, H, H, 1, activation=jax.nn.gelu, key=k[-3])
        self.slot = 0.1 * jax.random.normal(k[-2], (F_MAX, H))
        self.rho = eqx.nn.MLP(H, 2 * dz, 2 * H, 2, activation=jax.nn.gelu, key=k[-1])

    def __call__(self, g):
        """g: one equation's padded arrays (data/canon.pad row) -> mu (dz,), log sigma (dz,)."""
        nm, em = g["node_mask"][:, None], g["edge_mask"][:, None]
        n = nm.shape[0]
        child, parent, pos = g["edges"][:, 0], g["edges"][:, 1], g["edges"][:, 2]
        lin = lambda f, x: jax.vmap(f)(x)
        h = jax.nn.gelu(lin(self.emb, g["x"])) * nm
        for up, down, upd in zip(self.up, self.down, self.upd):
            m_up = jax.ops.segment_sum((lin(up, h[child]) + self.pos[pos]) * em, parent, n)
            m_dn = jax.ops.segment_sum(lin(down, h[parent]) * em, child, n)
            h = _ln(h + lin(upd, jnp.concatenate([h, m_up, m_dn], -1))) * nm
        psi = jax.ops.segment_sum(h, g["node_term"], g["term_mask"].shape[0])
        psi = (lin(self.term, psi) + self.slot[g["term_slot"]]) * g["term_mask"][:, None]
        out = self.rho(_ln(psi.sum(0)))
        mu = _ln(out[: self.dz])
        logsig = self.lo + (self.hi - self.lo) * jax.nn.sigmoid(out[self.dz :])
        return mu, logsig

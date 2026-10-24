"""Coefficient templates: an n-field equation is a coefficient matrix (n, K) over a fixed term library, so one
compile per template serves every equation, vmapped over equations and initial conditions.

Term kinds per target field f (symbols: u v w fields, a static, ux = dx(u) ..., gu = |grad u|^2,
G = (1 + tanh(kap (u - th)))/2 gate, R = 1/(|u| + h) saturation, S = sin(u)):
  L  constant-coefficient linear operator on f (exact/stiff in ETDRK): 1 dx dy dxx dyy bih dxlap dylap
  P  pointwise monomial            X / Y  monomial inside dx / dy      Q  monomial inside lap
  D  div(mob grad pot) for (mob, pot) pairs
X, Y and Q terms are summed before one FFT each, so evaluation cost does not grow with the number of terms.
"""

import itertools
import math
import re
from functools import partial

import equinox as eqx
import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from exponax.nonlin_fun import BaseNonlinearFun

FIELDS = ("u", "v", "w")
OPS = ("1", "dx", "dy", "dxx", "dyy", "bih", "dxlap", "dylap")
OP_STR = {"1": "{f}", "dx": "dx({f})", "dy": "dy({f})", "dxx": "dx(dx({f}))", "dyy": "dy(dy({f}))",
          "bih": "lap(lap({f}))", "dxlap": "dx(lap({f}))", "dylap": "dy(lap({f}))"}
SYM = {n: sp.Symbol(n) for n in "u v w a ux uy vx vy wx wy gu gv gw G R S".split()}


def monomials(F, lo, hi):
    return ["*".join(c) or "1" for d in range(lo, hi + 1) for c in itertools.combinations_with_replacement(F, d)]


def library(nf):
    """Slots shared by every target field; '{f}' is the target.  Returns [(kind, item)]."""
    F = FIELDS[:nf]
    uv = ["u*v"] if nf > 1 else []
    P = (monomials(F, 0, 3) + ["G"] + [f"G*{g}" for g in F] + [f"G*{m}" for m in uv]
         + ["R", "R*u*u"] + [f"R*{g}" for g in F] + [f"R*{m}" for m in uv + (["v*v"] if nf > 1 else [])]
         + ["S"] + [f"g{g}" for g in F] + [f"{g}*{{f}}x" for g in F] + [f"{g}*{{f}}y" for g in F]
         + ["a"] + [f"a*{g}" for g in F])
    XY = monomials(F, 1, 2) + [f"a*{g}" for g in F]
    return ([("L", o) for o in OPS] + [("P", p) for p in P] + [("X", m) for m in XY] + [("Y", m) for m in XY]
            + [("Q", q) for q in list(F) + ["{f}**3"]] + [("D", d) for d in [("a", "{f}")] + [("{f}", g) for g in F]])


LIB = {nf: library(nf) for nf in (2, 3)}
K = {nf: len(lib) for nf, lib in LIB.items()}


def _index(nf, f):
    """sympy monomial -> slot, per kind, for target f."""
    idx = {}
    for k, (kind, item) in enumerate(LIB[nf]):
        if kind in "PXYQ":
            idx[(kind, sp.sympify(item.format(f=f), locals=SYM))] = k
    return idx


def encode(eq, nf):
    """eq["terms"][f] = {"L": {op: c}, "P"/"X"/"Y"/"Q": sympy expr, "D": {(mob, pot): c}} -> (nf, K) array."""
    C = np.zeros((nf, K[nf]))
    slot = {(kind, item): k for k, (kind, item) in enumerate(LIB[nf])}
    for i, f in enumerate(eq["fields"]):
        t, idx = eq["terms"][f], _index(nf, f)
        for op, c in t.get("L", {}).items():
            C[i, slot[("L", op)]] += c
        for kind in "PXYQ":
            for m, c in sp.expand(t.get(kind, 0)).as_coefficients_dict().items():
                if m == SYM[f]:  # self-linear -> exact operator
                    for op in {"P": ["1"], "X": ["dx"], "Y": ["dy"], "Q": ["dxx", "dyy"]}[kind]:
                        C[i, slot[("L", op)]] += float(c)
                elif c != 0:
                    C[i, idx[(kind, m)]] += float(c)  # KeyError = term outside the library
        for (mob, pot), c in t.get("D", {}).items():  # div(a grad f) or div(f grad g)
            assert pot == f if mob == "a" else mob == f, (f, mob, pot)
            C[i, slot[("D", ("a", "{f}") if mob == "a" else ("{f}", pot))]] += c
    return C


def _fmt(x):
    return f"{x:.6g}"


def rhs_strings(fields, C, prm, nf):
    """Coefficient matrix -> {field: sympy-parseable rhs} for pde.py / the graph."""
    kap, th, h = prm
    sub = [(r"\b([uvw])([xy])\b", r"d\2(\1)"), (r"\bg([uvw])\b", r"(dx(\1)**2 + dy(\1)**2)"),
           (r"\bG\b", f"(1 + tanh({_fmt(kap)}*(u - {_fmt(th)})))/2"), (r"\bR\b", f"1/(Abs(u) + {_fmt(h)})"),
           (r"\bS\b", "sin(u)")]
    out = {}
    for i, f in enumerate(fields):
        c = dict(enumerate(C[i]))
        terms = []
        lin = {op: c[k] for k, (kind, op) in enumerate(LIB[nf]) if kind == "L"}
        if lin["dxx"] != 0 and lin["dxx"] == lin["dyy"]:  # isotropic diffusion reads as lap
            terms.append(f"{_fmt(lin.pop('dxx'))}*lap({f})")
            lin.pop("dyy")
        terms += [f"{_fmt(v)}*{OP_STR[op].format(f=f)}" for op, v in lin.items() if v != 0]
        for k, (kind, item) in enumerate(LIB[nf]):
            if kind == "L" or c[k] == 0:
                continue
            if kind == "D":
                mob, pot = (s.format(f=f) for s in item)
                s = f"(dx({mob}*dx({pot})) + dy({mob}*dy({pot})))"
            else:
                s = item.format(f=f)
                for pat, rep in sub:
                    s = re.sub(pat, rep, s)
                s = {"P": "({})", "X": "dx({})", "Y": "dy({})", "Q": "lap({})"}[kind].format(s)
            terms.append(f"{_fmt(c[k])}*{s}")
        out[f] = " + ".join(terms) or "0"
    return out


def _k(N, L):
    s = 2 * jnp.pi / L
    return 1j * s * jnp.fft.rfftfreq(N, 1 / N)[None, :], 1j * s * jnp.fft.fftfreq(N, 1 / N)[:, None]


class Nonlin(BaseNonlinearFun):
    c: jax.Array
    prm: jax.Array
    kx: jax.Array
    ky: jax.Array
    nf: int = eqx.field(static=True)

    def __init__(self, c, prm, L, nf, N):
        super().__init__(2, N, dealiasing_fraction=2 / 3)
        self.c, self.prm, self.nf = c, prm, nf
        self.kx, self.ky = _k(N, L)

    def __call__(self, uh):  # (nf + 1, N, N//2+1); last channel is the static field
        nf, F = self.nf, FIELDS[: self.nf]
        f = self.ifft(uh)
        fx, fy = self.ifft(self.kx * uh[:nf]), self.ifft(self.ky * uh[:nf])
        env = {"a": f[nf], "jnp": jnp}
        for i, g in enumerate(F):
            env.update({g: f[i], g + "x": fx[i], g + "y": fy[i], "g" + g: fx[i] ** 2 + fy[i] ** 2})
        kap, th, h = self.prm
        env.update(G=0.5 * (1 + jnp.tanh(kap * (f[0] - th))), R=1 / (jnp.abs(f[0]) + h), S=jnp.sin(f[0]))
        memo = {}

        def ev(s):
            if s not in memo:
                memo[s] = eval(s, env)  # library strings only
            return memo[s]

        out = []
        for i, g in enumerate(F):
            acc = {k: jnp.zeros_like(f[0]) for k in "PXYQ"}
            for k, (kind, item) in enumerate(LIB[nf]):
                if kind == "L":
                    continue
                c = self.c[i, k]
                if kind == "D":
                    mob, pot = (s.format(f=g) for s in item)
                    acc["X"] += c * ev(mob) * ev(pot + "x")
                    acc["Y"] += c * ev(mob) * ev(pot + "y")
                else:
                    acc[kind] += c * ev(item.format(f=g))
            P, X, Y, Q = self.fft(jnp.stack([acc[k] for k in "PXYQ"]))
            out.append(P + self.kx * X + self.ky * Y + (self.kx**2 + self.ky**2) * Q)
        return jnp.concatenate([jnp.stack(out), jnp.zeros_like(uh[nf:])])


def linear_operator(c, L, nf, N):
    kx, ky = _k(N, L)
    lap = kx**2 + ky**2
    sym = {"1": 1.0, "dx": kx, "dy": ky, "dxx": kx**2, "dyy": ky**2, "bih": lap**2, "dxlap": kx * lap, "dylap": ky * lap}
    ops = [sum(c[i, k] * sym[op] for k, (kind, op) in enumerate(LIB[nf]) if kind == "L") + 0 * lap for i in range(nf)]
    return jnp.concatenate([jnp.stack(ops), jnp.zeros_like(lap)[None]])


def _one(c, prm, L, dt, sched, u0, shift, scale, nf, N):
    """One equation: u0 (B, nf+1, N, N), sched (F-1,) substeps per frame -> (B, F, nf, N, N) float16 of
    (u - shift) / scale.  sched entries may differ per equation; vmap masks the shorter loops."""
    integ = ex.etdrk.ETDRK2(dt, linear_operator(c, L, nf, N), Nonlin(c, prm, L, nf, N))
    step = jax.vmap(integ.step_fourier)
    norm = lambda uh: ((jnp.fft.irfft2(uh[:, :nf], s=(N, N)) - shift[:, None, None]) / scale[:, None, None]).astype(jnp.float16)

    def frame(uh, s):
        uh = jax.lax.fori_loop(0, s, lambda _, x: step(x), uh)
        return uh, norm(uh)

    uh0 = jnp.fft.rfft2(u0)
    _, traj = jax.lax.scan(frame, uh0, sched)
    return jnp.concatenate([norm(uh0)[:, None], jnp.moveaxis(traj, 0, 1)], 1)


@partial(jax.jit, static_argnames=("nf", "N"))
def rollout(c, prm, L, dt, sched, u0, shift, scale, *, nf, N):
    """Batched over equations: c (E, nf, K), prm (E, 3), L dt (E,), sched (E, F-1) int, u0 (E, B, nf+1, N, N),
    shift scale (E, nf) -> (E, B, F, nf, N, N) float16."""
    return jax.vmap(partial(_one, nf=nf, N=N))(c, prm, L, dt, sched, u0, shift, scale)


if __name__ == "__main__":  # template == generic sympy solver on a random dense coefficient matrix
    import json

    import pde

    rng = np.random.default_rng(0)
    for nf in (2, 3):
        N, Lx = 32, 2 * math.pi * 3
        C = rng.normal(0, 0.05, (nf, K[nf])) * (rng.random((nf, K[nf])) < 0.5)
        for i in range(nf):  # keep it well-posed: diffusion + mild hyperdiffusion
            C[i, OPS.index("dxx")] = C[i, OPS.index("dyy")] = 0.5
            C[i, OPS.index("bih")] = -0.01
        for i, f in enumerate(FIELDS[:nf]):  # self-linear slots are folded into L by encode(); mirror that
            for kind in "PXYQ":
                C[i, LIB[nf].index((kind, f))] = 0
        prm = np.array([4.0, 0.2, 0.5])
        fields = list(FIELDS[:nf])
        spec = {"fields": fields, "static": ["a"], "rhs": rhs_strings(fields, C, prm, nf), "L": Lx, "N": N}
        X, Y = pde.grid(spec)
        u0 = jnp.stack([0.5 * jnp.sin(X * (i + 1) * 2 * math.pi / Lx + Y * 2 * math.pi / Lx) for i in range(nf)]
                       + [1 + 0.3 * jnp.cos(Y * 2 * math.pi / Lx)])[None]
        ref = pde.rollout(pde.solver_key(spec), u0, 0.01, 20, 4)[0, :, :nf]
        got = rollout(jnp.asarray(C)[None], jnp.asarray(prm)[None], jnp.array([Lx]), jnp.array([0.01]),
                      jnp.full((1, 3), 20), u0[None], jnp.zeros((1, nf)), jnp.ones((1, nf)), nf=nf, N=N)[0, 0]
        err = float(jnp.abs(got.astype(jnp.float32) - ref).max())
        assert err < 5e-3 * float(jnp.abs(ref).max()), err  # float16 output rounding
        g = pde.graph(spec)
        print(f"nf={nf}: K={K[nf]}, max err {err:.2e}, graph {len(g['nodes'])} nodes")
    # encode round trip: sampler-style expressions land in the right slots, self-linear terms go exact
    u, v = SYM["u"], SYM["v"]
    eq = {"fields": ["u", "v"], "terms": {"u": {"L": {"dxx": 1.0, "dyy": 1.0}, "P": 8 * u * (1 - u) * (u - 0.1) - u * v},
                                          "v": {"P": 0.01 * (u - v), "X": -u * v, "D": {("a", "v"): 0.2}}}}
    C = encode(eq, 2)
    s = rhs_strings(["u", "v"], C, (1, 0, 1), 2)
    assert C[0, OPS.index("1")] == -0.8 and C[1, OPS.index("1")] == -0.01, s
    print(json.dumps(s))
    print("template.py ok")

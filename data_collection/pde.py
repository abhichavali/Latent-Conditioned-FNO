"""PDE spec format, its computational graph, and one generic exponax solver for all of them.

A spec is a JSON-able dict:
  fields   evolving scalar fields u(x, y, t), e.g. ["u", "v"]
  static   optional time-independent fields a(x, y) (coefficients, bathymetry), resampled per IC
  rhs      {field: "sympy string"} = d(field)/dt.  Operators: dx dy lap, + - * / **,
           sin cos exp tanh log Abs.  Every field (evolving or static) is a symbol.
  L, N     periodic square [0, L)^2 on an N x N grid.  Arrays are (..., Ny, Nx): dx acts on the last axis
  dt, substeps, frames   solver step, steps per saved frame, saved frames (frame 0 = initial condition)
  ic       {field: {"mean", "amp", "mode"}} initial-condition ranges (see sample.py)

Constant-coefficient self terms (c * D field, D a chain of dx/dy/lap) go into exponax's stiff diagonal
ETDRK operator; everything else is evaluated pseudo-spectrally with 2/3 dealiasing.
"""

import json
import math
from functools import partial

import equinox as eqx
import exponax as ex
import jax
import jax.numpy as jnp
import sympy as sp
from exponax.nonlin_fun import BaseNonlinearFun

class _Op(sp.Function):
    nargs = 1

    def _latex(self, p):
        a = self.args[0]
        return f"{self.sym} {p._print(a)}" if a.is_Symbol or isinstance(a, _Op) else rf"{self.sym}\left({p._print(a)}\right)"


class dx(_Op):
    sym = r"\partial_x"


class dy(_Op):
    sym = r"\partial_y"


class lap(_Op):
    sym = r"\nabla^2"


OPS = {"dx": dx, "dy": dy, "lap": lap}
FUNCS = {"sin": jnp.sin, "cos": jnp.cos, "exp": jnp.exp, "tanh": jnp.tanh, "log": jnp.log, "Abs": jnp.abs}


def parse(spec):
    names = spec["fields"] + spec.get("static", [])
    syms = {n: sp.Symbol(n) for n in names}
    rhs = [sp.sympify(spec["rhs"][f], locals={**syms, **OPS}) for f in spec["fields"]]
    return [syms[n] for n in names], rhs


def graph(spec):
    """Expression DAG over all right-hand sides; identical subexpressions share one node.

    nodes: {"op": field|static|const|add|mul|pow|dx|dy|lap|sin|...|dt, "i" or "value"}
    edges: [child, parent, arg position] (position only matters for pow: base 0, exponent 1)
    One "dt" root per evolving field i, fed by its right-hand side.
    """
    syms, rhs = parse(spec)
    nf = len(spec["fields"])
    nodes, edges, ids = [], [], {}

    def visit(e):
        if e in ids:
            return ids[e]
        if e.is_Symbol:
            i = syms.index(e)
            node = {"op": "field", "i": i} if i < nf else {"op": "static", "i": i - nf}
        elif e.is_number:
            node = {"op": "const", "value": float(e)}
        else:
            node = {"op": type(e).__name__.lower()}
        kids = [] if e.is_Atom else [visit(a) for a in e.args]
        ids[e] = len(nodes)
        nodes.append(node)
        edges.extend([k, ids[e], p if node["op"] == "pow" else 0] for p, k in enumerate(kids))
        return ids[e]

    for i, r in enumerate(rhs):
        root = visit(r)
        nodes.append({"op": "dt", "i": i})
        edges.append([root, len(nodes) - 1, 0])
    return {"nodes": nodes, "edges": edges}


def latex(spec):
    syms, rhs = parse(spec)
    return {f: sp.latex(r) for f, r in zip(spec["fields"], rhs)}


def linear_part(expr, field):
    """Split expr into constant-coefficient self terms [(c, [ops])] and the remaining expression."""
    lin, rest = [], []
    for t in sp.Add.make_args(sp.expand(expr)):
        c, m = t.as_coeff_Mul()
        ops = []
        while type(m).__name__ in OPS:
            ops.append(type(m).__name__)
            m = m.args[0]
        if m == field and c.is_number:
            lin.append((float(c), tuple(ops)))
        else:
            rest.append(t)
    return tuple(lin), sp.Add(*rest)


def _symbols(deriv):  # exponax derivative operator is i*k with k[0] along axis -2 (y), k[1] along axis -1 (x)
    return {"dx": deriv[1], "dy": deriv[0], "lap": deriv[0] ** 2 + deriv[1] ** 2}


class Nonlin(BaseNonlinearFun):
    exprs: tuple = eqx.field(static=True)
    syms: tuple = eqx.field(static=True)
    k: dict

    def __init__(self, exprs, syms, deriv, num_points):
        super().__init__(2, num_points, dealiasing_fraction=2 / 3)
        self.exprs, self.syms, self.k = exprs, syms, _symbols(deriv)

    def __call__(self, u_hat):
        u = self.ifft(u_hat)
        memo = {}

        def ev(e):
            if e in memo:
                return memo[e]
            if e.is_Symbol:
                i = self.syms.index(e)
                r = u[i : i + 1]
            elif e.is_number:
                r = float(e)
            elif e.is_Add:
                r = sum(ev(a) for a in e.args)
            elif e.is_Mul:
                r = math.prod(ev(a) for a in e.args)
            elif e.is_Pow:
                r = ev(e.base) ** (int(e.exp) if e.exp.is_Integer else ev(e.exp))
            elif type(e).__name__ in OPS:
                a = e.args[0]
                i = self.syms.index(a) if a.is_Symbol else None
                a_hat = u_hat[i : i + 1] if a.is_Symbol else self.fft(ev(a))
                r = self.ifft(self.k[type(e).__name__] * a_hat)
            else:
                r = FUNCS[type(e).__name__](ev(e.args[0]))
            memo[e] = r
            return r

        zero = jnp.zeros_like(u_hat[:1])  # (1, N, N//2+1): exponax's dealias mask keeps a channel axis
        out = [zero if e == 0 else self.fft(ev(e) + jnp.zeros_like(u[:1])) for e in self.exprs]
        return jnp.concatenate(out + [zero] * (len(self.syms) - len(out)))


class Stepper(ex.BaseStepper):
    lin: tuple = eqx.field(static=True)
    nonlin: tuple = eqx.field(static=True)
    syms: tuple = eqx.field(static=True)

    def __init__(self, spec, dt, order=2):
        syms, rhs = parse(spec)
        split = [linear_part(r, s) for r, s in zip(rhs, syms)]
        self.syms = tuple(syms)
        self.lin = tuple(l for l, _ in split)
        self.nonlin = tuple(n for _, n in split)
        super().__init__(
            num_spatial_dims=2, domain_extent=spec["L"], num_points=spec["N"], dt=dt,
            num_channels=len(syms), order=order,
        )

    def _build_linear_operator(self, deriv):
        k = _symbols(deriv)
        zero = jnp.zeros(deriv.shape[1:], deriv.dtype)
        ops = [sum((c * math.prod(k[o] for o in ops) for c, ops in l), zero) for l in self.lin]
        return jnp.stack(ops + [zero] * (len(self.syms) - len(ops)))

    def _build_nonlinear_fun(self, deriv):
        return Nonlin(self.nonlin, self.syms, deriv, self.num_points)


@partial(jax.jit, static_argnums=(0, 4))
def rollout(spec_json, u0, dt, substeps, frames):
    """u0 (B, C, N, N) -> (B, frames, C, N, N), frame 0 = u0.  dt and substeps are traced, so one
    compile per (equation, batch shape, frames) serves every timestep size."""
    spec = json.loads(spec_json)
    step = jax.vmap(Stepper(spec, dt, spec.get("order", 2)).step_fourier)

    def frame(uh, _):
        uh = jax.lax.fori_loop(0, substeps, lambda _, x: step(x), uh)
        return uh, jnp.fft.irfft2(uh, s=u0.shape[-2:])

    _, traj = jax.lax.scan(frame, jnp.fft.rfft2(u0), None, length=frames - 1)
    return jnp.concatenate([u0[:, None], jnp.moveaxis(traj, 0, 1)], 1)


def solver_key(spec):
    """Only what changes the compiled program: dt and substeps are traced, so they are left out."""
    return json.dumps({k: spec[k] for k in ("fields", "static", "rhs", "L", "N", "order") if k in spec}, sort_keys=True)


def simulate(spec, u0):
    return rollout(solver_key(spec), u0, spec["dt"], spec["substeps"], spec["frames"])


def grid(spec):
    x = jnp.arange(spec["N"]) * spec["L"] / spec["N"]
    return jnp.meshgrid(x, x)  # X varies along the last axis


if __name__ == "__main__":  # self-checks against exact solutions
    base = {"L": 2 * math.pi, "N": 32, "dt": 0.01, "substeps": 10, "frames": 6, "order": 2}
    X, Y = grid(base)

    s = {**base, "fields": ["u"], "rhs": {"u": "0.3*lap(u)"}}  # diffusion of a mode: exact decay
    tr = simulate(s, jnp.sin(2 * X + Y)[None, None])
    assert jnp.allclose(tr[0, -1, 0], jnp.exp(-0.3 * 5 * 0.5) * jnp.sin(2 * X + Y), atol=1e-5)

    s = {**base, "fields": ["u"], "rhs": {"u": "u*(1 - u)"}}  # logistic ODE (all-nonlinear path)
    tr = simulate(s, jnp.full((1, 1, 32, 32), 0.1))
    exact = 1 / (1 + 9 * math.exp(-0.5))
    assert abs(float(tr[0, -1, 0, 0, 0]) - exact) < 1e-5, (tr[0, -1, 0, 0, 0], exact)

    # explicit cross-coupled waves u_t=-dx(v), v_t=-dx(u): u+v advects right at speed 1; static a untouched
    s = {**base, "fields": ["u", "v"], "static": ["a"], "rhs": {"u": "-dx(v)", "v": "-dx(u) + 0*a"}, "order": 4}
    u0 = jnp.stack([jnp.sin(X), jnp.sin(X), jnp.cos(Y)])[None]
    tr = simulate(s, u0)
    assert jnp.allclose(tr[0, -1, 0], jnp.sin(X - 0.5), atol=1e-4)
    assert jnp.array_equal(tr[0, -1, 2], u0[0, 2]) or jnp.allclose(tr[0, -1, 2], u0[0, 2], atol=1e-6)

    s = {**base, "fields": ["u"], "rhs": {"u": "-dx(u**2)/2 + 0.1*lap(u)"}}  # Burgers: mean is conserved
    tr = simulate(s, jnp.sin(X)[None, None] + 0.5)
    assert abs(float(tr[0, -1].mean()) - 0.5) < 1e-5

    assert latex({"fields": ["u"], "rhs": {"u": "lap(u) + dx(u*u)"}})["u"] == r"\partial_x\left(u^{2}\right) + \nabla^2 u"
    g = graph({"fields": ["u"], "rhs": {"u": "lap(u) + u*(1 - u)"}})
    assert [n["op"] for n in g["nodes"]].count("field") == 1 and g["nodes"][-1] == {"op": "dt", "i": 0}
    print("pde.py ok")

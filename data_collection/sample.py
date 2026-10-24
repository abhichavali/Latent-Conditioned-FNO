"""Random equations in the template vocabulary, GPU initial-condition pools, and max-entropy IC selection.

Families (weights): transport/waves near SWE 0.35, excitable/reaction-diffusion near Aliev-Panfilov 0.35,
pattern formation 0.2, free composition 0.1; then 0-2 random extra library terms.  Coefficients are drawn
relative to the grid (diffusion ~ (few dx)^2 * rate, viscosity ~ U dx) so fronts and shocks stay resolved.
Holdout: the exact SWE core (flux div(h u) + material advection + g grad h) and the Aliev-Panfilov core
(cubic kinetics + -uv + v/(u+mu) gate) are never emitted.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from template import FIELDS, LIB, SYM

N = 128
u, v, w, a = (SYM[k] for k in "uvwa")
ux, uy, vx, vy, wx, wy = (SYM[k] for k in "ux uy vx vy wx wy".split())
G, R, S, gu = SYM["G"], SYM["R"], SYM["S"], SYM["gu"]
MODES = {"signed": 0, "patch": 1, "any": 2}


def lu(r, lo, hi):
    return float(math.exp(r.uniform(math.log(lo), math.log(hi))))


def new(r, family, n, L):
    fs = list(FIELDS[:n])
    return {"family": family, "fields": fs, "L": float(L), "tags": set(), "static": None, "dt0": 0.1,
            "prm": [lu(r, 2, 20), float(r.uniform(-0.5, 0.5)), lu(r, 0.1, 1)],
            "ic": {f: {"mean": 0.0, "amp": 1.0, "mode": "any"} for f in fs},
            "terms": {f: {"L": {}, "P": 0, "X": 0, "Y": 0, "Q": 0, "D": {}} for f in fs}}


def lin(e, f, op, c):
    e["terms"][f]["L"][op] = e["terms"][f]["L"].get(op, 0.0) + float(c)


def diff(e, f, D, Dy=None):
    lin(e, f, "dxx", D)
    lin(e, f, "dyy", D if Dy is None else Dy)


def add(e, f, kind, expr):
    e["terms"][f][kind] = e["terms"][f][kind] + expr


def ic(e, f, mean, amp, mode="any"):
    e["ic"][f] = {"mean": float(mean), "amp": float(amp), "mode": mode}


# ---------------------------------------------------------------- transport / waves (near shallow water)
def transport(r):
    L = lu(r, 5, 50)
    dx = L / N
    n = int(r.choice([1, 2, 3], p=[0.2, 0.35, 0.45]))

    def visc(e, f, U):
        if r.random() < 0.3:
            lin(e, f, "bih", -lu(r, 0.3, 2) * U * dx**3)
        else:
            diff(e, f, lu(r, 0.3, 2) * U * dx)

    if n == 3:
        e = new(r, "transport/shallow-like", 3, L)
        H, g = lu(r, 0.5, 2), lu(r, 0.5, 2)
        c = math.sqrt(g * H)
        flux, adv = r.random() < 0.4, r.random() < 0.4
        if flux:
            add(e, "u", "X", -u * v), add(e, "u", "Y", -u * w), ic(e, "u", H, 0.15 * H)
            e["tags"].add("flux")
        elif r.random() < 0.3:  # linear waves over a variable medium (bathymetry-like)
            add(e, "u", "X", -a * v), add(e, "u", "Y", -a * w), ic(e, "u", 0, 0.3 * H)
            e["static"] = {"mean": H, "amp": 0.4 * H}
            e["tags"].add("medium")
        else:
            add(e, "u", "X", -H * v), add(e, "u", "Y", -H * w), ic(e, "u", 0, 0.3 * H)
        if flux and r.random() < 0.3:
            add(e, "v", "X", -g / (2 * H) * u**2), add(e, "w", "Y", -g / (2 * H) * u**2)
            e["tags"].add("grad_sq")
        else:
            add(e, "v", "X", -g * u), add(e, "w", "Y", -g * u)
            e["tags"].add("grad_lin")
        if adv:
            add(e, "v", "P", -v * vx - w * vy), add(e, "w", "P", -v * wx - w * wy)
            e["tags"].add("adv")
        if r.random() < 0.35:
            fc = lu(r, 0.5, 5) * 2 * math.pi * c / L
            add(e, "v", "P", fc * w), add(e, "w", "P", -fc * v)
            e["tags"].add("coriolis")
        if r.random() < 0.3:
            rd = lu(r, 0.01, 0.3) * 2 * math.pi * c / L
            lin(e, "v", "1", -rd), lin(e, "w", "1", -rd)
            e["tags"].add("drag")
        for f in "vw":
            ic(e, f, 0, lu(r, 0.05, 0.3) * c, "signed")
        for f in "uvw":
            visc(e, f, c)
        e["dt0"] = 0.2 * dx / c
        return e
    kind = r.choice(["wave", "burgers", "kdv"], p=[0.4, 0.35, 0.25]) if n == 2 else r.choice(["burgers1", "kdv1"])
    if kind == "wave":
        e = new(r, "transport/wave", 2, L)
        c = lu(r, 0.5, 2)
        add(e, "u", "P", v), add(e, "v", "Q", c * c * u)
        if r.random() < 0.4:
            lin(e, "v", "1", -lu(r, 0.01, 0.3)), e["tags"].add("damped")
        x = r.random()
        if x < 0.3:
            add(e, "v", "P", -(lu(r, 0.5, 4) * 2 * math.pi * c / L) ** 2 * u)
            e["tags"].add("klein-gordon")
        elif x < 0.6:
            add(e, "v", "P", -lu(r, 0.2, 2) * S), e["tags"].add("sine-gordon")
        elif x < 0.8:
            k = lu(r, 0.2, 2)
            add(e, "v", "P", k * (u - u**3)), e["tags"].add("phi4")
        elif x < 0.9:
            m2 = (lu(r, 0.5, 4) * 2 * math.pi * c / L) ** 2
            add(e, "v", "P", -a * u), e.update(static={"mean": m2, "amp": 0.5 * m2}), e["tags"].add("variable-mass")
        for f in "uv":
            diff(e, f, lu(r, 0.3, 2) * c * dx)
        ic(e, "u", 0, lu(r, 0.3, 1.5) * (2 if "sine-gordon" in e["tags"] else 1))
        ic(e, "v", 0, 0.5 * c, "signed")
        e["dt0"] = 0.2 * dx / c
        return e
    if kind.startswith("burgers"):
        e = new(r, "transport/burgers", n, L)
        U = lu(r, 0.3, 1.5)
        if n == 1:
            add(e, "u", "P", -lu(r, 0.5, 1) * u * ux - lu(r, 0.2, 1) * u * uy)
        elif r.random() < 0.3:  # conservative momentum form
            add(e, "u", "X", -u * u / 2), add(e, "u", "Y", -u * v), add(e, "v", "X", -u * v), add(e, "v", "Y", -v * v / 2)
            e["tags"].add("conservative")
        else:
            add(e, "u", "P", -u * ux - v * uy), add(e, "v", "P", -u * vx - v * vy)
            e["tags"].add("adv")
        if n == 2 and r.random() < 0.3:
            fc = lu(r, 0.5, 5) * 2 * math.pi * U / L
            add(e, "u", "P", fc * v), add(e, "v", "P", -fc * u), e["tags"].add("coriolis")
        for f in e["fields"]:
            diff(e, f, U * dx * lu(r, 0.5, 2))
            ic(e, f, 0, U, "signed")
            if r.random() < 0.3:
                lin(e, f, "1", -lu(r, 0.01, 0.2) * 2 * math.pi * U / L)
        e["dt0"] = 0.2 * dx / U
        return e
    e = new(r, "transport/kdv", n, L)  # KdV / Zakharov-Kuznetsov scalar, plus a passive tracer when n == 2
    A, c = lu(r, 0.3, 1.5), lu(r, 0.05, 1)
    lin(e, "u", "dx", -c)
    add(e, "u", "X", -lu(r, 0.5, 2) * u * u / 2)
    lin(e, "u", "dxlap", -A * (lu(r, 1.5, 4) * dx) ** 2)
    if r.random() < 0.5:
        lin(e, "u", "dylap", -A * (lu(r, 1.5, 4) * dx) ** 2)
    diff(e, "u", lu(r, 0.2, 1) * A * dx)
    ic(e, "u", 0, A)
    if n == 2:
        add(e, "v", "P", -u * vx - r.uniform(0, 1) * u * vy), diff(e, "v", lu(r, 0.3, 2) * A * dx), ic(e, "v", 0, 1)
        e["tags"].add("tracer")
    e["dt0"] = 0.2 * dx / (c + A)
    return e


# ---------------------------------------------------------------- excitable / reaction-diffusion (near Aliev-Panfilov)
EXCITABLE = ("fhn", "barkley", "nagumo", "ms", "oregonator")
KIN_P = {"fhn": 0.14, "barkley": 0.12, "nagumo": 0.16, "ms": 0.12, "oregonator": 0.08,
         "grayscott": 0.1, "brusselator": 0.08, "schnakenberg": 0.08, "predprey": 0.12}


def excitable(r):
    L = lu(r, 10, 200)
    dx = L / N
    kin = str(r.choice(list(KIN_P), p=list(KIN_P.values())))
    n = 3 if kin in EXCITABLE + ("predprey",) and r.random() < 0.3 else 2
    e = new(r, f"excitable/{kin}", n, L)
    e["tags"].add(kin)
    Dv_ratio = None
    if kin == "fhn":
        eps, A, B = lu(r, 0.01, 0.1), r.uniform(0.5, 1.0), r.uniform(0.4, 0.9)
        add(e, "u", "P", u - u**3 / 3 - v), add(e, "v", "P", eps * (u + A - B * v))
        ic(e, "u", -1.2, 2.5, "patch"), ic(e, "v", -0.6, 1.0)
        rate = 1.0
    elif kin == "barkley":
        eps, A, B = lu(r, 0.01, 0.08), r.uniform(0.5, 0.9), r.uniform(0.01, 0.08)
        add(e, "u", "P", u * (1 - u) * (u - (v + B) / A) / eps), add(e, "v", "P", u - v)
        ic(e, "u", 0, 1, "patch"), ic(e, "v", 0, r.uniform(0.3, 1), "patch")
        rate = 1 / eps
    elif kin == "nagumo":
        k, th = r.uniform(5, 10), r.uniform(0.05, 0.25)
        add(e, "u", "P", k * u * (1 - u) * (u - th))
        x = r.random()
        if x < 0.4:
            add(e, "u", "P", -u * v), e["tags"].add("uv")
        elif x < 0.8:
            add(e, "u", "P", -v)
        else:
            add(e, "u", "P", -lu(r, 0.5, 2) * u * v * v)
        gam, eps = r.uniform(0.5, 2), lu(r, 0.002, 0.05)
        gate = r.choice(["const", "tanh", "rational"], p=[0.5, 0.25, 0.25])
        if gate == "const":
            gv = r.choice(["lin", "ap_v", "quad"])
            G_v = {"lin": u - gam * v, "ap_v": -v - k * u * (u - th - 1), "quad": -v + gam * u**2}[gv]
            add(e, "v", "P", eps * G_v)
        elif gate == "tanh":
            e["prm"][:2] = [lu(r, 5, 30), r.uniform(0.1, 0.5)]
            add(e, "v", "P", (eps + lu(r, 0.01, 0.2) * G) * (u - gam * v)), e["tags"].add("tanh_gate")
        else:
            e["prm"][2] = r.uniform(0.1, 0.5)
            add(e, "v", "P", (eps + lu(r, 0.05, 0.3) * v * R) * (u - gam * v)), e["tags"].add("rational_gate")
        ic(e, "u", 0, 1, "patch"), ic(e, "v", 0, r.uniform(0.5, 2), "patch")
        rate = k
    elif kin == "ms":  # two-current (Mitchell-Schaeffer-like) gating with a smooth activation G(u)
        tin, tout, topen, tclose = r.uniform(0.2, 0.5), r.uniform(4, 8), r.uniform(60, 150), r.uniform(80, 200)
        e["prm"][:2] = [r.uniform(10, 30), r.uniform(0.1, 0.2)]
        add(e, "u", "P", v * G * (1 - u) / tin - u / tout)
        add(e, "v", "P", (1 - v) * (1 - G) / topen - v * G / tclose)
        ic(e, "u", 0, 1, "patch"), ic(e, "v", 1, -r.uniform(0.3, 1), "patch")
        rate = 1 / tin
    elif kin == "oregonator":
        eps, f, q = lu(r, 0.02, 0.1), r.uniform(0.5, 2.5), r.uniform(0.02, 0.1)
        e["prm"][2] = q
        add(e, "u", "P", (u * (1 - u) - f * v * (u - q) * R) / eps), add(e, "v", "P", u - v)
        ic(e, "u", 0, 0.8, "patch"), ic(e, "v", 0, 0.3, "patch")
        rate = 1 / eps
    elif kin == "grayscott":
        F, k = r.uniform(0.01, 0.06), r.uniform(0.045, 0.065)
        add(e, "u", "P", -u * v**2 + F * (1 - u)), add(e, "v", "P", u * v**2 - (F + k) * v)
        ic(e, "u", 1, -0.5, "patch"), ic(e, "v", 0, r.uniform(0.25, 0.5), "patch")
        rate, Dv_ratio = 1.0, 0.5
    elif kin == "brusselator":
        A, B = r.uniform(1, 3), r.uniform(1.5, 4)
        add(e, "u", "P", A - (B + 1) * u + u**2 * v), add(e, "v", "P", B * u - u**2 * v)
        ic(e, "u", A, 0.3 * A, "signed"), ic(e, "v", B / A, 0.3 * B / A, "signed")
        rate, Dv_ratio = B + 1, lu(r, 2, 20)
    elif kin == "schnakenberg":
        al, be = r.uniform(0.05, 0.3), r.uniform(0.5, 1.5)
        add(e, "u", "P", al - u + u**2 * v), add(e, "v", "P", be - u**2 * v)
        us, vs = al + be, be / (al + be) ** 2
        ic(e, "u", us, 0.3 * us, "signed"), ic(e, "v", vs, 0.3 * vs, "signed")
        rate, Dv_ratio = 1.0, lu(r, 10, 40)
    else:  # predator-prey with Holling-II saturation u v / (u + h)
        al, be, de = r.uniform(0.5, 2), r.uniform(0.5, 2), r.uniform(0.1, 0.5)
        e["prm"][2] = r.uniform(0.1, 0.5)
        add(e, "u", "P", u * (1 - u) - al * u * v * R), add(e, "v", "P", be * u * v * R - de * v)
        ic(e, "u", 0.05, 0.8, "patch"), ic(e, "v", 0.05, 0.5, "patch")
        rate, Dv_ratio = 1.0, lu(r, 0.2, 5)

    Du = (lu(r, 1.5, 4) * dx) ** 2 * rate
    if r.random() < 0.25:
        e["terms"]["u"]["D"][("a", "u")] = 1.0
        e["static"] = {"mean": Du, "amp": 0.6 * Du}
        e["tags"].add("medium")
    elif r.random() < 0.27:
        diff(e, "u", Du * lu(r, 0.3, 1), Du), e["tags"].add("anisotropic")
    else:
        diff(e, "u", Du)
    if Dv_ratio is None:
        Dv_ratio = 0 if r.random() < 0.5 else lu(r, 0.05, 1)
    if Dv_ratio:
        diff(e, "v", Du * Dv_ratio)
    if n == 3:
        add(e, "u", "P", -r.uniform(0.2, 1) * w), add(e, "w", "P", lu(r, 0.005, 0.1) * rate * (u - w))
        diff(e, "w", Du * lu(r, 1, 10)), ic(e, "w", 0, 0.5)
        e["tags"].add("inhibitor")
    if r.random() < 0.15:
        c1, c2 = (lu(r, 0.1, 1) * math.sqrt(Du * rate) * r.choice([-1, 1]) for _ in range(2))
        for f in e["fields"]:
            lin(e, f, "dx", -c1), lin(e, f, "dy", -c2)
        e["tags"].add("flow")
    if kin in ("grayscott", "brusselator", "schnakenberg", "predprey") and r.random() < 0.1:
        e["terms"]["u"]["D"][("u", "v")] = -Du * lu(r, 0.1, 1) / max(abs(e["ic"]["v"]["amp"]), 0.1)
        e["tags"].add("chemotaxis")
    if r.random() < 0.1:
        add(e, "u", "Q", Du * r.uniform(-0.3, 0.3) * v), e["tags"].add("cross-diffusion")
    e["dt0"] = 0.1 / rate
    return e


# ---------------------------------------------------------------- pattern formation
PAT_P = {"ks": 0.15, "sh": 0.2, "ch": 0.1, "ac": 0.12, "fisher": 0.08, "cgl": 0.2, "keller-segel": 0.15}


def pattern(r):
    kind = str(r.choice(list(PAT_P), p=list(PAT_P.values())))
    if kind == "ks":
        L = lu(r, 30, 200)
        lam = L / lu(r, 4, 12)
        ell2 = (lam / (2 * math.pi * math.sqrt(2))) ** 2
        e = new(r, "pattern/kuramoto-sivashinsky", 1, L)
        diff(e, "u", -1.0), lin(e, "u", "bih", -ell2), add(e, "u", "P", -r.uniform(0.3, 1) / 2 * gu)
        ic(e, "u", 0, lu(r, 0.1, 1), "signed")
        e["dt0"] = 0.1 * 4 * ell2
    elif kind == "sh":
        L = lu(r, 20, 200)
        k0 = 2 * math.pi / (L / lu(r, 4, 15))
        e = new(r, "pattern/swift-hohenberg", 1, L)
        lin(e, "u", "1", r.uniform(0.05, 0.6) - 1), diff(e, "u", -2 / k0**2), lin(e, "u", "bih", -1 / k0**4)
        add(e, "u", "P", r.uniform(0, 1.5) * u**2 - u**3)
        ic(e, "u", 0, lu(r, 0.05, 0.5), "signed")
        e["dt0"] = 0.2
    elif kind == "ch":
        L = lu(r, 10, 100)
        dx = L / N
        gam = (lu(r, 1.5, 3) * dx) ** 2
        e = new(r, "pattern/cahn-hilliard", 1, L)
        add(e, "u", "Q", u**3), diff(e, "u", -1.0), lin(e, "u", "bih", -gam)
        ic(e, "u", r.uniform(-0.4, 0.4), lu(r, 0.05, 0.5), "signed")
        e["dt0"] = 0.02 * dx**2
    elif kind == "ac":
        L = lu(r, 10, 100)
        e = new(r, "pattern/allen-cahn", 1, L)
        diff(e, "u", (lu(r, 1.5, 4) * L / N) ** 2), lin(e, "u", "1", 1.0), add(e, "u", "P", -u**3)
        ic(e, "u", r.uniform(-0.3, 0.3), lu(r, 0.1, 1), "signed")
    elif kind == "fisher":
        L = lu(r, 10, 100)
        e = new(r, "pattern/fisher-kpp", 1, L)
        diff(e, "u", (lu(r, 1.5, 4) * L / N) ** 2), lin(e, "u", "1", 1.0), add(e, "u", "P", -u**2)
        ic(e, "u", 0, 1, "patch")
    elif kind == "cgl":
        L = lu(r, 20, 200)
        ell2 = (lu(r, 3, 6) * L / N) ** 2
        al, be = r.uniform(-1, 1), r.uniform(-2, 2)
        e = new(r, "pattern/complex-ginzburg-landau", 2, L)
        for f, g, s in (("u", v, 1), ("v", u, -1)):
            lin(e, f, "1", 1.0), diff(e, f, ell2), add(e, f, "Q", -s * al * ell2 * g)
        add(e, "u", "P", -(u**2 + v**2) * (u - be * v)), add(e, "v", "P", -(u**2 + v**2) * (v + be * u))
        ic(e, "u", 0, lu(r, 0.1, 1), "signed"), ic(e, "v", 0, lu(r, 0.1, 1), "signed")
        e["dt0"] = 0.05
    else:
        L = lu(r, 10, 100)
        Du = (lu(r, 1.5, 3) * L / N) ** 2
        e = new(r, "pattern/keller-segel", 2, L)
        diff(e, "u", Du)
        e["terms"]["u"]["D"][("u", "v")] = -Du * lu(r, 2, 20)
        lin(e, "u", "1", r.uniform(0.1, 1)), add(e, "u", "P", -e["terms"]["u"]["L"]["1"] * u**2)
        diff(e, "v", Du * lu(r, 1, 10)), add(e, "v", "P", u), lin(e, "v", "1", -1.0)
        ic(e, "u", 0.2, 0.8, "patch"), ic(e, "v", 0.2, 0.2, "signed")
    e["tags"].add(kind)
    return e


# ---------------------------------------------------------------- free composition and mutations
def _p_slots(e):
    """P-library monomials (as strings for target '{f}') whose symbols exist in this equation."""
    ok = set(e["fields"]) | {g + d for g in e["fields"] for d in "xy"} | {"g" + g for g in e["fields"]} | {"G", "R", "S", "1"}
    if e["static"]:
        ok.add("a")
    nf = 3 if len(e["fields"]) == 3 else 2
    out = []
    for kind, item in LIB[nf]:
        if kind == "P":
            names = set(item.replace("**", "*").replace("{f}", "u").split("*")) - {"2", "3"}
            if names <= ok:
                out.append(item)
    return out


def mutate(r, e, scale):
    for _ in range(int(r.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))):
        f = str(r.choice(e["fields"]))
        m = str(r.choice(_p_slots(e))).format(f=f)
        add(e, f, "P", lu(r, 0.02, 0.2) * scale * r.choice([-1, 1]) * sp.sympify(m, locals=SYM))
        e["tags"].add(f"+{f}:{m}")
    return e


def free(r):
    n = int(r.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
    L = lu(r, 10, 60)
    dx = L / N
    e = new(r, "free", n, L)
    if r.random() < 0.3:
        e["static"] = {"mean": 1.0, "amp": 0.5}
    vmax = 0.0
    for f in e["fields"]:
        nu = (lu(r, 1.5, 4) * dx) ** 2
        diff(e, f, nu), lin(e, f, "1", -lu(r, 0.05, 0.5)), add(e, f, "P", -lu(r, 0.1, 1) * SYM[f] ** 3)
        ic(e, f, 0, lu(r, 0.3, 1))
        for _ in range(int(r.integers(2, 6))):
            kind = r.choice(["P", "X", "Y", "Q", "D"], p=[0.55, 0.15, 0.15, 0.1, 0.05])
            if kind == "P":
                add(e, f, "P", lu(r, 0.05, 1) * r.choice([-1, 1]) * sp.sympify(str(r.choice(_p_slots(e))).format(f=f), locals=SYM))
            elif kind in "XY":
                ms = [g for g in e["fields"]] + [f"{g}*{h}" for g in e["fields"] for h in e["fields"]]
                c = lu(r, 0.05, 1)
                vmax = max(vmax, c)
                add(e, f, kind, c * r.choice([-1, 1]) * sp.sympify(str(r.choice(ms)), locals=SYM))
            elif kind == "Q":
                add(e, f, "Q", nu * lu(r, 0.1, 0.5) * r.choice([-1, 1]) * SYM[str(r.choice(e["fields"]))])
            else:
                pot = str(r.choice(e["fields"]))
                e["terms"][f]["D"][(f, pot)] = e["terms"][f]["D"].get((f, pot), 0) + nu * lu(r, 0.1, 1) * r.choice([-1, 1])
    e["dt0"] = min(0.05, 0.2 * dx / (vmax + 1e-9))
    return e


FAMILIES = {"transport": (transport, 0.35), "excitable": (excitable, 0.35), "pattern": (pattern, 0.2), "free": (free, 0.1)}


def holdout(e):
    t = e["tags"]
    return {"flux", "adv", "grad_lin"} <= t or {"nagumo", "uv", "rational_gate"} <= t


def equation(r):
    while True:
        name = str(r.choice(list(FAMILIES), p=[p for _, p in FAMILIES.values()]))
        e = FAMILIES[name][0](r)
        if name != "free":
            rate = 0.1 / e["dt0"] if name == "excitable" else 1.0
            mutate(r, e, rate)
        if not holdout(e):
            e["tags"] = sorted(e["tags"])
            e["n"] = len(e["fields"])
            e["nf"] = 3 if e["n"] == 3 else 2
            return e


def ic_arrays(e):
    """(nf+1, 3) rows of (mean, amp, mode) for the template channels; unused fields are zero, static last."""
    rows = [[e["ic"][f]["mean"], e["ic"][f]["amp"], MODES[e["ic"][f]["mode"]]] for f in e["fields"]]
    rows += [[0.0, 0.0, 0]] * (e["nf"] - e["n"])
    s = e["static"] or {"mean": 0.0, "amp": 0.0}
    return np.array(rows + [[s["mean"], s["amp"], 3]], np.float32)


# ---------------------------------------------------------------- initial-condition pools (GPU)
def _modes(n):
    return jnp.fft.rfftfreq(n, 1 / n)[None, :], jnp.fft.fftfreq(n, 1 / n)[:, None]


def _unit(p):
    p = p - p.mean()
    return p / (p.std() + 1e-8)


def _lowpass(p, n):
    kx, ky = _modes(n)
    return jnp.fft.irfft2(jnp.fft.rfft2(p) * (kx**2 + ky**2 <= (n / 3) ** 2), s=(n, n))


FAM_NAMES = ("grf", "fourier", "noise", "blobs", "shapes", "waves", "vortex")
FAM_W = jnp.array([0.24, 0.12, 0.1, 0.16, 0.14, 0.12, 0.12])


def pattern_field(key, n=N):
    """Zero-mean unit-std random pattern on the unit torus from a 7-family mixture (+ optional multiscale GRF)."""
    kx, ky = _modes(n)
    kk = jnp.sqrt(kx**2 + ky**2)
    X, Y = jnp.meshgrid(jnp.arange(n) / n, jnp.arange(n) / n)
    U = lambda k, lo, hi, shape=(): jax.random.uniform(k, shape, minval=lo, maxval=hi)
    LU = lambda k, lo, hi, shape=(): jnp.exp(U(k, math.log(lo), math.log(hi), shape))
    wrap = lambda d: (d + 0.5) % 1 - 0.5

    def spectral(k, amp):
        k1, k2 = jax.random.split(k)
        return jnp.fft.irfft2((jax.random.normal(k1, kk.shape) + 1j * jax.random.normal(k2, kk.shape)) * amp, s=(n, n))

    def grf(k):
        k = jax.random.split(k, 5)
        th, s = U(k[2], 0, jnp.pi), LU(k[3], 0.25, 4)
        kr, kp = kx * jnp.cos(th) + ky * jnp.sin(th), -kx * jnp.sin(th) + ky * jnp.cos(th)
        return spectral(k[4], (kr**2 / s + kp**2 * s + LU(k[1], 1, 8) ** 2) ** (-U(k[0], 1, 3)))

    def fourier(k):
        k1, k2 = jax.random.split(k)
        return spectral(k2, (kk <= jax.random.randint(k1, (), 2, 13)) * (kk > 0))

    def noise(k):
        k1, k2 = jax.random.split(k)
        return spectral(k2, jnp.exp(-2 * jnp.pi**2 * (kk * LU(k1, 0.008, 0.05)) ** 2))

    def blobs(k):
        k = jax.random.split(k, 5)
        c, wd = jax.random.uniform(k[0], (8, 2)), LU(k[1], 0.02, 0.2, (8,))
        sg = jnp.where(jax.random.bernoulli(k[2], 0.7, (8,)), 1.0, -1.0) * (jnp.arange(8) < jax.random.randint(k[3], (), 1, 9))
        d2 = wrap(X[None] - c[:, 0, None, None]) ** 2 + wrap(Y[None] - c[:, 1, None, None]) ** 2
        return jnp.sum(sg[:, None, None] * jnp.exp(-d2 / (2 * wd[:, None, None] ** 2)), 0)

    def shapes(k):  # disks and axis-aligned bands with smooth edges
        k = jax.random.split(k, 7)
        c, rad = jax.random.uniform(k[0], (4, 2)), U(k[1], 0.05, 0.35, (4,))
        edge = LU(k[2], 0.01, 0.05, (4,))
        dxs, dys = wrap(X[None] - c[:, 0, None, None]), wrap(Y[None] - c[:, 1, None, None])
        disk = rad[:, None, None] - jnp.sqrt(dxs**2 + dys**2)
        band = rad[:, None, None] - jnp.where(jax.random.bernoulli(k[3], 0.5, (4,))[:, None, None], jnp.abs(dxs), jnp.abs(dys))
        sd = jnp.where(jax.random.bernoulli(k[4], 0.6, (4,))[:, None, None], disk, band)
        on = (jnp.arange(4) < jax.random.randint(k[5], (), 1, 5)) * jnp.where(jax.random.bernoulli(k[6], 0.8, (4,)), 1.0, -1.0)
        return jnp.sum(on[:, None, None] * (1 + jnp.tanh(sd / edge[:, None, None])) / 2, 0)

    def waves(k):  # superposed periodic plane waves (stripes, squares, hexagon-like)
        k = jax.random.split(k, 4)
        m = jax.random.randint(k[0], (3, 2), -10, 11)
        m = m.at[:, 0].set(jnp.where((m[:, 0] == 0) & (m[:, 1] == 0), 1, m[:, 0]))
        on = jnp.arange(3) < jax.random.randint(k[1], (), 1, 4)
        ph = U(k[2], 0, 2 * jnp.pi, (3,))
        return jnp.sum(on[:, None, None] * jnp.cos(2 * jnp.pi * (m[:, 0, None, None] * X + m[:, 1, None, None] * Y) + ph[:, None, None]), 0)

    def vortex(k):  # phase-singularity pair (spiral seed), windowed
        k = jax.random.split(k, 4)
        c = jax.random.uniform(k[0], (2,)) * 0.5 + 0.25
        sep = U(k[1], 0.05, 0.3)
        d1x, d1y = wrap(X - c[0]), wrap(Y - c[1])
        d2x, d2y = wrap(X - c[0] - sep), wrap(Y - c[1])
        th = jnp.arctan2(d1y, d1x) - jnp.arctan2(d2y, d2x)
        r = jnp.sqrt(d1x**2 + d1y**2)
        return jnp.cos(th + U(k[2], 0, 40) * r) * jnp.exp(-(r / U(k[3], 0.2, 0.5)) ** 2)

    k = jax.random.split(key, 4)
    fam = jax.random.categorical(k[0], jnp.log(FAM_W))
    p = jax.lax.switch(fam, [grf, fourier, noise, blobs, shapes, waves, vortex], k[1])
    p = _unit(_lowpass(p, n)) + jax.random.bernoulli(k[2], 0.3) * 0.4 * _unit(grf(k[3]))
    return _unit(_lowpass(p, n)), fam


def _one_ic(key, rows, n):
    C = rows.shape[0]
    k = jax.random.split(key, 6)
    pats, fams = jax.vmap(lambda kk: pattern_field(kk, n))(jax.random.split(k[0], C))
    rho = U01(k[1], (C,)) * 1.8 - 0.9
    corr = jax.random.bernoulli(k[2], 0.3, (C,)) & (jnp.arange(C) > 0)
    pats = jnp.where(corr[:, None, None], rho[:, None, None] * pats[:1] + jnp.sqrt(1 - rho**2)[:, None, None] * pats, pats)
    # "wake": a later channel is channel 0's pattern shifted a little (excited front + refractory tail = travelling wave)
    sh = jax.random.randint(k[1], (2,), -n // 10, n // 10 + 1)
    wake = jax.random.bernoulli(k[2], 0.25, (C,)) & (jnp.arange(C) > 0) & ~corr
    pats = jnp.where(wake[:, None, None], jnp.roll(pats[0], (sh[0], sh[1]), (0, 1))[None], pats)
    pats = jax.vmap(_unit)(pats)
    mean, amp, mode = rows[:, 0], rows[:, 1], rows[:, 2]
    scale = U01(k[3], (C,)) * 0.9 + 0.3
    rest = jax.random.bernoulli(k[4], 0.25, (C,)) & (jnp.arange(C) > 0)
    kp = jax.random.split(k[5], 3)
    th, kap = U01(kp[0], (C,)) * 2.5 - 1.0, U01(kp[1], (C,)) * 6 + 2
    signed = pats * scale[:, None, None]
    patch = jax.nn.sigmoid(kap[:, None, None] * (pats - th[:, None, None]))
    use_patch = (mode == 1) | ((mode == 2) & jax.random.bernoulli(kp[2], 0.5, (C,)))
    shape = jnp.where(use_patch[:, None, None], patch, signed)
    shape = jnp.where((mode == 3)[:, None, None], jnp.tanh(pats), shape)
    shape = jnp.where(rest[:, None, None] & (mode != 3)[:, None, None], 0.0, shape)
    return mean[:, None, None] + amp[:, None, None] * shape, fams[0]


def U01(k, shape):
    return jax.random.uniform(k, shape)


IC_CHUNK = 1024


@jax.jit
def _ic_chunk(keys, rows):  # fixed shapes (IC_CHUNK, 2), (IC_CHUNK, 4, 3): this compile is slow (~2 min), so do it once
    return jax.vmap(lambda k, r: _one_ic(k, r, N))(keys, rows)


def ic_pool(keys, rows, M):
    """keys (E, 2), rows (E, nf+1, 3) -> ICs (E, M, nf+1, N, N), family id of channel 0 (E, M)."""
    E, C = rows.shape[:2]
    r4 = np.zeros((E, 4, 3), np.float32)
    r4[:, : C - 1], r4[:, 3] = rows[:, :-1], rows[:, -1]  # layout u v w static; static is always channel 3
    k = jax.vmap(lambda kk: jax.random.split(kk, M))(jnp.asarray(keys)).reshape(E * M, -1)
    r = np.repeat(r4, M, 0)
    pad = -(E * M) % IC_CHUNK
    k = jnp.concatenate([k, jnp.repeat(k[:1], pad, 0)])
    r = np.concatenate([r, np.repeat(r[:1], pad, 0)])
    outs = [_ic_chunk(k[i : i + IC_CHUNK], jnp.asarray(r[i : i + IC_CHUNK])) for i in range(0, len(r), IC_CHUNK)]
    x = jnp.concatenate([o[0] for o in outs])[: E * M]
    fam = jnp.concatenate([o[1] for o in outs])[: E * M]
    x = x[:, list(range(C - 1)) + [3]]
    return x.reshape(E, M, C, N, N), fam.reshape(E, M)


# ---------------------------------------------------------------- max-entropy subset selection (GPU)
def _features(x, cm):
    """x (B, T, C, n, n) normalized trajectories, cm (C,) active-channel mask -> shift-invariant descriptors."""
    B, T, C, n, _ = x.shape
    x = x * cm[None, None, :, None, None]
    kx, ky = _modes(n)
    kk = jnp.sqrt(kx**2 + ky**2)
    frames = jnp.array([0, 1, 2, 4, 7, T - 1])
    xf = x[:, frames]
    P = jnp.abs(jnp.fft.rfft2(xf - xf.mean((-2, -1), keepdims=True))) ** 2
    edges = jnp.geomspace(1, n / 3 + 1, 17)
    rb = ((kk >= edges[:-1, None, None]) & (kk < edges[1:, None, None])).astype(x.dtype)
    ang = jnp.arctan2(ky * jnp.ones_like(kx), kx * jnp.ones_like(ky))  # in [-pi/2, pi/2]
    sec = (jnp.floor((ang + jnp.pi / 2) / (jnp.pi / 4)).clip(0, 3) == jnp.arange(4)[:, None, None]) & (kk >= 1) & (kk < 16)
    radial = jnp.log(jnp.einsum("btcij,kij->btck", P, rb) + 1e-6)
    sectors = jnp.log(jnp.einsum("btcij,kij->btck", P, sec.astype(x.dtype)) + 1e-6)
    std = jnp.log(x.std((-2, -1)) + 1e-4)
    mean = x.mean((-2, -1))
    rate = jnp.log(jnp.sqrt(((x[:, 1:] - x[:, :-1]) ** 2).mean((-2, -1))) + 1e-4)
    return jnp.concatenate([radial.reshape(B, -1), sectors.reshape(B, -1), std.reshape(B, -1),
                            mean.reshape(B, -1), rate.reshape(B, -1)], 1)


def _greedy_logdet(Kmat, valid, k, noise):
    """Greedy maximisation of log det(I + K_S / noise): the GP joint entropy of the chosen set.  Monotone
    submodular, so greedy is within (1 - 1/e) of the best subset.  Incremental Cholesky (Chen et al. 2018)."""
    L = Kmat / noise + jnp.eye(Kmat.shape[0])
    d0 = jnp.where(valid, jnp.diag(L), -jnp.inf)

    def body(i, st):
        cis, d, sel = st
        j = jnp.argmax(d)
        e = (L[j] - cis.T @ cis[:, j]) / jnp.sqrt(d[j])
        return cis.at[i].set(e), (d - e**2).at[j].set(-jnp.inf), sel.at[i].set(j)

    st = (jnp.zeros((k, L.shape[0])), d0, jnp.zeros(k, jnp.int32))
    return jax.lax.fori_loop(0, k, body, st)[2]


@partial(jax.jit, static_argnames=("k",))
def select(traj, cm, valid, *, k, noise=1e-2):
    """traj (E, M, T, C, n, n), cm (E, C), valid (E, M) -> (E, k) indices of a max-entropy subset per equation."""

    def one(x, cm, ok):
        f = _features(x.astype(jnp.float32), cm)
        okf = ok[:, None]
        mu = jnp.sum(f * okf, 0) / jnp.maximum(ok.sum(), 1)
        sd = jnp.sqrt(jnp.sum(((f - mu) * okf) ** 2, 0) / jnp.maximum(ok.sum(), 1))
        z = jnp.where(okf, (f - mu) / (sd + 1e-3), 0.0)
        d2 = jnp.maximum(jnp.sum(z**2, 1)[:, None] + jnp.sum(z**2, 1)[None] - 2 * z @ z.T, 0)
        h = jnp.nanmedian(jnp.where(ok[:, None] & ok[None] & ~jnp.eye(len(ok), dtype=bool), d2, jnp.nan))
        return _greedy_logdet(jnp.exp(-d2 / (h + 1e-6)), ok, k, noise)

    return jax.vmap(one)(traj, cm, valid)


if __name__ == "__main__":
    from template import encode, rhs_strings
    import pde

    r = np.random.default_rng(0)
    fams = {}
    for i in range(3000):  # every sampled equation must encode into its template and parse back
        e = equation(r)
        C = encode(e, e["nf"])
        s = rhs_strings(e["fields"], C, e["prm"], e["nf"])
        if i < 300:
            pde.parse({"fields": e["fields"], "static": ["a"], "rhs": s})
        assert not holdout({"tags": set(e["tags"])})
        fams[e["family"]] = fams.get(e["family"], 0) + 1
    print({k: fams[k] for k in sorted(fams)})
    es = [e for e in (equation(r) for _ in range(50)) if e["nf"] == 2][:2]
    rows = jnp.asarray(np.stack([ic_arrays(e) for e in es]))
    x, fam = ic_pool(jax.random.split(jax.random.PRNGKey(0), 2), np.asarray(rows), M=64)
    assert x.shape == (2, 64, 3, N, N) and bool(jnp.isfinite(x).all())
    sel = select(jnp.repeat(x[:, :, None], 11, 2)[:, :, :, :2], jnp.ones((2, 2)), jnp.ones((2, 64), bool), k=16)
    assert all(len(set(map(int, s))) == 16 for s in sel), sel
    print("sample.py ok")

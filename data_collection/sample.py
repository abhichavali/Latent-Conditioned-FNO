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
            "terms": {f: {"L": {}, "P": 0, "X": 0, "Y": 0, "Q": 0, "D": {}} for f in fs},
            "core": {f: {"L": {}, "P": 0, "X": 0, "Y": 0, "Q": 0, "D": {}} for f in fs}}  # kept by dropout


def lin(e, f, op, c):
    e["terms"][f]["L"][op] = e["terms"][f]["L"].get(op, 0.0) + float(c)


def diff(e, f, D, Dy=None):
    lin(e, f, "dxx", D)
    lin(e, f, "dyy", D if Dy is None else Dy)


def add(e, f, kind, expr):
    e["terms"][f][kind] = e["terms"][f][kind] + expr


def addc(e, f, kind, expr):  # a building block the family is built around: never dropped
    e["core"][f][kind] = e["core"][f][kind] + expr


SWE_PIECES = ("flux", "adv", "grad", "bathy")  # div(h u), u.grad u, g grad h, bathymetry coupling
AP_PIECES = ("cubic", "uv", "gate")  # cubic reaction, -u v, v / (u + mu) recovery gate


def pick(r, pieces, size_p, forbidden):
    """Random subset of building blocks with a given size distribution (so every single and every pair is common);
    subsets containing the forbidden core are redrawn."""
    while True:
        k = int(r.choice(len(size_p), p=size_p))
        sel = frozenset(str(x) for x in r.choice(pieces, k, replace=False)) if k else frozenset()
        if not forbidden <= sel:
            return sel


def ic(e, f, mean, amp, mode="any"):
    e["ic"][f] = {"mean": float(mean), "amp": float(amp), "mode": mode}


# ---------------------------------------------------------------- transport / waves (near shallow water)
def transport(r):
    L = lu(r, 5, 50)
    dx = L / N
    n = int(r.choice([1, 2, 3], p=[0.15, 0.25, 0.6]))

    def visc(e, f, U):
        if r.random() < 0.3:
            lin(e, f, "bih", -lu(r, 0.3, 2) * U * dx**3)
        else:
            diff(e, f, lu(r, 0.3, 2) * U * dx)

    if n == 3:  # shallow-water building blocks, each alone and in pairs (never flux + adv + grad together)
        e = new(r, "transport/shallow-like", 3, L)
        H, g = lu(r, 0.5, 2), lu(r, 0.5, 2)
        c = math.sqrt(g * H)
        pc = pick(r, SWE_PIECES, [0.05, 0.35, 0.45, 0.15], frozenset({"flux", "adv", "grad"}))
        e["tags"] |= set(pc)
        if "bathy" in pc:  # depth from a static bathymetry field a
            depth, total = a, False
            e["static"] = {"mean": H, "amp": 0.4 * H}
        else:  # constant depth, or none when u is itself the total height carried by the nonlinear flux
            total = "flux" in pc and r.random() < 0.5
            depth = 0 if total else H
        addc(e, "u", "X", -depth * v), addc(e, "u", "Y", -depth * w)
        if "flux" in pc:
            addc(e, "u", "X", -u * v), addc(e, "u", "Y", -u * w)
        ic(e, "u", H if total else 0, (0.15 if total else 0.3) * H)
        if "grad" in pc:
            addc(e, "v", "X", -g * u), addc(e, "w", "Y", -g * u)
        elif r.random() < 0.5:  # a different pressure law: not the g grad h block
            add(e, "v", "X", -g / (2 * H) * u**2), add(e, "w", "Y", -g / (2 * H) * u**2)
            e["tags"].add("grad_sq")
        if "adv" in pc:
            addc(e, "v", "P", -v * vx - w * vy), addc(e, "w", "P", -v * wx - w * wy)
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
    lin(e, "u", "dxlap", -A * (lu(r, 1.5, 6) * dx) ** 2)
    if r.random() < 0.5:
        lin(e, "u", "dylap", -A * (lu(r, 1.5, 6) * dx) ** 2)
    diff(e, "u", lu(r, 0.2, 1) * A * dx)
    ic(e, "u", 0, A)
    if n == 2:
        add(e, "v", "P", -u * vx - r.uniform(0, 1) * u * vy), diff(e, "v", lu(r, 0.3, 2) * A * dx), ic(e, "v", 0, 1)
        e["tags"].add("tracer")
    e["dt0"] = 0.2 * dx / (c + A)
    return e


# ---------------------------------------------------------------- excitable / reaction-diffusion (near Aliev-Panfilov)
EXCITABLE = ("fhn", "barkley", "composite", "ms", "oregonator")
KIN_P = {"fhn": 0.1, "barkley": 0.09, "composite": 0.35, "ms": 0.09, "oregonator": 0.06,
         "grayscott": 0.08, "brusselator": 0.06, "schnakenberg": 0.06, "predprey": 0.11}


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
    elif kin == "composite":  # Aliev-Panfilov building blocks, each alone and in pairs (never all three)
        pc = pick(r, AP_PIECES, [0.1, 0.45, 0.45], frozenset(AP_PIECES))
        e["tags"] |= set(pc)
        k, th = r.uniform(5, 10), r.uniform(0.05, 0.25)
        if "cubic" in pc:
            addc(e, "u", "P", k * u * (1 - u) * (u - th))
        elif r.random() < 0.5:  # bistable switch without a cubic: u relaxes to G(u), a smooth step at th
            e["prm"][:2] = [r.uniform(8, 25), th]
            addc(e, "u", "P", k / 2 * (G - u)), e["tags"].add("switch")
        else:
            addc(e, "u", "P", k / 4 * u * (1 - u)), e["tags"].add("logistic")
        addc(e, "u", "P", -u * v if "uv" in pc else -lu(r, 0.5, 1.5) * v)
        gam, eps = r.uniform(0.5, 2), lu(r, 0.002, 0.05)
        if "gate" in pc:
            e["prm"][2] = r.uniform(0.1, 0.5)
            addc(e, "v", "P", (eps + lu(r, 0.05, 0.3) * v * R) * (u - gam * v))
        else:
            gv = str(r.choice(["lin", "ap_v", "tanh"]))
            if gv == "tanh":
                if "switch" not in e["tags"]:
                    e["prm"][:2] = [lu(r, 5, 30), r.uniform(0.1, 0.5)]
                addc(e, "v", "P", (eps + lu(r, 0.01, 0.2) * G) * (u - gam * v)), e["tags"].add("tanh_gate")
            else:
                addc(e, "v", "P", eps * (u - gam * v if gv == "lin" else -v - k * u * (u - th - 1)))
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

    Du = (lu(r, 1.5, 6) * dx) ** 2 * rate
    if r.random() < 0.4:
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
        lam = L / lu(r, 2.5, 14)
        ell2 = (lam / (2 * math.pi * math.sqrt(2))) ** 2
        e = new(r, "pattern/kuramoto-sivashinsky", 1, L)
        diff(e, "u", -1.0), lin(e, "u", "bih", -ell2), add(e, "u", "P", -r.uniform(0.3, 1) / 2 * gu)
        ic(e, "u", 0, lu(r, 0.1, 1), "signed")
        e["dt0"] = 0.1 * 4 * ell2
    elif kind == "sh":
        L = lu(r, 20, 200)
        k0 = 2 * math.pi / (L / lu(r, 3, 16))
        e = new(r, "pattern/swift-hohenberg", 1, L)
        lin(e, "u", "1", r.uniform(0.05, 0.6) - 1), diff(e, "u", -2 / k0**2), lin(e, "u", "bih", -1 / k0**4)
        add(e, "u", "P", r.uniform(0, 1.5) * u**2 - u**3)
        ic(e, "u", 0, lu(r, 0.05, 0.5), "signed")
        e["dt0"] = 0.2
    elif kind == "ch":
        L = lu(r, 10, 100)
        dx = L / N
        gam = (lu(r, 1.5, 5) * dx) ** 2
        e = new(r, "pattern/cahn-hilliard", 1, L)
        add(e, "u", "Q", u**3), diff(e, "u", -1.0), lin(e, "u", "bih", -gam)
        ic(e, "u", r.uniform(-0.4, 0.4), lu(r, 0.05, 0.5), "signed")
        e["dt0"] = 0.02 * dx**2
    elif kind == "ac":
        L = lu(r, 10, 100)
        e = new(r, "pattern/allen-cahn", 1, L)
        diff(e, "u", (lu(r, 1.5, 6) * L / N) ** 2), lin(e, "u", "1", 1.0), add(e, "u", "P", -u**3)
        ic(e, "u", r.uniform(-0.3, 0.3), lu(r, 0.1, 1), "signed")
    elif kind == "fisher":
        L = lu(r, 10, 100)
        e = new(r, "pattern/fisher-kpp", 1, L)
        diff(e, "u", (lu(r, 1.5, 6) * L / N) ** 2), lin(e, "u", "1", 1.0), add(e, "u", "P", -u**2)
        ic(e, "u", 0, 1, "patch")
    elif kind == "cgl":
        L = lu(r, 20, 200)
        ell2 = (lu(r, 2.5, 8) * L / N) ** 2
        al, be = r.uniform(-1, 1), r.uniform(-2, 2)
        e = new(r, "pattern/complex-ginzburg-landau", 2, L)
        for f, g, s in (("u", v, 1), ("v", u, -1)):
            lin(e, f, "1", 1.0), diff(e, f, ell2), add(e, f, "Q", -s * al * ell2 * g)
        add(e, "u", "P", -(u**2 + v**2) * (u - be * v)), add(e, "v", "P", -(u**2 + v**2) * (v + be * u))
        ic(e, "u", 0, lu(r, 0.1, 1), "signed"), ic(e, "v", 0, lu(r, 0.1, 1), "signed")
        e["dt0"] = 0.05
    else:
        L = lu(r, 10, 100)
        Du = (lu(r, 1.5, 5) * L / N) ** 2
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
    """0-4 extra library terms of any kind (pointwise, flux, cross-diffusion, div(f grad g), linear operator) on
    random fields.  Sized off the grid (dx) and the family's rate so they perturb the dynamics, not replace them."""
    dx, F = e["L"] / N, e["fields"]
    for _ in range(int(r.choice(5, p=[0.15, 0.3, 0.3, 0.15, 0.1]))):
        f, g = str(r.choice(F)), str(r.choice(F))
        c = lu(r, 0.02, 0.3) * scale * r.choice([-1, 1])
        kind = str(r.choice(["P", "X", "Q", "D", "L"], p=[0.55, 0.15, 0.1, 0.05, 0.15]))
        if kind == "P":
            add(e, f, "P", c * sp.sympify(str(r.choice(_p_slots(e))).format(f=f), locals=SYM))
        elif kind == "X":  # flux of a degree <= 2 monomial along x or y, or of a static coefficient times a field
            m = SYM[g] * (SYM[str(r.choice(F))] if r.random() < 0.5 else 1)
            if e["static"] and r.random() < 0.3:
                m = a * SYM[g]
            add(e, f, str(r.choice(["X", "Y"])), c * dx * m)
        elif kind == "Q":  # cross-diffusion, or nonlinear (porous-medium) diffusion lap(g^3) with a stable sign
            add(e, f, "Q", (c * SYM[g] if r.random() < 0.7 else abs(c) * SYM[f] ** 3) * dx**2)
        elif kind == "D":
            e["terms"][f]["D"][(f, g)] = e["terms"][f]["D"].get((f, g), 0.0) + c * dx**2
        else:  # advection, dispersion or hyperdiffusion
            op = str(r.choice(["dx", "dy", "dxlap", "dylap", "bih"]))
            lin(e, f, op, -abs(c) * dx**4 if op == "bih" else c * dx ** (1 if len(op) == 2 else 3))
        e["tags"].add(f"+{f}:{kind}")
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
        nu = (lu(r, 1.5, 6) * dx) ** 2
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


FAMILIES = {"transport": (transport, 0.3), "excitable": (excitable, 0.3), "pattern": (pattern, 0.2), "free": (free, 0.2)}


def dropout(r, e, p):
    """Drop each non-diffusive term independently, so terms a family always emits together also appear apart.
    Diffusion operators (dxx dyy bih, div(a grad f)) and core building blocks are kept."""
    for f, t in e["terms"].items():
        for kind in "PXYQ":
            t[kind] = sp.Add(*[x for x in sp.Add.make_args(sp.expand(t[kind])) if r.random() >= p])
        t["L"] = {op: c for op, c in t["L"].items() if op in ("dxx", "dyy", "bih") or r.random() >= p}
        t["D"] = {kk: c for kk, c in t["D"].items() if kk == ("a", f) or r.random() >= p}


def merge_core(e):
    for f, t in e.pop("core").items():
        for kind in "PXYQ":
            add(e, f, kind, t[kind])
        for op, c in t["L"].items():
            lin(e, f, op, c)
        for kk, c in t["D"].items():
            e["terms"][f]["D"][kk] = e["terms"][f]["D"].get(kk, 0.0) + c


def pieces(C, nf, n):
    """SWE / AP building blocks present in a coefficient matrix (internal roles: SWE h, velocity, velocity;
    AP activator, recovery).  Read off the coefficients, so terms added by mutations count too."""
    idx = {(kind, item): k for k, (kind, item) in enumerate(LIB[nf])}
    has = lambda i, kind, item: i < n and (kind, item) in idx and C[i, idx[(kind, item)]] != 0
    out = set()
    if n == 3:
        if has(0, "X", "u*v") or has(0, "Y", "u*w"):
            out.add("flux")
        if any(has(i, "P", m) for i in (1, 2) for m in ("v*{f}x", "w*{f}y")):
            out.add("adv")
        if has(1, "X", "u") or has(2, "Y", "u"):
            out.add("grad")
        if has(0, "X", "a*v") or has(0, "Y", "a*w"):
            out.add("bathy")
    if n >= 2:
        if has(0, "P", "u*u*u"):
            out.add("cubic")
        if has(0, "P", "u*v"):
            out.add("uv")
        if any(has(1, "P", m) for m in ("R*v", "R*u*v", "R*v*v")):
            out.add("gate")
    return out


def holdout(ps):
    return {"flux", "adv", "grad"} <= ps or set(AP_PIECES) <= ps


def equation(r):
    from template import encode

    while True:
        name = str(r.choice(list(FAMILIES), p=[p for _, p in FAMILIES.values()]))
        e = FAMILIES[name][0](r)
        if name != "free":
            mutate(r, e, 0.1 / e["dt0"] if name == "excitable" else 1.0)
        if r.random() < 0.6:
            dropout(r, e, 0.1)
            e["tags"].add("dropout")
        merge_core(e)
        e["n"] = len(e["fields"])
        e["nf"] = 3 if e["n"] == 3 else 2
        e["C"] = encode(e, e["nf"])
        ps = pieces(e["C"], e["nf"], e["n"])
        if not holdout(ps):
            e["pieces"], e["tags"] = sorted(ps), sorted(e["tags"])
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
    scale = U01(k[3], (C,)) * 1.3 + 0.2
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
    """x (B, T, C, n, n) normalized trajectories, cm (C,) active-channel mask -> blocks of shift-invariant
    descriptors.  Spectra are per-frame distributions under a sqrt (Hellinger) embedding, so distances stay
    bounded: log spectra made near-single-mode ICs (stripes) look like extreme outliers and got over-picked."""
    B, T, C, n, _ = x.shape
    x = x * cm[None, None, :, None, None]
    kx, ky = _modes(n)
    kk = jnp.sqrt(kx**2 + ky**2)
    frames = jnp.array([0, 1, 2, 4, 7, T - 1])
    xf = x[:, frames]
    P = jnp.abs(jnp.fft.rfft2(xf - xf.mean((-2, -1), keepdims=True))) ** 2
    tot = P.sum((-2, -1))[..., None] + 1e-6 * n**4  # floor ~ a field with std 1e-3 of its IC amplitude
    edges = jnp.geomspace(1, n / 3 + 1, 17)
    rb = ((kk >= edges[:-1, None, None]) & (kk < edges[1:, None, None])).astype(x.dtype)
    ang = jnp.arctan2(ky * jnp.ones_like(kx), kx * jnp.ones_like(ky))  # in [-pi/2, pi/2]
    sec = (jnp.floor((ang + jnp.pi / 2) / (jnp.pi / 4)).clip(0, 3) == jnp.arange(4)[:, None, None]) & (kk >= 1)
    radial = jnp.sqrt(jnp.einsum("btcij,kij->btck", P, rb) / tot)
    sectors = jnp.sqrt(jnp.einsum("btcij,kij->btck", P, sec.astype(x.dtype)) / tot)
    amp = jnp.concatenate([jnp.log(x.std((-2, -1)) + 1e-2), x.mean((-2, -1))], 1)
    rate = jnp.log(jnp.sqrt(((x[:, 1:] - x[:, :-1]) ** 2).mean((-2, -1))) + 1e-2)
    return [f.reshape(B, -1) for f in (radial, sectors, amp, rate)]


def _greedy_logdet(Kmat, valid, k, noise, group, gcap, late, early_cap):
    """Greedy maximisation of log det(I + K_S / noise), the GP joint entropy of the chosen set (monotone submodular,
    so greedy is within 1 - 1/e of the best subset; incremental Cholesky, Chen et al. 2018), under caps: at most
    gcap picks per group (IC family) and at most early_cap from early windows.  Caps are dropped if infeasible."""
    L = Kmat / noise + jnp.eye(Kmat.shape[0])
    d0 = jnp.where(valid, jnp.diag(L), -jnp.inf)
    G = len(FAM_NAMES)

    def body(i, st):
        cis, d, sel, cnt, early = st
        blocked = (cnt[group] >= gcap) | (~late & (early >= early_cap))
        dc = jnp.where(blocked, -jnp.inf, d)
        j = jnp.argmax(jnp.where(dc.max() > -jnp.inf, dc, d))
        e = (L[j] - cis.T @ cis[:, j]) / jnp.sqrt(d[j])
        return (cis.at[i].set(e), (d - e**2).at[j].set(-jnp.inf), sel.at[i].set(j),
                cnt.at[group[j]].add(1), early + (~late[j]).astype(jnp.int32))

    st = (jnp.zeros((k, L.shape[0])), d0, jnp.zeros(k, jnp.int32), jnp.zeros(G, jnp.int32), jnp.int32(0))
    return jax.lax.fori_loop(0, k, body, st)[2]


@partial(jax.jit, static_argnames=("k",))
def select(traj, cm, valid, group, late, *, k, noise=1e-2, family_share=0.25, early_share=2 / 3):
    """traj (E, M, T, C, n, n), cm (E, C), valid / group (IC family) / late (developed window) (E, M)
    -> (E, k) indices of a capped max-entropy subset per equation."""
    gcap, early_cap = math.ceil(family_share * k), math.ceil(early_share * k)

    def one(x, cm, ok, g, lt):
        okf, cnt = ok[:, None], jnp.maximum(ok.sum(), 1)
        zs = []
        for f in _features(x.astype(jnp.float32), cm):  # each block gets unit total variance: equal say
            mu = jnp.sum(f * okf, 0) / cnt
            var = jnp.sum(((f - mu) * okf) ** 2) / cnt
            zs.append(jnp.where(okf, (f - mu) / jnp.sqrt(var + 1e-12), 0.0))
        z = jnp.concatenate(zs, 1)
        d2 = jnp.maximum(jnp.sum(z**2, 1)[:, None] + jnp.sum(z**2, 1)[None] - 2 * z @ z.T, 0)
        h = jnp.nanmedian(jnp.where(ok[:, None] & ok[None] & ~jnp.eye(len(ok), dtype=bool), d2, jnp.nan))
        return _greedy_logdet(jnp.exp(-d2 / (h + 1e-6)), ok, k, noise, g, gcap, lt, early_cap)

    return jax.vmap(one)(traj, cm, valid, group, late)


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
        assert not holdout(set(e["pieces"]))
        fams[e["family"]] = fams.get(e["family"], 0) + 1
    print({k: fams[k] for k in sorted(fams)})
    es = [e for e in (equation(r) for _ in range(50)) if e["nf"] == 2][:2]
    rows = jnp.asarray(np.stack([ic_arrays(e) for e in es]))
    x, fam = ic_pool(jax.random.split(jax.random.PRNGKey(0), 2), np.asarray(rows), M=64)
    assert x.shape == (2, 64, 3, N, N) and bool(jnp.isfinite(x).all())
    late = jnp.tile(jnp.arange(64) >= 32, (2, 1))
    sel = select(jnp.repeat(x[:, :, None], 11, 2)[:, :, :, :2], jnp.ones((2, 2)), jnp.ones((2, 64), bool), fam, late, k=16)
    assert all(len(set(map(int, s))) == 16 for s in sel), sel
    for s, f, lt in zip(np.asarray(sel), np.asarray(fam), np.asarray(late)):  # caps hold: <= 4 per family, <= 11 early
        assert max(np.bincount(f[s])) <= 4 and (~lt[s]).sum() <= 11, (f[s], lt[s])
    print("sample.py ok")

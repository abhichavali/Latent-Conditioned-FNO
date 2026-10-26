"""Unseen equations from the pretraining families (fresh generator draws, dataset C, seed 99): how many simulations
does an FNO from scratch need to match the v3 conditioned FNO, which adapts with zero gradient steps from k examples?

  python scratch_vs_ctx.py split   --data /workspace/data/C --out O     # O/split.json: equations, test / train windows
  python fewshot_ctx.py fresh      --split O/split.json --out O         # O/ctx_kshot.json (v3, k = 0 .. 64)
  python baselines.py eq           --split O/split.json --out O/scratch # O/scratch/runs/<eq>_N<N>_s<seed>.json
  python scratch_vs_ctx.py summary --out O                              # summary.json, scratch_vs_ctx.png

Per equation: test = n_test windows of a seeded random subset of its IC groups; train = every window of the other
groups in a fixed seeded order.  v3 at k conditions on the first k train windows; scratch at N trains on the first N
(nested prefixes; max(2, 0.1 N) of them held out for early stopping).  Both are scored on the same test windows with
the same metric: rel_slot0 = relL2 per (window, frame) of the primary field in the model's normalized space (the
training loss, uncapped; eval.summarize = baselines.stats), averaged over frames 1..14 and windows.  A "simulation"
is one 15-frame window (some share a pool IC: consecutive windows of one rollout).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PHASES = [[[8, 32, 128], 0], [[16, 64], 0], [[8, 32], 1]]  # scratch (sizes, seed) in run order; a deadline cuts the tail
CORE = (8, 32, 128)
KS_FEWER = (0, 1, 2, 5, 16, 64)


def split(a):
    from data import adapter

    eqs = adapter.load_equations([{"tag": a.tag, "path": a.data}])
    adapter.drop_nonfinite(eqs, f"{a.out}/nonfinite_{a.tag}.json")
    out, skipped = {2: [], 3: []}, {}
    for e in eqs:
        rng = np.random.default_rng([a.seed, int(e.spec["id"][3:])])
        test, tg = [], set()
        for g in rng.permutation(np.unique(e.groups)):
            if len(test) >= a.n_test:
                break
            tg.add(int(g))
            test += [int(w) for w, gg in zip(e.windows, e.groups) if gg == g]
        test = test[: a.n_test]
        train = [int(w) for w in rng.permutation([w for w, g in zip(e.windows, e.groups) if int(g) not in tg])]
        # trivial dynamics: primary-field change IC -> last frame, relative to the IC's spatial std
        fr = np.stack([e.frames_raw(w, [0, e.frames - 1])[:, 0] for w in test])
        change = float(np.median(np.sqrt(((fr[:, 1] - fr[:, 0]) ** 2).mean((-2, -1))) / np.maximum(fr[:, 0].std((-2, -1)), 1e-3)))
        why = f"{e.nf}-field" if e.nf not in out else "few windows" if len(train) < max(CORE) or len(test) < a.n_test else "trivial" if change < a.min_change else None
        if why or len(out[e.nf]) >= a.per_nf:
            skipped[e.gid] = why or "enough of this nf"
            continue
        out[e.nf].append({"name": f"{a.tag}_{e.spec['id']}", "dir": e.dir, "nf": e.nf, "ns": e.ns, "rhs": e.spec["rhs"],
                          "fields": e.spec["fields"], "change": change, "test": test, "train": train})
    sp = {"tag": a.tag, "data": a.data, "seed": a.seed, "eqs": out[2] + out[3], "skipped": skipped, "scratch_phases": PHASES}
    json.dump(sp, open(f"{a.out}/split.json", "w"), indent=1)
    print(f"{len(sp['eqs'])} equations ({len(out[2])} 2-field, {len(out[3])} 3-field), skipped {skipped}", flush=True)


def fewer(n, err, target):
    """Scratch sims to reach target on the curve (n, err), log-log interpolated: (value, bound) with bound "<=" when
    the smallest N already reaches it, ">" when the largest does not (value = max N), "=" otherwise."""
    from active import sims_to

    s = sims_to(n, err, target)
    return (float(n[-1]), ">") if s is None else (s, "<=") if s == n[0] else (s, "=")


def curves(sp, C, out, metric):
    per = {}
    for x in sp["eqs"]:
        nm = x["name"]
        if nm not in C["eqs"]:
            continue
        sc = {}
        for f in glob.glob(f"{out}/scratch/runs/{nm}_N*_s*.json"):
            if ".ckpt" in f:
                continue
            r = json.load(open(f))
            sc.setdefault(r["N"], []).append(r["test"][metric])
        per[nm] = {"nf": x["nf"], "rhs": x["rhs"], "persistence": C["eqs"][nm]["persistence"][metric],
                   "ctx": {k: C["eqs"][nm][f"k{k}"][metric] for k in C["ks"]},
                   "scratch": {N: float(np.mean(v)) for N, v in sorted(sc.items())}, "scratch_seeds": {N: len(v) for N, v in sorted(sc.items())}}
    eqs = [nm for nm, p in per.items() if all(N in p["scratch"] for N in CORE)]
    Ns = [N for N in sorted({N for nm in eqs for N in per[nm]["scratch"]}) if all(N in per[nm]["scratch"] for nm in eqs)]
    mean = {"equations": eqs, "ctx": {k: float(np.mean([per[nm]["ctx"][k] for nm in eqs])) for k in C["ks"]},
            "scratch": {N: float(np.mean([per[nm]["scratch"][N] for nm in eqs])) for N in Ns},
            "persistence": float(np.mean([per[nm]["persistence"] for nm in eqs]))}
    n, e = np.array(Ns, float), np.array([mean["scratch"][N] for N in Ns])
    mean["fewer"] = {}
    slope, icpt = np.polyfit(np.log(n), np.log(e), 1)  # scratch err ~ N^slope
    mean["scratch_powerlaw"] = {"slope": float(slope), "intercept": float(icpt)}
    for k in KS_FEWER:
        s, b = fewer(n, e, mean["ctx"][k])
        mean["fewer"][k] = {"ctx_error": mean["ctx"][k], "scratch_sims": s, "bound": b, "x_fewer": s / k if k else None}
        if b == ">" and slope < 0:  # beyond the largest N: power-law extrapolation, an estimate only
            mean["fewer"][k]["scratch_sims_powerlaw"] = float(np.exp((np.log(mean["ctx"][k]) - icpt) / slope))
    for nm in eqs:
        p = per[nm]
        n1 = np.array(sorted(p["scratch"]), float)
        e1 = np.array([p["scratch"][N] for N in sorted(p["scratch"])])
        p["fewer"] = {k: dict(zip(("scratch_sims", "bound"), fewer(n1, e1, p["ctx"][k]))) for k in KS_FEWER}
    mean["fewer_per_eq_median"] = {k: float(np.median([per[nm]["fewer"][k]["scratch_sims"] for nm in eqs])) for k in KS_FEWER}
    return per, mean


def summary(a):
    sp = json.load(open(f"{a.out}/split.json"))
    C = json.load(open(f"{a.out}/ctx_kshot.json"))
    res = {"note": __doc__.split("Per equation:")[1].strip(), "ckpt": C["ckpt"], "metrics": {}}
    for metric in ("rel_slot0", "raw_slot0_median"):
        per, mean = curves(sp, C, a.out, metric)
        res["metrics"][metric] = {"mean": mean, "per_equation": per}
    json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1)
    plot(res["metrics"]["rel_slot0"], f"{a.out}/scratch_vs_ctx.png")
    m = res["metrics"]["rel_slot0"]["mean"]
    print(json.dumps({"equations": len(m["equations"]), "ctx": {k: round(v, 3) for k, v in m["ctx"].items()},
                      "scratch": {N: round(v, 3) for N, v in m["scratch"].items()},
                      "fewer": {k: (round(v["scratch_sims"], 1), v["bound"]) for k, v in m["fewer"].items()}}), flush=True)


def plot(R, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    m, per = R["mean"], R["per_equation"]
    blue, orange, ink, grid = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
    x0 = 0.5  # k = 0 drawn here
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for nm in m["equations"]:  # per-equation curves, faint
        p = per[nm]
        ax.plot([max(int(k), x0) for k in p["ctx"]], list(p["ctx"].values()), color=blue, lw=0.8, alpha=0.18)
        ax.plot([int(N) for N in p["scratch"]], list(p["scratch"].values()), color=orange, lw=0.8, alpha=0.18)
    ks = [int(k) for k in m["ctx"]]
    ax.plot([max(k, x0) for k in ks], list(m["ctx"].values()), "o-", color=blue, lw=2, ms=6,
            label="v3 conditioned FNO: k examples in context, 0 gradient steps")
    Ns = [int(N) for N in m["scratch"]]
    ax.plot(Ns, list(m["scratch"].values()), "s-", color=orange, lw=2, ms=6, label="FNO trained from scratch on N simulations")
    ax.axhline(m["persistence"], color=ink, lw=1, ls="--", label="persistence u(t) = u(0)")
    over = False  # dotted connectors at equal error: k sims (v3) vs N sims (scratch); "beats N max" once
    for k, f in m["fewer"].items():
        k = int(k)
        if k not in (0, 1, 5, 16) or f["bound"] == "<=" or (f["bound"] == ">" and over):
            continue
        ax.plot([max(k, x0), f["scratch_sims"]], [f["ctx_error"]] * 2, ":", color=ink, lw=1)
        if f["bound"] == ">":
            over = True
            ax.annotate(f"k = {k} beats scratch at N = {f['scratch_sims']:.0f}", (f["scratch_sims"], f["ctx_error"]),
                        xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=7.5, color=ink)
        else:
            lab = f"scratch needs N = {f['scratch_sims']:.0f} to match k = {k}" + (f" ({f['x_fewer']:.0f}x fewer)" if k else "")
            ax.annotate(lab, (max(k, x0), f["ctx_error"]), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=ink)
    ticks = [x0] + [t for t in (1, 2, 4, 8, 16, 32, 64, 128) if t <= max(Ns + ks)]
    ax.set(xscale="log", yscale="log", xlabel="simulations of the new equation (k for v3, N for scratch; 0 at left)",
           ylabel="test relL2, primary field (normalized space)")
    ax.set_xticks(ticks, ["0"] + [str(t) for t in ticks[1:]])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    lo, hi = ax.get_ylim()
    yt = [t for t in (0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5) if lo <= t <= hi]
    ax.set_yticks(yt, [f"{t:g}" for t in yt])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(which="both", labelsize=8, colors=ink)
    ax.grid(True, which="major", color=grid, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    ax.set_title(f"Unseen equations from the pretraining families: {len(m['equations'])} fresh draws\n"
                 "(mean over equations; faint lines = each equation)", fontsize=9, color=ink)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "summary"])
    ap.add_argument("--data", default="/workspace/data/C")
    ap.add_argument("--tag", default="C")
    ap.add_argument("--out", default="/workspace/results/scratch_vs_ctx")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--per_nf", type=int, default=5, help="equations per field count")
    ap.add_argument("--min_change", type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"split": split, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()

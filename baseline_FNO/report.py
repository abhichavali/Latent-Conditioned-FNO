"""results.json -> error_vs_data.png + RESULTS.md (written by train.py at the end of a run)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAMES = {"swe": ("Shallow water", "η"), "ap": ("Aliev–Panfilov", "u")}


def table(r):
    """{N: (mean, std) per metric} for the primary field (index 0), averaged over seeds."""
    by = {}
    for k, v in r.items():
        if "/" in k:
            by.setdefault(v["N"], []).append(v)
    return {N: {m: (np.mean([x[m][0] for x in vs]), np.std([x[m][0] for x in vs]), len(vs))
                for m in ("one_step", "rollout_mean", "rollout_final")} for N, vs in sorted(by.items())}


def slope(t, m):
    N = np.array(list(t))
    return float(np.polyfit(np.log(N), np.log([t[n][m][0] for n in N]), 1)[0])


def write(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (name, r) in zip(axes, res.items()):
        t = table(r)
        N = list(t)
        for m, lab, c in (("one_step", "one step", "#4c72b0"), ("rollout_mean", "19-step rollout (mean)", "#dd8452"),
                          ("rollout_final", "19-step rollout (final)", "#c44e52")):
            ax.errorbar(N, [t[n][m][0] for n in N], yerr=[t[n][m][1] for n in N], marker="o", ms=4, capsize=2, label=lab, color=c)
        ax.set(xscale="log", yscale="log", xlabel="training initial conditions", title=f"{NAMES[name][0]} ({NAMES[name][1]})")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("test normalized RMSE")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/error_vs_data.png", dpi=150)

    doc = os.path.join(os.path.dirname(os.path.abspath(out)), "RESULTS.md")
    L = ["# FNO baseline: error vs. training data", "",
         "No prior, no active learning: a plain FNO2d (4 Fourier layers, width 64, 16 modes, residual one-step map "
         "u(t) → u(t+Δt), MSE, AdamW + cosine, batch 16) trained on N random-IC rollouts of each held-out demo "
         "equation and tested on 128 unseen ICs. Data: generic spectral solver, 128², periodic, 20 evenly spaced frames "
         "per rollout (SWE Δt = 8 s, AP Δt = 2.5 t.u.). Steps per run = clip(15 epochs, 1500, 6000); 3 seeds (data subset "
         "+ init) for N ≤ 64. Error = normalized RMSE of the primary field (RMSE / its test-set std); rollout = 19 autoregressive "
         "steps from the IC.", "", "![error vs data](results/error_vs_data.png)", ""]
    for name, r in res.items():
        t = table(r)
        L += [f"## {NAMES[name][0]} ({NAMES[name][1]})", "",
              "| train ICs | one step | rollout mean | rollout final |", "|---:|---:|---:|---:|"]
        f = lambda x: f"{x[0]:.3g}" + (f" ± {x[1]:.2g}" if x[2] > 1 else "")
        L += [f"| {N} | {f(v['one_step'])} | {f(v['rollout_mean'])} | {f(v['rollout_final'])} |" for N, v in t.items()]
        L += ["", f"Log-log slope (error ∝ N^s): one step s = {slope(t, 'one_step'):.2f}, rollout mean s = {slope(t, 'rollout_mean'):.2f}."]
        if "viz" in r:
            e = r["viz"]["rel_l2_at"]
            ic = r["viz"]["interp_check"]
            L += [f"Largest model, held-out IC rolled out {r['viz']['steps']} steps (3× training horizon): nRMSE "
                  + ", ".join(f"{v:.3g} @ step {k}" for k, v in e.items()) + ".",
                  f"Demo playback inserts frames between FNO steps by motion-compensated interpolation; on solver truth it "
                  f"reconstructs the skipped frames with rel L2 {ic['flow']:.3g} vs {ic['linear']:.3g} for linear crossfade."]
        L += [""]
    L += ["## Notes", "",
          "- N = 512 and 1024 both hit the 6000-step cap (~5 and ~2.5 epochs), so the flattening at the top end is at "
          "least partly a compute limit, not only a data limit.",
          "- Long-horizon errors fall once the truth itself decays (the tsunami is absorbed by the sponge coast), so "
          "steps 38/57 for SWE mostly measure the quiet basin after impact.", ""]
    L += ["## Viewing rollouts", "",
          "`results/viz/{swe,ap}_{truth,fno}/` are in the viz page formats (solver truth vs FNO, same held-out IC; "
          "SWE land is a damping sponge in the solver, drawn as beach and hills):", "",
          "```sh", "python3 -m http.server 8000   # repo root",
          "open 'http://localhost:8000/viz/tsunami/?data=../../baseline_FNO/results/viz/swe_fno/'",
          "open 'http://localhost:8000/viz/heart/?data=../../baseline_FNO/results/viz/ap_fno/&geometry=sheet'",
          "# swap _fno for _truth to compare", "```", ""]
    open(doc, "w").write("\n".join(L))
    print("wrote", doc)

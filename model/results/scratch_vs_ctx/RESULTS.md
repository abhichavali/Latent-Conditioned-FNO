# Conditioned FNO (v3) vs FNO from scratch, on unseen equations (2026-10-26)

**Setup:**
- **Equations:** 10 fresh equations drawn from the pretraining generator with a new seed (99), so they are never seen in training: 5 two-field, 5 three-field.
- **Data:** each has 152 train and 40 test windows, where one "simulation" is one 15-frame window.
- **v3:** the conditioned FNO adapts with *zero gradient steps*. Its transformer reads the equation graph plus the first k train windows.
- **Scratch:** an FNO with the same backbone, trained only on the first N train windows (early stopping on a held-out 10%).
- **Shared evaluation:** both are scored on the same 40 test windows with the same metric: rel L2 of the primary field in the model's normalized space, averaged over frames 1–14.
- **Code:** `model/queue_svc.sh`, `model/scratch_vs_ctx.py`, `baselines.py eq`, `fewshot_ctx.py` (C-set k-shot).

Mean over the 10 equations (summary.json, scratch_vs_ctx.png):

| simulations | 0 | 1 | 2 | 5 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| v3 conditioned (k) | **0.416** | **0.381** | 0.357 | **0.353** | | 0.333 | 0.328 | 0.327 | |
| scratch FNO (N) | | | | | 0.907 | 0.610 | 0.521 | 0.429 | **0.397** |

Persistence (u(t) = u(0)) scores 0.73.

**Headline:**
- **Equation alone:** the conditioned model's zero-shot error (0.416) takes a scratch FNO about **83 simulations** to match.
- **One simulation:** it reaches 0.381, which scratch does not reach within 128. That is **≥ 128× fewer** (a power-law fit to the scratch curve puts the match at about 110–190).
- **Five simulations:** 0.353, again **≥ 25× fewer** than scratch at its best (128).

**Caveats:**
- single seed per configuration;
- 10 equations;
- fresh draws from the *pretraining families* (in-distribution), not new physics. On genuinely new physics (tsunami, wildfire) the conditioned model does not beat fine-tuning (see `model/results/ctx/RESULTS.md`);
- the conditioned model is far larger (515M params) and pretrained on ~200k simulations, while scratch sees only N;
- per equation, it fails on one three-field equation, where it is worse than persistence and does not improve with context (faint blue line near 0.8–1.1);
- context saturates by about 16 simulations, while scratch keeps improving.

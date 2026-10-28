# v3 vs FNO from scratch on unseen equations (2026-10-26)

10 new equations drawn from the pretraining generator with a new seed (99), 5 two-field and 5 three-field. Each has
152 train and 40 test windows (one "simulation" = one 15-frame window).

- v3 adapts with zero gradient steps: the transformer reads the equation graph plus the first k train windows.
- Scratch is the same FNO backbone trained only on the first N train windows (early stopping on 10% held out).
- Both are scored on the same 40 test windows: rel L2 of the primary field in normalized space, frames 1-14.

Code: `model/queue_svc.sh`, `model/scratch_vs_ctx.py`, `baselines.py eq`, `fewshot_ctx.py fresh`.

Mean over the 10 equations (`summary.json`, `scratch_vs_ctx.png`):

| simulations | 0 | 1 | 2 | 5 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| v3 (k) | 0.416 | 0.381 | 0.357 | 0.353 | | 0.333 | 0.328 | 0.327 | |
| scratch (N) | | | | | 0.907 | 0.610 | 0.521 | 0.429 | 0.397 |

Persistence (u(t) = u(0)) is 0.73.

- With just the equation (k = 0), v3 gets 0.416. Scratch needs about 83 simulations to match that.
- With 1 simulation v3 gets 0.381, which scratch doesn't reach even at 128 (a power-law fit says ~110-190).
- With 5 it gets 0.353, so at least 25x fewer than scratch's best.

Caveats: single seed, only 10 equations, and these are from the pretraining families, not new physics (on the
tsunami / wildfire targets v3 doesn't beat fine-tuning, see `model/results/ctx/RESULTS.md`). v3 is also much
bigger (515M params) and pretrained on ~200k simulations while scratch only sees N. It fails on one three-field
equation, where it's worse than persistence and doesn't improve with context (the blue line near 0.8-1.1). Context
saturates around 16 sims while scratch keeps improving.

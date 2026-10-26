# v3: amortized, simulation-conditioned latent (final, 2026-10-26)

**Model:** a transformer reads the equation graph G and any number of the equation's simulations C, and outputs q(z | G, C). z (d = 128) drives FiLM, bank mixtures and rank-8 weight deltas of a 515M-param FNO (width 96, 6 layers).

**Training:**
- Neural-Process objective on A∪B (5,120 equations), from scratch, 110k steps on one RTX 5090 (log.jsonl, config.yaml).
- Context: 0–64 windows per equation.
- Graph-term dropout 0.5.
- Encoder frozen for the first 5%; β = 0 (KL off) after the freeze, encoder lr ×0.1.

**Evaluation:**
- Branch (a): adaptation is one forward pass of q(z | G, C) with no gradient steps.
- Branch (b): adds refinement of z on the N sims.
- Branch (c): full fine-tuning was started but stopped when the GPU box was shut down; no (c) numbers.
- Single seed.

## Held-out equation families (rehearsal: 60 equations whose term structures were held out of pretraining)

rel L2 of the primary field (rehearsal_kshot.json, 685 windows, final checkpoint):

| context sims | 0 | 1 | 2 | 5 | 10 | 16 | 32 | 64 | persistence |
|---|---|---|---|---|---|---|---|---|---|
| rel_slot0 | 0.448 | 0.383 | 0.352 | 0.311 | 0.297 | 0.295 | 0.291 | 0.292 | 1.318 |

Five simulations cut the error by 31%, all in one forward pass. Validation during training (log.jsonl) also showed that with the equation graph removed, 16 simulations reach the same error as with it (≈ 0.42 rel_all vs 0.42), so the transformer extracts the dynamics from the simulations themselves. (Metric note: log.jsonl "val" numbers are rel_all on a smaller validation batch; the table above is rel_slot0 on the full rehearsal set.)

## Held-out targets (never in pretraining), raw_slot0, active acquisition, 5 sims per round

| target | branch | 0 | 1 | 2 | 5 | 10 | 25 | 100 | v1 fine-tuned @5 / @100 | scratch FNO @8 |
|---|---|---|---|---|---|---|---|---|---|---|
| tsunami (SWE) | a | 1.81 | 1.20 | 1.15 | 1.05 | 1.03 | 1.02 | 1.02 | 0.75 / 0.31 | 1.09 |
|  | b | 1.81 | 1.19 | 1.13 | 1.03 | 1.02 | 0.99 | 0.97 | | |
| heart (AP) | a | 2.82 | 1.02 | 0.90 | 0.91 | 0.93 | 0.99 | 1.06 | 0.90 / 0.76 | 1.64 |
|  | b | 2.82 | 0.92 | 0.87 | 0.89 | 0.86 | 0.88 | 0.96 | | |
| wildfire | a | 0.97 | 0.94 | 0.93 | 0.91 | 0.89 | 0.87 | 0.86 | 0.69 / 0.50 | 0.91 |
|  | b | 0.97 | 0.93 | 0.92 | 0.89 | 0.87 | 0.85 | 0.80 | | |

## Verdict

- **In distribution, the formulation works.** On held-out equation families, error falls steadily with context (0.45 → 0.31 with 5 simulations), and simulations alone substitute for the equation.
- **On genuinely new physics, it doesn't beat fine-tuning yet:**
  - **Heart:** with 1–5 simulations, the forward-pass latent matches v1's fine-tuned model at 5 (0.87–0.89 vs 0.90). It then plateaus, while fine-tuning keeps improving.
  - **Tsunami:** stays at about persistence level (~1.0).
  - **Wildfire:** improves slowly (0.97 → 0.80) but trails v1's fine-tuning (0.50 at 100).
  - The latent's expressiveness, not the update rule, looks like the limit outside the pretraining families.
- **Caveats:**
  - single seed;
  - the model is 3.4× larger than v1 (size confound);
  - β = 0 means q's uncertainty is not calibrated, so branch (b) uses an anchor scaled by the across-equation spread of the latent;
  - (c) not run.

## Getting here (all fixed)

1. NaN at lr 5e-4.
2. Context ignored, because the graph already carried every coefficient → graph-term dropout.
3. Latent collapsed to a single z → encoder-freeze warm-up and β off.
4. The disk filled with 6.25 GB checkpoints → keep = 1.
5. Evaluation OOM, because the FNO was captured as a 2 GB jit constant → passed as an argument.

## Artifacts (local, gitignored: `artifacts/`)

- `ckpt/v3_ctx/`: the final v3 checkpoint.
- `ckpt/v1_AB/`: the v1 A∪B step-200k checkpoint.
- `ckpt/final_models/`: the v1 demo exports.
- `results/ctx/`: the full results, including acquired sims.

The GPU box was destroyed afterwards. Datasets A and B (~200 GB) were not copied; they regenerate from `data_collection/` (see its README).

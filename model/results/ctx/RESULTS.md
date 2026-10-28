# v3 results (2026-10-26)

v3 is a transformer that reads the equation graph G and any number of simulations C and outputs q(z | G, C). z
(d = 128) drives FiLM, bank mixtures and rank-8 weight deltas of a 515M param FNO (width 96, 6 layers).

Training: Neural Process objective on A∪B (5120 equations) from scratch, 110k steps on one RTX 5090. 0-64 context
windows per equation, graph-term dropout 0.5, encoder frozen for the first 5%, then β = 0 (no KL) and encoder lr
x0.1. See `log.jsonl` and `config.yaml`.

Eval branches: (a) z from one forward pass, no gradient steps; (b) (a) plus refining z on the N sims; (c) full
fine-tuning, which didn't finish before the GPU box was shut down. Single seed.

## Held-out equation families

60 rehearsal equations whose term structures were held out of pretraining, rel_slot0 (`rehearsal_kshot.json`, 685
windows, final checkpoint):

| context sims | 0 | 1 | 2 | 5 | 10 | 16 | 32 | 64 | persistence |
|---|---|---|---|---|---|---|---|---|---|
| rel_slot0 | 0.448 | 0.383 | 0.352 | 0.311 | 0.297 | 0.295 | 0.291 | 0.292 | 1.318 |

5 simulations cut the error by 31% in one forward pass. In validation during training, with the graph removed, 16
simulations got the same error as with it (≈ 0.42 rel_all both ways), so the model can get the dynamics from the
simulations alone. (The log.jsonl val numbers are rel_all on a smaller batch, the table is rel_slot0 on the full set.)

## Held-out targets

raw_slot0, active acquisition, 5 sims per round:

| target | branch | 0 | 1 | 2 | 5 | 10 | 25 | 100 | v1 fine-tuned @5 / @100 | scratch FNO @8 |
|---|---|---|---|---|---|---|---|---|---|---|
| tsunami (SWE) | a | 1.81 | 1.20 | 1.15 | 1.05 | 1.03 | 1.02 | 1.02 | 0.75 / 0.31 | 1.09 |
|  | b | 1.81 | 1.19 | 1.13 | 1.03 | 1.02 | 0.99 | 0.97 | | |
| heart (AP) | a | 2.82 | 1.02 | 0.90 | 0.91 | 0.93 | 0.99 | 1.06 | 0.90 / 0.76 | 1.64 |
|  | b | 2.82 | 0.92 | 0.87 | 0.89 | 0.86 | 0.88 | 0.96 | | |
| wildfire | a | 0.97 | 0.94 | 0.93 | 0.91 | 0.89 | 0.87 | 0.86 | 0.69 / 0.50 | 0.91 |
|  | b | 0.97 | 0.93 | 0.92 | 0.89 | 0.87 | 0.85 | 0.80 | | |

## Takeaways

Inside the pretraining families it works: error drops steadily with context and simulations can stand in for the
equation.

On new physics it doesn't beat fine-tuning yet. Heart: with 1-5 sims it matches v1 fine-tuned at 5 (0.87-0.89 vs
0.90) but then plateaus while fine-tuning keeps going. Tsunami stays around persistence (~1.0). Wildfire improves
slowly (0.97 -> 0.80) but trails v1 (0.50 at 100). My guess is the latent isn't expressive enough outside the
training families, not that the update rule is wrong.

Caveats: single seed, v3 is 3.4x bigger than v1, β = 0 means q's uncertainty isn't calibrated (so (b) uses an
anchor scaled by the across-equation spread of z), and (c) never ran.

## Problems along the way

- NaN at lr 5e-4.
- Context was ignored because the graph already had every coefficient. Fixed with graph-term dropout.
- The latent collapsed to a single z. Fixed with the encoder freeze warm-up and β off.
- 6.25 GB checkpoints filled the disk, so keep = 1.
- Eval ran out of memory because the FNO was captured as a 2 GB jit constant. Now it's passed as an argument.

## Artifacts

Local only, gitignored under `artifacts/`: `ckpt/v3_ctx/` (final v3), `ckpt/v1_AB/` (v1 A∪B step 200k),
`ckpt/final_models/` (v1 per-target exports), `results/ctx/` (full results incl. acquired sims). The GPU box is
gone and datasets A and B weren't copied, they regenerate from `data_collection/`.

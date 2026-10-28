# Model

v1: an FNO steered by a per-equation latent z (d_z = 32, FiLM + a mixture of spectral filter banks), plus a graph
encoder that maps the equation graph to a prior N(mu, sigma^2) over z. On a new PDE we start from that prior and
update with a small simulation budget (active learning + fine-tuning).

v3: no latent table. A transformer reads the equation graph and any number of simulations and outputs q(z | G, C),
trained with a Neural Process objective. Wider FNO (515M params).

Everything is JAX / equinox and was run on rented GPUs, so paths below are `/workspace/...` on the box.

## Files

- `config/`: `A.yaml` is the commented v1 config, `B.yaml` / `AB.yaml` only change the data roots, `ctx.yaml` is v3,
  `scratch*.yaml` are the from-scratch baselines, `smoke.yaml` is a 1-minute test
- `data/`: dataset loading and normalization (`adapter.py`), canonical equation graphs (`canon.py`), splits
  (`splits.py`), v3 context cache (`ctx.py`)
- `models/`: conditioned FNO (`fno.py`), graph encoder (`encoder.py`), v3 context encoder + wider FNO (`context.py`)
- `train.py`, `run.sh`: v1 training; `run.sh` restarts from the latest checkpoint until `DONE`
- `train_ctx.py`: v3 training
- `targets.py`, `targets_fire.py`: the held-out targets (shallow water tsunami, Aliev-Panfilov heart with a scar,
  wildfire) with samplers, solver and fixed 100-IC test sets
- `eval.py`: zero-shot eval, latent collapse test, prior calibration
- `active.py`, `active_v2.py`: few-shot active learning for v1 (`active_v2` is an EPIG-style variant)
- `fewshot_ctx.py`: few-shot / k-shot eval for v3
- `baselines.py`: FNO from scratch on N simulations
- `scratch_vs_ctx.py`: v3 vs scratch on fresh equations
- `queue_ctx.sh`, `queue_svc.sh`: unattended eval queues I ran on the GPU box

## Data

- Up to 3 field slots and 2 static slots (padded, with presence flags), plus a mask channel and (x, y) coordinates.
- Each trajectory is normalized by its IC (subtract mean, divide by std). The mean / std go in as extra channels so
  the absolute level isn't lost.
- Time is t = k / (frames - 1); log T and log L go into the conditioning.
- Loss is relative L2 per (sample, time, field), capped at 4 in training. Slot 0 is the main field.
- Splits: "rehearsal" holds out ~5% of equations by term structure (pairs/triples of library terms), "val_ic" holds
  out 10% of ICs of training equations. The targets aren't in either dataset.

Switching datasets is just `data.roots` in the config. `AB.yaml` warm-starts from the A run (`train.init_from`) and
keeps A's splits (`split.inherit`).

## Running

```sh
bash run.sh config/smoke.yaml                                   # smoke test
setsid nohup bash run.sh config/A.yaml > /dev/null 2>&1 < /dev/null & disown   # v1 pretraining
python targets.py --out /workspace/targets                      # target test sets
python eval.py --run /workspace/runs/pretrain_A [--ckpt CK]     # zero-shot eval
python active.py run --run RUN --ckpt CK --target swe|ap|fire --method active|random|generic
python active.py summary --results /workspace/results/<run>_s<step>

python data/ctx.py --config config/ctx.yaml                      # v3 context cache
TRAIN=train_ctx.py bash run.sh config/ctx.yaml                   # v3 training
python fewshot_ctx.py run --run /workspace/runs/ctx --ckpt CK --target swe --method active
python fewshot_ctx.py rehearsal --run /workspace/runs/ctx --ckpt CK

python baselines.py gen --config config/scratch.yaml
python baselines.py sweep --config config/scratch.yaml --target swe
python baselines.py summary --config config/scratch.yaml
```

A run dir has `config.yaml`, `splits.json`, `log.jsonl`, `tb/`, and `ckpt/` (last 3, every 15 min, also on
SIGTERM). Resuming is exact since data order only depends on (seed, step). To stop: kill `run.sh` first, then
`pkill -TERM -f train.py`.

Metrics: relL2 over frames 1..14 from the IC. `raw` is in physical units (the headline number), `rel` is in the
normalized space (same as the training loss).

## Shallow water units

SWE runs in mixed units (km, ks, m) so that log L and log T land inside the pretraining range. In pure SI the
domain is way bigger than anything in pretraining, and in pure km the IC std falls below the normalization floor.
`targets.check_swe_mix` checks it matches the SI solver. No pretraining equation has a second static, so for SWE
the sponge slot copies slot 0's weights (`eval.adapt_statics`).

## Scratch baseline

Same backbone with `dz: 0` (no latent), trained on N simulations of one target, early stopping on 10% held out.
Test relL2 on the 100-IC test sets (SWE: seed 0, AP: seeds 0 and 1):

| N | SWE raw η | SWE raw all | SWE rel η | AP raw u | AP raw all | AP rel u |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.089 | 1.026 | 1.090 | 1.641 ± 0.075 | 1.163 ± 0.033 | 0.763 ± 0.001 |
| 16 | 0.548 | 0.607 | 0.544 | 1.113 ± 0.072 | 0.848 ± 0.029 | 0.627 ± 0.000 |
| 32 | 0.417 | 0.511 | 0.413 | 1.107 ± 0.016 | 0.812 ± 0.002 | 0.562 ± 0.013 |
| 64 | 0.284 | 0.344 | 0.282 | 0.880 ± 0.076 | 0.659 ± 0.044 | 0.446 ± 0.015 |
| 128 | 0.203 | 0.205 | 0.201 | 0.543 ± 0.013 | 0.453 ± 0.011 | 0.360 ± 0.013 |
| 256 | 0.122 | 0.117 | 0.121 | 0.412 ± 0.019 | 0.354 ± 0.006 | 0.288 ± 0.004 |
| 512 | 0.078 | 0.073 | 0.077 | 0.286 ± 0.003 | 0.244 ± 0.007 | 0.215 ± 0.002 |
| persistence | 2.353 | 1.451 | 2.340 | 1.784 | 1.460 | 1.083 |

Small N overfits a lot (train 0.02-0.04, val 0.3-1.2). SWE N = 256 / 512 hit the 20k step cap so they're a bit
under-trained. AP raw u goes above 1 for small N because u is near 0 at rest, which blows up the per-frame ratio.

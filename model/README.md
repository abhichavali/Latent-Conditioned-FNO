# Latent-conditioned FNO + graph-encoder prior

A frozen-after-pretraining FNO steered by a per-equation latent `z` (d_z = 32), and a graph encoder that maps the
equation graph to a prior `N(mu, diag sigma^2)` over `z`. On a new PDE we start from that prior and update it with a
small simulation budget (later phases).

| File | Role |
|---|---|
| `config/A.yaml` | pretraining on dataset A (all settings, commented); `B.yaml`, `AB.yaml` change only `data.roots`; `smoke.yaml` is a tiny run |
| `data/adapter.py` | equation table over dataset roots (ids `tag/eq_XXXX`), duplicate / non-finite window removal, test-time normalization, mmap batch stream with a prefetch thread pool |
| `data/canon.py` | canonical equation graphs (sympy, cached): one expression tree per additive term, node features |
| `data/splits.py` | seeded splits: rehearsal by structure, val-IC, train |
| `models/fno.py` | conditioned, time-conditioned FNO (FiLM + softmax mixture of spectral filter banks, zero-init conditioning) |
| `models/encoder.py` | graph encoder p_phi(z \| G) |
| `train.py` | objective, logging, checkpoints, resume |
| `run.sh` | restart loop (resumes from the latest checkpoint until `DONE`) |
| `targets.py` | SWE coastal tsunami and Aliev–Panfilov-with-scar: specs, canonical graphs, knob samplers, solver, fixed test sets |

## Data interface

- Field slot k = `spec.fields[k]` = graph field node k (padded to 3, presence flags); static slot k = `spec.static[k]` =
  graph static node k (padded to 2). One reserved mask channel `chi` (zeros), coordinates (x, y) in [0, 1)^2.
- Normalization per trajectory and field, from the IC only: subtract the IC spatial mean, divide by the IC std
  (floor 1e-3; constant ICs such as momentum at rest use 1.0). The model also gets `tanh(mean/std/5)` and
  `log10(std)/3` per slot (and `log10(rms)/3` per static) as constant channels, so absolute levels are not lost.
  Statics are divided by their RMS. Stored data is de-normalized first (`u * scale + shift`).
- Time t = k / (frames - 1) in [0, 1]; log T and log L (T = (frames - 1) * frame_dt) go into the conditioning, divided by 4.
- Loss: relative L2 per (sample, query time, field slot), masked by presence, averaged; in training each term is
  soft-capped at 4 (`rel_cap`). Primary observable = slot 0 (`rel_slot0` in the logs).
- Windows that are duplicates (same pool IC and start frame) are kept once, and windows with non-finite values are
  dropped (cached in `cache_dir/nonfinite_<tags>.json`).

## Switching datasets

Only `data.roots` changes: `[{tag: A, path: ...}]`, `[{tag: B, path: ...}]`, or both for A ∪ B (ids stay unique by tag).
Splits, normalization, counts of equations / ICs / frames are all read from disk. On `pdebox3` (RTX 5090) the data
lives in `/workspace/data/A` and `/workspace/data/B`.

**Warm start A ∪ B from A** (`config/AB.yaml`): `train.init_from: /workspace/runs/pretrain_A` loads that run's latest
checkpoint into a run that has none of its own: backbone and encoder weights as they are, latent rows copied by
equation id (`A/eq_0012`), B rows initialised N(0, 0.01^2), fresh optimizer. `split.inherit:
/workspace/runs/pretrain_A/splits.json` keeps every A equation's split and val-IC windows and reuses A's held-out
structures, so B equations containing them are rehearsal too.

## Running

```sh
cd /workspace/repo/model
bash run.sh config/smoke.yaml                                   # ~1 min smoke test (foreground)
setsid nohup bash run.sh config/A.yaml > /dev/null 2>&1 < /dev/null & disown   # pretraining, detached
python targets.py --out /workspace/targets                      # SWE / AP test sets (100 ICs each, ~20 s)
```

Run dir `<run_root>/<name>/` (`/workspace/runs/pretrain_A/`): `config.yaml` and `splits.json` (frozen at first start),
`log.jsonl`, `tb/` (TensorBoard), `ckpt/step_XXXXXXXX.eqx` (model, encoder, latents, optimizer state; last 3, every
15 min) with a `.json` sidecar (step, training seconds), `stdout.log`, `start_time`, `DONE`. Log lines carry
`elapsed_h` (wall clock since the run's first start) and `train_h` (training time summed over restarts). Data order and per-step RNG depend only on (seed, step), so a resumed run
continues exactly. Restarting `run.sh` (or the loop itself after a crash) resumes from the latest checkpoint.

Watch: `tail -f /workspace/runs/pretrain_A/stdout.log`, or
`tensorboard --logdir /workspace/runs --port 6006` then `ssh -N -L 6006:localhost:6006 pdebox3`.

Stop cleanly (keeps checkpoints; `train.py` checkpoints on SIGTERM): kill the loop first, then python:
`pkill -f '[r]un.sh config/A.yaml'; pkill -TERM -f '[t]rain.py --config config/A.yaml'`.

## Validation logged every 10 min and at every (re)start

On fixed val-IC batches (held-out ICs of training equations) and rehearsal batches (held-out structures):
`persistence` (u(t) = u(0), computed once), `z_table` (latent table), `prior_mu` (z = mu_phi(G), prior only),
`swap` (table latents rolled across the batch's equations); `ratio_to_persistence`, `swap_ratio` = swap / z_table
(> 1 means the backbone uses z), plus uncapped means and medians. Train lines carry relL2, NLL, mean sigma_phi,
|z| stats, encoder-sample fraction, schedules, steps/s and `elapsed_h` since the run's first start.

## Splits (`splits.json`)

- rehearsal: a seeded greedy pick of library-term pairs / triples (`kind:item`, as in `coverage.py`) each covering
  0.8–2.5% of equations, until ~5% of equations are held out; every equation containing one is held out. Each
  term of a chosen structure still appears in training, and terms that form the SWE / AP pieces are never used.
- val_ic: 10% of the IC groups of each training equation with at least 10 unique ICs.
- the SWE / AP targets are outside both datasets.

## Targets (`targets.py`, test sets in `/workspace/targets/{swe,ap}/`, pretraining layout + `params.npy`)

| | fields / statics | L | frame_dt | T | knobs |
|---|---|---|---|---|---|
| swe | eta, u, v / b, s (fixed demo coast) | 4000 m | 10 s (0.25 s x 40) | 140 s | 2 fault sources: on, x in [0.08, 0.36], y in [0.2, 0.8], angle, amp, width, length, subsidence lobe (on, side, offset, amp) |
| ap | u, v / D (scar, per IC) | 64 | 2.5 (0.02 x 125) | 35 | 2 broken fronts (on, x, y, angle, width, half length, wake, v amp), 2 stimuli (on, x, y, radius), scar (x, y, radius, D_min in [0.02, 0.05]) |

`targets.sample(name, n, rng)` -> (params, ics, statics); `targets.features(name, params)` -> knob features in [0, 1]
for k-means; `targets.simulate(name, ics, statics)` -> trajectories; `targets.canonical_graph(name)`. Load a test set
with `data.adapter.Eq("SWE", "/workspace/targets/swe")` like any training equation.

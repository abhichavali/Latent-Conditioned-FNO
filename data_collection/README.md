# PDE data collection

Every PDE — training equations and the two held-out demos alike — is a JSON **spec**: evolving scalar fields
`u(x, y, t)` on a periodic square, optional static fields `a(x, y)`, and one sympy right-hand side per field.

```json
{"fields": ["u", "v"], "static": ["a"], "L": 40.0, "N": 128,
 "rhs": {"u": "dx(a*dx(u)) + dy(a*dy(u)) + 8*u*(1 - u)*(u - 0.1) - v",
         "v": "0.01*(u - v)"},
 "dt": 0.05, "substeps": 12, "frames": 15}
```

Operators: `dx dy lap`, `+ - * / **`, `sin cos exp tanh log Abs`. Arrays are `(..., y, x)`; `dx` acts on the last axis.

**Graph** (`pde.graph(spec)`): the expression DAG of all right-hand sides, identical subexpressions shared.
Nodes are `field i`, `static i`, `const value`, operators (`add mul pow dx dy lap sin tanh abs …`) and one `dt i`
root per field; edges are `[child, parent, arg position]` (position matters only for `pow`).

## Files

| File | Role |
|---|---|
| `pde.py` | spec → graph / LaTeX; generic exponax ETDRK solver for any spec (one compile per spec) |
| `template.py` | 2- and 3-field coefficient templates: an equation is a coefficient matrix over a fixed term library, so one compile serves all equations, vmapped over equations × ICs; specs are generated from the coefficients |
| `sample.py` | equation families (transport/waves near SWE, excitable/reaction–diffusion near Aliev–Panfilov, pattern formation, free composition) + SWE / AP building-block subsets, mutations and term dropout; GPU IC pools (7 pattern families, patches, wakes, vortex seeds); max-entropy (greedy log-det over bounded spectral/amplitude/rate features) window selection, capped at 25% per IC family and 2/3 early windows |
| `generate.py` | per-equation dt / frame-interval calibration, pool rollout over `--windows` consecutive 14-step windows, selection of `--keep` windows, writer; resumable, `--id-offset` lets several GPUs share one `--out` |
| `coverage.py` | building-block coverage: equations and partner terms per library term, SWE / AP pieces alone and in pairs, terms that always co-occur; for the sampler (`--sample 4096`) or a dataset (`--out DIR`, writes `coverage.json`) |
| `merge_extra.py` | merges a `--supplement` output into a dataset in place (numpy only) |
| `demos.py` | shallow water and Aliev–Panfilov as specs, simulated and written for `viz/tsunami` and `viz/heart` |

## Dataset layout (`/workspace/pdegen/out` on the GPU box)

`index.jsonl` (one line per equation) and `eq_XXXX/`:
- `spec.json` — spec + `latex`, `graph`, `template` coefficients, `norm`, per-sample IC family and window start
- `u.npy` — float16 `(ICs, 15, fields, 128, 128)` = `(u - norm.shift) / norm.scale`
- `static.npy` — float16 `(ICs, 128, 128)`, raw, iff `spec["static"]` is non-empty

## Datasets

**A** — 1024 equations x 73 ICs, pre-v2 sampler (before `ec1a5f0`): 512-IC pool, 2 windows, fixed per-frame change 0.12.

```sh
python generate.py --out /workspace/pdegen/out --pool 512
```

**B** — 4096 equations x 30 ICs, current sampler: 18 generated, one worker per GPU (worker 1: `CUDA_VISIBLE_DEVICES=1 ... --id-offset 2048 --seed 2`):

```sh
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.9 python generate.py --out /workspace/pdegen/out \
    --n 2048 --keep 18 --pool 128 --windows 3 --target 0.06 0.24 --permute 0.5 --id-offset 0 --seed 1
```

- building blocks: 3-field transport equations take a random subset of the SWE pieces {flux, adv, grad, bathy},
  composite excitable ones a subset of the AP pieces {cubic, uv, gate}, so each appears alone and in pairs; the
  holdout cores (flux+adv+grad, cubic+uv+gate) are never emitted
- dropout: in 60% of equations each non-diffusive, non-core term is dropped with p = 0.1
- field-label permutation: with p = 0.5 the saved field order is shuffled
- frame interval: each equation's per-frame change target is log-uniform in [0.06, 0.24]
- 128 ICs x 3 consecutive windows = 384 candidate windows, 18 kept

Extra spec keys in B: `pieces` (SWE / AP blocks present, read off the coefficients), `perm` (saved field j is
internal field `perm[j]`; `fields`, `rhs`, `ic`, `norm` already renamed to match), `target_change`.

then 12 more per equation by a supplement pass (same pools re-simulated bit-exactly, stored picks forced first),
merged in place:

```sh
python generate.py --supplement /workspace/pdegen/out --extra 12 --out /workspace/pdegen/extra --pool 128 --windows 3 \
    --n 2048 --id-offset 0             # and --id-offset 2048 on GPU 1; --check 5 first verifies the reproduction
python merge_extra.py B_DIR EXTRA_DIR 18 30   # resumable; removes each supplement dir once merged
```

The 18-IC selection had a bug, fixed in `sample.select`: one blown-up candidate window (inf / nan) poisoned the
feature normalization, so 1280 equations picked only ICs 0 and 1 of the first window, alternating (45 of them with
nan frames).  The supplement reselects those 30 windows from scratch; the other 2816 keep their 18.

```sh
python coverage.py --sample 4096                                   # what the sampler draws
JAX_PLATFORMS=cpu python coverage.py --out /workspace/pdegen/out   # a dataset; writes out/coverage.json
python ../viz/dataset/serve.py --data /workspace/pdegen/out        # viewer, then: ssh -N -L 8000:localhost:8000 <box>
```

Self-checks: `python pde.py`, `python template.py`, `python sample.py`.

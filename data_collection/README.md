# Data collection

Every PDE is a JSON spec: scalar fields `u(x, y, t)` on a periodic square, optional static fields `a(x, y)`, and a
sympy right-hand side per field.

```json
{"fields": ["u", "v"], "static": ["a"], "L": 40.0, "N": 128,
 "rhs": {"u": "dx(a*dx(u)) + dy(a*dy(u)) + 8*u*(1 - u)*(u - 0.1) - v",
         "v": "0.01*(u - v)"},
 "dt": 0.05, "substeps": 12, "frames": 15}
```

Operators: `dx dy lap`, `+ - * / **`, `sin cos exp tanh log Abs`. Arrays are `(..., y, x)`.

`pde.graph(spec)` turns the right-hand sides into an expression DAG (shared subexpressions merged). This graph is
what the model's encoder sees.

## Files

- `pde.py`: spec -> graph / LaTeX, and a generic exponax ETDRK solver for any spec
- `template.py`: 2- and 3-field templates. An equation is a coefficient matrix over a fixed term library, so one
  compile covers every equation (vmapped over equations x ICs)
- `sample.py`: equation families (transport near shallow water, excitable near Aliev-Panfilov, pattern formation,
  random), IC pools, and window selection (greedy log-det over simple features)
- `generate.py`: dt calibration, rollouts, window selection, writing. Resumable; `--id-offset` lets several GPUs
  share one output dir
- `coverage.py`: how often each library term / SWE piece / AP piece shows up
- `merge_extra.py`: merges a `--supplement` run into a dataset
- `demos.py`: shallow water and Aliev-Panfilov written as specs (used by `model/targets.py` and `baseline_FNO`)

Self-checks: `python pde.py`, `python template.py`, `python sample.py`.

## Output layout

`index.jsonl` (one line per equation) and `eq_XXXX/` with:
- `spec.json`: spec + latex, graph, template coefficients, normalization, IC family per sample
- `u.npy`: float16 `(ICs, 15, fields, 128, 128)`, normalized
- `static.npy`: float16 `(ICs, 128, 128)` if the spec has statics

## Datasets

The datasets (~200 GB) aren't stored anywhere, these commands regenerate them.

A: 1024 equations x 73 ICs, old sampler (before `ec1a5f0`).

```sh
python generate.py --out /workspace/pdegen/out --pool 512
```

351 of A's equations were hit by the selection bug below and got reselected:

```sh
python generate.py --supplement A_SPECS_AFFECTED --extra 0 --pool 512 --windows 2 --ic-scale 0.9 0.3 --out A_FIX --n 1024
python merge_extra.py A_DIR A_FIX 73 73
```

B: 4096 equations x 30 ICs, current sampler. One worker per GPU (second GPU: `--id-offset 2048 --seed 2`):

```sh
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.9 python generate.py --out /workspace/pdegen/out \
    --n 2048 --keep 18 --pool 128 --windows 3 --target 0.06 0.24 --permute 0.5 --id-offset 0 --seed 1
python generate.py --supplement /workspace/pdegen/out --extra 12 --out /workspace/pdegen/extra --pool 128 --windows 3 \
    --n 2048 --id-offset 0
python merge_extra.py B_DIR EXTRA_DIR 18 30
```

What's different in B:
- transport equations use a random subset of the SWE pieces (flux, adv, grad, bathy) and excitable ones a subset of
  the AP pieces (cubic, uv, gate). The full SWE and AP cores are never generated, so they stay held out
- 60% of equations drop each non-core term with p = 0.1
- field order is shuffled with p = 0.5 (`perm` in the spec)
- per-frame change target is log-uniform in [0.06, 0.24]

Bug note: in the original 18-window selection, one blown-up candidate (inf/nan) broke the feature normalization, so
1280 equations only picked ICs 0 and 1. The supplement pass reselects all 30 windows for those.

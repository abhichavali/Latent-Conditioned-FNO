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
| `sample.py` | equation families (transport/waves near SWE, excitable/reaction–diffusion near Aliev–Panfilov, pattern formation, free composition) + mutations; GPU IC pools (7 pattern families, patches, wakes, vortex seeds); max-entropy (greedy log-det over bounded spectral/amplitude/rate features) window selection, capped at 25% per IC family and 2/3 early windows |
| `generate.py` | per-equation dt / frame-interval calibration, pool rollout over two 14-step windows, selection of 73 windows, writer |
| `coverage.py` | building-block coverage: equations and partner terms per library term, SWE / AP pieces alone and in pairs, terms that always co-occur; for the sampler (`--sample 4096`) or a dataset (`--out DIR`, writes `coverage.json`) |
| `demos.py` | shallow water and Aliev–Panfilov as specs, simulated and written for `viz/tsunami` and `viz/heart` |

## Dataset layout (`/workspace/pdegen/out` on the GPU box)

`index.jsonl` (one line per equation) and `eq_XXXX/`:
- `spec.json` — spec + `latex`, `graph`, `template` coefficients, `norm`, per-sample IC family and window start
- `u.npy` — float16 `(73, 15, fields, 128, 128)` = `(u - norm.shift) / norm.scale`
- `static.npy` — float16 `(73, 128, 128)`, raw, when the equation has a static field

```sh
python generate.py --out /workspace/pdegen/out --pool 512                          # 1024 eqs x 73 ICs, resumable
python generate.py --out /workspace/pdegen/out --n 4096 --keep 18 --pool 128 --seed 1   # 4096 eqs x 18 ICs, disjoint
python ../viz/dataset/serve.py --data /workspace/pdegen/out       # viewer, then: ssh -N -L 8000:localhost:8000 pdebox
```

Self-checks: `python pde.py`, `python template.py`, `python sample.py`.

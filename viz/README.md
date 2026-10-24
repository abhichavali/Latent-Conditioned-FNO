# Visualization engines

Two solver-agnostic three.js viewers. They only consume gridded fields per timestep, so the
numerical solver and the FNO plug in by writing the same `.npy` files.

| Engine | Page | Field |
|---|---|---|
| Shallow water / tsunami | `tsunami/index.html` | water depth `h(t, y, x)` over bed `b(y, x)` |
| Aliev–Panfilov cardiac | `heart/index.html` | transmembrane potential `u(t, y, x)` |

## Run

```sh
python3 -m http.server 8000          # from the repo root
open http://localhost:8000/viz/tsunami/
open http://localhost:8000/viz/heart/
```

With no data, each page streams a synthetic demo. three.js comes from jsdelivr, so the first load needs network.

## Feeding data

Point a page at a directory: `?data=../../data_collection/run_042/` (relative to the page), or drag
the files onto the window. Arrays are C-order float32/float64; `(T, 1, Ny, Nx)` is squeezed.

**Tsunami** — `h.npy` `(T, Ny, Nx)` water depth in m (≥ 0, dry = 0) and `b.npy` `(Ny, Nx)` bed
elevation in m with sea level at 0 (negative = under water). Without `b.npy` a flat bed at the mean
initial depth is assumed. If your solver outputs surface elevation η, save `h = np.maximum(eta - b, 0)`.

**Heart** — `u.npy` `(T, Ny, Nx)`, any units. In `ventricle` geometry x is periodic (cell `i` sits
at angle `2πi/Nx`, `y` runs apex → base); use `sheet` for non-periodic 2D tissue.

`meta.json` (optional, all keys optional):

| Key | Used by | Meaning |
|---|---|---|
| `dt`, `time_unit` | both | time per frame and its label |
| `title` | both | page heading |
| `lx`, `ly` | both | domain size (tsunami: metres; heart: sheet aspect) |
| `vexag`, `wexag`, `dry` | tsunami | vertical / wave exaggeration, dry threshold in m |
| `geometry`, `range` | heart | `ventricle` or `sheet`; colour range `[lo, hi]` (default: data min/max) |

```python
np.save(f"{out}/h.npy", h.astype(np.float32))   # (T, Ny, Nx)
np.save(f"{out}/b.npy", b.astype(np.float32))   # (Ny, Nx)
json.dump({"dt": dt, "lx": Lx, "ly": Ly}, open(f"{out}/meta.json", "w"))
```

## Live / programmatic

Each page exposes `window.viz = { engine, player, start, push }`. A solver running in the browser
(or a WebSocket client) calls `viz.push(frame)` with one `Float32Array` of `Ny*Nx` per step; the
timeline grows as frames arrive. Engines themselves only take `setFrames(frameA, frameB, alpha)`
and interpolate between the two on the GPU.

Any view setting can be deep-linked, e.g. `?mode=eta&sun=0.6&t=120` (`t` = frame index, pauses).
Keys: space play/pause, ←/→ step.

## Files

- `common.js` — `.npy` parser, loaders, colormap LUTs, GPU grid sampling, timeline
- `tsunami/engine.js`, `heart/engine.js` — renderers (no solver code)
- `tsunami/demo.js`, `heart/demo.js` — synthetic stand-in data

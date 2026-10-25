# Simula: the HackPSU deck

**Simula: Speeding Up Simulation with AI.** A 4-minute pitch whose demo is half the talk. The script is
`README.org` (the user's script, the source of truth for the slides). The deck is React + GSAP, and the
3D views are the repo's own three.js engines (`../viz/`), bundled in. Once built, it runs fully offline: three.js is
vendored through npm and the fonts are self-hosted. The archived research deck is still here (see the end).

## Run

```sh
npm install
npm run mock-bank  # rehearsal data for the demo (needs ../baseline_FNO/results/viz); skip once the real bank is in
npm run dev        # http://localhost:5173, hot reload
npm run build      # type-check (tsc) + production build into dist/
npm run preview    # serve dist/ at http://localhost:4173
npm run check      # self-check for the navigation maths
```

Requires Node 20.19+ (built with Node 26.5, npm 11.17). **For the talk:** `npm run build`, then `npm run preview`
(or any static server over `dist/`) and press `f` for fullscreen. No network needed.

## The slides (`src/slides/`)

| # | Slide | Builds (clicks) |
|---|---|---|
| 1 | Hook + title: solver vs Simula side by side, then "Simula" | 0 (the title comes in on its own after ~3 s) |
| 2 | The problem: tsunami + heart, punchline | 2 |
| 3 | Why AI hasn't fixed it: the piles of simulations, punchline | 2 |
| 4 | Simula: learned physics → reads your equation → picks smart experiments | 3 |
| 5 | Live demo: Tsunami · Flood-risk map · Heart (one tab per build step) | 2 |
| 6 | Results: two big numbers + error-vs-simulations chart | 1 |
| 7 | Where else this goes | 1 |
| 8 | What's next + team + closing line | 1 |

Every slide has speaker notes with the script's timing and lines (`p` opens them).

## Keys

| Key | Action |
|---|---|
| → ↓ Space PageDown, click, swipe left | next build step, then next slide |
| ← ↑ Shift+Space PageUp, click in the left quarter, swipe right | previous |
| Home / End | first / last slide |
| `o` | overview grid (click a slide to jump; `o`, Esc or Enter closes it) |
| `f` | fullscreen |
| `p` | presenter window: notes, current and next build, timer. The two windows stay in sync. |

The URL hash records your position (`#/5/2` is slide 5, build step 2). Under `prefers-reduced-motion` every build,
camera move and ambient loop is instant or off. Clicks inside the demo area never change slides; the arrow keys
still do.

## The demo (slide 5)

The tabs follow the build steps (→ / ←) and can also be clicked.

- **Tsunami.** Press and drag anywhere on the sea: the epicentre marker follows and the forecast switches to the
  nearest precomputed earthquake, so it updates instantly. Faint rings mark the precomputed epicentres. On release,
  the marker snaps to the scenario it's showing. The card on the right is the race: the numerical solver's clock runs
  in real time toward its measured duration, and the forecast appears after Simula's measured time. Every new
  epicentre restarts both.
- **Flood-risk map.** The camera rises over the coast. Press **Run N earthquakes**: the counter climbs, and the heat
  map builds up over the coast through the cumulative snapshots, taking the measured batch time. When it's done it
  shows "forecast Z · solver W".
- **Heart.** Click the tissue to place a scar (snaps to the nearest precomputed scar; rings mark them) and the
  arrhythmia forms. The candidate ablation lines (A, B, C) appear on the tissue: click a line, or its button on the
  right. Partway through the rollout, the verdict shows ("Arrhythmia stopped" / "Still spiralling"), the line turns
  blue or orange, and the button records it. Clicking the tissue away from the lines moves the scar.
- **Backup: play video** (top right) plays `public/demo/backup.mp4` (`dist/demo/backup.mp4` in a build). If the
  file isn't there, it says so. Esc closes it. Record it with any screen recorder (H.264 .mp4).
- A **"Mock data"** badge (and "Mock forecast" in place of "Simula") means the rehearsal bank is loaded, not
  Simula's predictions.
- If the 3D views stutter on the venue machine, add `?dpr=0.75` to the URL. On a strong GPU, `?dpr=2` makes them
  crisper. The default is 1, which keeps two live views (the hook) at 60 fps on a retina laptop.

## Scenario bank (what the GPU job writes)

The demo plays precomputed predictions; interactions snap to the nearest scenario. The bank is fetched at runtime
from `public/demo/` (copied to `dist/demo/` by the build), so you can also drop a new bank straight into
`dist/demo/` without rebuilding. It is git-ignored (too big). All paths in an `index.json` are relative to its folder.

**Arrays:** `.npy`, C order, float32 (`<f4`) or float16 (`<f2`). A frame stack is `(T, ny, nx)` (a `(T, 1, ny, nx)`
array also works). Row `j` is y, column `i` is x. **Positions** are grid fractions:
`x = i / (nx - 1)`, `y = j / (ny - 1)`, with `(0, 0)` at the first row and column. Playback interpolates linearly
between stored frames, so store enough frames that a wave front moves only a few cells per frame.

### `public/demo/tsunami/index.json`

```jsonc
{
  "version": 1,
  "mock": false,                       // true shows the "Mock data" badge
  "note": "free text, not shown",
  "grid": { "nx": 128, "ny": 128, "lx": 4000, "ly": 4000 },  // cells, and domain size in metres
  "bed": "b.npy",                      // (ny, nx) bed elevation in m, sea level 0, land > 0
  "dt": 4, "time_unit": "s",           // time between stored frames
  "view": { "vexag": 4, "wexag": 6 },  // optional: vertical and wave exaggeration for the engine
  "default": "e0",                     // scenario shown first, and the one the hook (slide 1) plays
  "scenarios": [
    {
      "id": "e0", "x": 0.134, "y": 0.693, // epicentre (grid fractions)
      "frames": "e0/h.npy",              // (T, ny, nx) water depth h in m (h = max(eta - b, 0)): Simula's forecast
      "solver": "e0/h_solver.npy"        // optional: the solver's frames for the same quake (hook's left panel)
    }
  ],
  "risk": {                            // optional: the flood-risk map tab
    "frames": "risk_map.npy",          // (K, ny, nx): max inundation depth on land (m) after the first k/K of the
                                       // batch, cumulative; K = 1 (just the final map) works too. 0 offshore.
    "count": 1000,                     // earthquakes in the batch (the counter)
    "unit": "m"
  }
}
```

Use float32 for `h`: float16 rounds deep-water depths to ~3 cm, which shows as ripples on the surface. Any extra
scenario keys (magnitude, depth, …) are allowed and ignored. Snapping uses the distance in `(x, y)`.

### `public/demo/heart/index.json`

```jsonc
{
  "version": 1,
  "mock": false,
  "grid": { "nx": 128, "ny": 128 },    // optional "lx", "ly": sheet aspect
  "geometry": "sheet",                 // "sheet" (flat tissue) or "ventricle" (x periodic, wrapped on a heart shape)
  "dt": 12.9, "time_unit": "ms",
  "range": [0, 1],                     // colour range of u (default [0, 1])
  "scars": [
    {
      "id": "s0", "x": 0.55, "y": 0.32, "r": 0.06,   // scar centre and radius (grid fractions)
      "frames": "s0/u.npy",                         // (T, ny, nx) potential u, no ablation: the arrhythmia forming
      "ablations": [
        {
          "id": "a", "label": "Line A",
          "line": [[0.61, 0.32], [0.95, 0.32]],      // ablation line endpoints (grid fractions)
          "frames": "s0/a.npy",                      // (T, ny, nx) u after ablating this line, from the arrhythmia
          "stops_arrhythmia": true                   // the verdict the UI reveals
        }
      ]
    }
  ]
}
```

Ablation rollouts start from the established arrhythmia (for example the scar rollout's last state) with the line
ablated at frame 0. The verdict shows 45 % of the way through. float16 is fine for `u`.

### The mock bank (rehearsal only)

`npm run mock-bank` (`scripts/mock-bank.mjs`) writes both banks from `../baseline_FNO/results/viz/` (solver truth
vs a from-scratch baseline FNO). **None of it is Simula.**

- Tsunami: 4 epicentres along the coast, made by shifting the baseline-FNO rollout in y (solver = the shifted
  truth), plus a 4-quake risk map.
- Heart: 2 scars (the rollout shifted) × 3 ablation lines. The lines and the "stopping" fade are drawn in, not
  simulated.

Both `index.json` files say `"mock": true`.

## `src/data/results.json`: every headline number

Nothing in the deck is typed in by hand. While `placeholder` is `true`, slide 6 shows "Results pending" and every
null shows as a dash. Fill in the file, set `"placeholder": false`, then rebuild (`npm run build`; `npm run dev`
picks it up live).

| Field | Used by |
|---|---|
| `fewer_simulations` | slide 6: "X× fewer simulations" |
| `timing.batch.{simula_seconds, solver_seconds, scenarios}` | slide 6: Y = solver / simula, "(N scenarios: Z vs W)"; the risk-map animation length and its caption |
| `timing.tsunami.{simula_seconds, solver_seconds}` | slide 1 (the race) and the demo's tsunami race card |
| `timing.heart.{simula_seconds, solver_seconds}` | the demo's heart race card |
| `chart.series[]` = `{ id, label, points: [{ sims, error }] }` | slide 6 chart. `id` `simula` is drawn in blue; others grey. |
| `rehearsal.*` | **not results**: durations the timers use while a measured timing is null. Every view using them says "placeholder timing". |

The "standard AI" curve is pre-filled with the measured from-scratch FNO (shallow water, 19-step rollout-mean nRMSE,
`baseline_FNO/RESULTS.md`). Simula's curve is empty and the chart says "results pending". **Plot both curves with
the same metric**: if Simula's numbers come from `model/active.py` (relL2), replace the standard-AI points with
`scratch_curve` from the same run.

Durations show in s below 2 min, then min, then h. The team slot on slide 8 reads `src/data/team.json`: names, plus
an optional photo path under `public/` (e.g. `"team/abhinav.jpg"` for `public/team/abhinav.jpg`). Without a photo,
it shows initials.

## Code map

| Path | What |
|---|---|
| `src/slides/` | the Simula deck, one file per slide, ordered by file name |
| `src/demo/bank.ts` | bank + results loading, schema types, timing helpers |
| `src/demo/engines.ts` | mounts the viz engines (orbit off, pixel ratio, teardown), frame playback, picking, risk heat map |
| `src/demo/Hook.tsx`, `Demo.tsx`, `HeartDemo.tsx`, `hud.tsx` | slide 1's race, slide 5's tabs/tsunami/risk, the heart tab, shared overlays |
| `src/viz.d.ts` | types for the `@viz/*` engine modules |
| `vite.config.ts` | `@viz` → `../viz`; `three` → the vendored copy (the engines import it by bare name) |
| `src/deck/`, `src/components/` | the deck engine and components (below) |

In thumbnails (overview, presenter window, `?print`), the 3D views become static posters, so the presenter window
creates no WebGL contexts.

## Deck engine

**Add a slide:** every file in `src/slides/` is a slide. Copy one, rename it, and edit:

```tsx
import type { SlideDef } from '../deck/SlideFrame'
import { Frame, Points, build } from '../components/layouts'

export default {
  title: 'Ablation',            // overview, presenter, footer
  tone: 'latent',               // accent colour: 'sea' | 'ember' | 'latent'
  section: 'Results',           // footer label
  steps: 2,                     // number of build steps (clicks before moving to the next slide)
  Component: () => (
    <Frame title="Ablation" lede="What the prior buys.">
      <Points items={['appears on click 1', 'appears on click 2']} />
      <p {...build(2, 'fade', 0.3)}>Also on click 2, 0.3 s later.</p>
    </Frame>
  ),
  notes: `Speaker notes. Newlines are kept.`,
} satisfies SlideDef
```

**Builds:** give any element `data-step={k}` (or spread `build(k, anim, delay)`) and it animates in at click `k`.
Step 0 is the slide's entrance. `anim` is one of `fade` (default), `appear`, `pop`, `draw` (SVG strokes, using
DrawSVG), `wipe`, `words` (SplitText), `count`, `dim` or `out`. Each slide has one paused GSAP timeline with a label
per step. For bespoke tweens, call `useBuild((k, tl, at) => …)` inside a component or set `build` on the slide.
Selectors inside these are scoped to the slide. `useStep()` gives the current build step to React code (the demo's
tabs use it), and `useAmbient(fn)` runs endless decorative tweens on the live stage only.

**Components** (`src/components/`): `Title`, `Section`, `Frame`, `TwoCol`, `Quote`, `Points` (layouts); `Stat` (big
number that counts up; `value={null}` shows a dash); `Img`, `Video`; `Code`; `Tex` (KaTeX); `Diagram`; `Chart`
(SVG line chart, log axes, legend, pending band); `LiveEmbed` (iframe of a viz page, used by the research deck).
Design tokens are CSS variables at the top of `src/theme.css` and match `viz/common.css`. The type is Newsreader,
IBM Plex Sans and IBM Plex Mono, self-hosted.

## Export to PDF

Open the built deck with `?print` (e.g. <http://localhost:4173/?print>) and print to PDF (Chrome: landscape,
margins none, background graphics on). There's one 1920 × 1080 page per slide, each at its last build step, with
posters in place of the 3D views. For the demo itself, use the backup video.

## The archived research deck

The earlier research-style deck (13 slides: architecture, data, baseline curves) lives in `src/slides_research/`.
Open it with `?deck=research`, e.g. <http://localhost:5173/?deck=research>. Its live-demo slide iframes the viz
pages, which need `python3 -m http.server 8000` at the repo root (`?viz=<url>` or `VITE_VIZ_BASE` to point
elsewhere).

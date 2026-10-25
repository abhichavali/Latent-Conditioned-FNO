# PDE Foundation Model: the deck

A React + GSAP slide deck for the HackPSU talk. It uses the same dark, glassy look as `viz/`, and it can embed the
viz pages live.

## Run

```sh
npm install
npm run dev        # http://localhost:5173, hot reload
npm run build      # type-check (tsc) + production build into dist/
npm run check      # self-check for the navigation maths
```

Requires Node 20.19+ (built with Node 26.5, npm 11.17).

## Live demos (viz embeds)

The demo slide puts `viz/tsunami/` and `viz/heart/` in iframes, playing the baseline rollouts in
`baseline_FNO/results/viz/`. Those pages need a static server at the repo root, and three.js loads from jsdelivr,
so the first load needs a network connection.

- **For the talk (recommended):** `npm run build`, then from the repo root run `python3 -m http.server 8000`
  and open <http://localhost:8000/presentation/dist/>. The deck finds the viz pages at `../../viz/`.
- **While editing:** keep `npm run dev` running and also run `python3 -m http.server 8000` at the repo root. In dev
  mode the deck looks for the viz pages at `http://localhost:8000/viz/`.
- To point the embeds somewhere else, add `?viz=<url>` to the deck URL or set `VITE_VIZ_BASE` at build time.

If a page can't be reached, the panel says so and shows these commands instead of a blank frame. Click
**Interact** to give the mouse and keyboard to the viz, and **Done** to hand them back to the deck.

## Keys

| Key | Action |
|---|---|
| → ↓ Space PageDown, click, swipe left | next build step, then next slide |
| ← ↑ Shift+Space PageUp, click in the left quarter, swipe right | previous |
| Home / End | first / last slide |
| `o` | overview grid (click a slide to jump; `o`, Esc or Enter closes it) |
| `f` | fullscreen |
| `p` | presenter window: notes, current and next build, timer. The two windows stay in sync. |

The URL hash records your position (`#/5/2` is slide 5, build step 2), so a refresh keeps your place. Under
`prefers-reduced-motion` every build and transition is instant.

## Add a slide

Every file in `src/slides/` is a slide, ordered by file name. Copy one, rename it (for example `06b-ablation.tsx`),
and edit:

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
per step. Clicking tweens the playhead to the next label, and going back plays it in reverse. For bespoke tweens,
call `useBuild((k, tl, at) => …)` inside a component, or set `build` on the slide (see `Code` focus and the
posterior morph in `07-active.tsx`). If a slide uses a `data-step` higher than its `steps`, dev mode warns in the
console.

## Components (`src/components/`)

| Component | Use |
|---|---|
| `Title`, `Section`, `Frame`, `TwoCol`, `Quote`, `Points` | layouts (`layouts.tsx`) |
| `Stat` | big number that counts up; `value={null}` shows a pending dash |
| `Img`, `Video` | media; `framed` puts plots on a white mat |
| `Code` | highlighted code; `focus={{ 1: [5] }}` lights up line 5 at step 1 |
| `Tex` | KaTeX math, inline or `block` |
| `Diagram` | nodes (HTML, so they can hold KaTeX) and edges that draw in step by step; `bend` curves an edge |
| `Chart` | SVG line chart from a JSON array: log or linear axes, error bars, legend, and a hatched `band` for pending results |
| `LiveEmbed` | iframe of a viz page with variant toggles (for example FNO or solver), plus a placeholder when it isn't served |

Design tokens (colours, type scale, easing) are CSS variables at the top of `src/theme.css`. The base colours match
`viz/common.css`. The type is Newsreader for display and IBM Plex Sans and Mono for text and code, all self-hosted
through Fontsource, so the deck works offline. Motion tokens are in `src/deck/motion.ts`.

## Export to PDF

Open the built deck with `?print`, for example <http://localhost:8000/presentation/dist/?print>, and print to PDF
(Chrome: landscape, margins none, background graphics on). Every slide is laid out at its last build step, one
1920 × 1080 page each. Headless alternative:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --force-prefers-reduced-motion \
  --no-pdf-header-footer --virtual-time-budget=8000 --print-to-pdf=deck.pdf \
  'http://localhost:8000/presentation/dist/?print'
```

In the PDF, the live embeds become static cards.

## What's real and what's a placeholder

- Real: dataset numbers (`data_collection/README.md`), baseline curves (`src/data/baseline.json`, transcribed from
  `baseline_FNO/RESULTS.md` and checked against it), and the appendix figure, which is bundled straight from
  `baseline_FNO/results/`.
- **Placeholder:** slide 10, the headline result. It shows only the measured from-scratch curves, plus a hatched
  "pending" region and two `Stat value={null}` slots. Add the few-shot series there when the results exist.
- The demo slide plays baseline FNO rollouts. Point it at the few-shot model's output once that exists.

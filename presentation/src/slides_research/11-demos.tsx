import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'
import { LiveEmbed } from '../components/LiveEmbed'

// Real rollouts from baseline_FNO/results/viz (paths relative to each viz page). Swap in the few-shot model's output when it lands.
const R = '../../baseline_FNO/results/viz'

export default {
  title: 'Live: tsunami and cardiac spiral waves',
  tone: 'ember',
  section: 'Demo',
  steps: 1,
  Component: () => (
    <Frame title="Live: a coastal tsunami and a scarred heart"
      lede="Held-out initial conditions, rolled out by the FNO and rendered live in three.js. Toggle to the solver to compare.">
      <div className="row" style={{ gap: 40, marginTop: -12 }}>
        <div style={{ flex: 1 }}>
          <LiveEmbed title="Shallow water" step={0} height={560} variants={[
            { label: 'FNO', src: `tsunami/?data=${R}/swe_fno/` },
            { label: 'Solver', src: `tsunami/?data=${R}/swe_truth/` },
          ]} />
        </div>
        <div style={{ flex: 1 }}>
          <LiveEmbed title="Aliev–Panfilov" step={1} height={560} variants={[
            { label: 'FNO', src: `heart/?data=${R}/ap_fno/&geometry=sheet` },
            { label: 'Solver', src: `heart/?data=${R}/ap_truth/&geometry=sheet` },
          ]} />
        </div>
      </div>
    </Frame>
  ),
  notes: `Both views are the repo's viz pages, live in iframes, playing real rollouts on held-out initial conditions.
Today these are the from-scratch baseline FNO (largest model); the few-shot model plugs into the same format.
Left: a tsunami from a fault offshore running up a sloping coast. Press "Solver" to compare against ground truth.
(click) Right: Aliev–Panfilov excitation waves on a 2D sheet.
"Interact" hands the mouse to the viz (orbit, scrub); press "Done" to give the deck its keys back.
If the panels say the view isn't reachable: run python3 -m http.server 8000 at the repo root.`,
} satisfies SlideDef

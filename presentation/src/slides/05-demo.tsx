import type { SlideDef } from '../deck/SlideFrame'
import { Demo } from '../demo/Demo'

export default {
  title: 'Live demo',
  section: 'Live demo',
  steps: 3, // 0 tsunami, 1 flood-risk map, 2 heart, 3 wildfire (tabs are clickable too)
  Component: Demo,
  notes: `(1:40–3:10) The demo is half the talk. → / ← switch tabs; clicking the tabs works too.

Frame everything in thousands: a forecaster doesn't run one scenario, they run thousands.

TSUNAMI (drag in the map)
- Drag the epicentre. Simula's forecast lands while the solver card is still computing.
- "A forecaster has to consider thousands of possible earthquakes. Each one, Simula forecasts ten times faster than the simulator."
- Move it once or twice more. Every move restarts the race.

FLOOD-RISK MAP (→)
- Click "Run 1,000 earthquakes". The map fills in over the coast with the counter.
- "Here are a thousand earthquakes at once: every stretch of coast, flood risk for each, in about five seconds."
- (Wave heights in the 3D view are exaggerated for visibility; the hint line says so.)

HEART (→)
- Click the tissue to place a scar: "Place a scar, and a spiral (the arrhythmia) forms around it."
- "A cardiologist would want to screen thousands of scar layouts and treatment plans. At ten times the speed of the simulator, that becomes practical."
- Try two or three scar positions. (There's no ablation step: in our simulator an ablation line slows the wave but never blocks it.)

WILDFIRE (→)
- Click the forest to start a fire: "A fire starts here, and the wind drives it toward the town."
- Click the firebreaks one by one: "Cut a firebreak here... the town burns. Here... the town is saved."
- "A fire crew would want to test thousands of firebreak plans before committing bulldozers. Simula makes that a few seconds."

If anything breaks: "Backup: play video" (top right) plays public/demo/backup.mp4. Esc closes it.
A "Mock data" badge means you're on the rehearsal bank, not Simula's predictions.`,
} satisfies SlideDef

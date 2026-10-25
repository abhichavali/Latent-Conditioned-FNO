import type { SlideDef } from '../deck/SlideFrame'
import { Demo } from '../demo/Demo'

export default {
  title: 'Live demo',
  section: 'Live demo',
  steps: 2, // 0 tsunami, 1 flood-risk map, 2 heart (tabs are clickable too)
  Component: Demo,
  notes: `(1:40–3:10) The demo is half the talk. → / ← switch tabs; clicking the tabs works too.

TSUNAMI (drag in the map)
- Drag the epicentre. The forecast updates instantly while the solver card on the right is still counting.
- "An earthquake here... instant forecast. The solver is still running."
- Move it once or twice more. Every move restarts the solver's clock.

FLOOD-RISK MAP (→)
- Click "Run 1,000 earthquakes". The map fills in over the coast with the counter.
- "1,000 earthquakes, every stretch of coast, in about five seconds."

HEART (→)
- Click the tissue to place a scar: "Place a scar, and a spiral (the arrhythmia) forms, instantly."
- Click the ablation lines (or the buttons on the right) one by one: "Try this line... still spiralling. This one... stopped."

If anything breaks: "Backup: play video" (top right) plays public/demo/backup.mp4. Esc closes it.
A "Mock data" badge means you're on the rehearsal bank, not Simula's predictions.`,
} satisfies SlideDef

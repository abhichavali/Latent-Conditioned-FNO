import type { SlideDef } from '../deck/SlideFrame'
import { demo } from '../demo/Demo'

export default {
  title: 'Tsunami demo',
  section: 'Applications',
  tone: 'sea',
  steps: 2,
  Component: demo([0, 1, 2]),
  notes: `SIDE BY SIDE: click "▶ Run both". "Same quake, same physics. The simulator is still computing; Simula is already done, 12 times faster."

Frame it in thousands: a forecaster doesn't run one scenario, they run thousands.

TSUNAMI (drag in the map)
- Drag the epicentre. Simula's forecast lands while the solver card is still computing.
- "A forecaster has to consider thousands of possible earthquakes. Each one, Simula forecasts ten times faster than the simulator."
- Move it once or twice more. Every move restarts the race.

FLOOD-RISK MAP (→ or click the tab)
- Click "Run 1,000 earthquakes". The map fills in over the coast with the counter.
- "Here are a thousand earthquakes at once: every stretch of coast, flood risk for each, in about five seconds."
- (Wave heights in the 3D view are exaggerated for visibility; the hint line says so.)

If anything breaks: "Backup: play video" (top right) plays public/demo/backup.mp4. Esc closes it.
A "Mock data" badge means you're on the rehearsal bank, not Simula's predictions.`,
} satisfies SlideDef

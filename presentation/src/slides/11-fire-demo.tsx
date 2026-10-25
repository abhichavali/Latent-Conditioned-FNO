import type { SlideDef } from '../deck/SlideFrame'
import { demo } from '../demo/Demo'

export default {
  title: 'Wildfire demo',
  section: 'Applications',
  tone: 'ember',
  steps: 1,
  Component: demo([5, 6]),
  notes: `SIDE BY SIDE: click "▶ Run both". "The same fire, simulated two ways. Simula is about 4 times faster here."

Frame it in thousands: a forecaster doesn't run one scenario, they run thousands.

WILDFIRE
- Click the forest to start a fire: "A fire starts here, and the wind drives it toward the town."
- Click a couple of other spots: "Every ignition point, forecast in a blink. A fire service would want thousands of these, for every wind and every dry season."
- (No firebreak step: Simula doesn't yet predict when a firebreak saves the town; the simulator does in about a third of cases.)

If anything breaks: "Backup: play video" (top right) plays public/demo/backup.mp4. Esc closes it.
A "Mock data" badge means you're on the rehearsal bank, not Simula's predictions.`,
} satisfies SlideDef

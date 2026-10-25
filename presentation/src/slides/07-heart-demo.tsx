import type { SlideDef } from '../deck/SlideFrame'
import { demo } from '../demo/Demo'

export default {
  title: 'Heart demo',
  section: 'Applications',
  tone: 'ember',
  steps: 1,
  Component: demo([3, 4]),
  notes: `SIDE BY SIDE: click "▶ Run both". "The same heart, simulated two ways. Simula finishes about 12 times faster."

Frame it in thousands: a forecaster doesn't run one scenario, they run thousands.

HEART
- Click the tissue to place a scar: "Place a scar, and a spiral (the arrhythmia) forms around it."
- "A cardiologist would want to screen thousands of scar layouts and treatment plans. At ten times the speed of the simulator, that becomes practical."
- Try two or three scar positions. (There's no ablation step: in our simulator an ablation line slows the wave but never blocks it.)

If anything breaks: "Backup: play video" (top right) plays public/demo/backup.mp4. Esc closes it.
A "Mock data" badge means you're on the rehearsal bank, not Simula's predictions.`,
} satisfies SlideDef

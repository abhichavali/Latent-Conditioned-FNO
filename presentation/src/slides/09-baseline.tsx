import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'

export default {
  title: 'A from-scratch FNO needs hundreds of simulations',
  tone: 'sea',
  section: 'Results',
  Component: () => <Frame title="A from-scratch FNO needs hundreds of simulations" lede="Baseline error vs. number of training simulations, held-out test ICs." />,
  notes: `Baseline numbers from baseline_FNO/RESULTS.md.`,
} satisfies SlideDef

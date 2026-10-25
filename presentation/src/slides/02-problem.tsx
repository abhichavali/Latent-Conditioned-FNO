import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'

export default {
  title: 'Every new physics problem starts from zero',
  tone: 'sea',
  section: 'Motivation',
  Component: () => <Frame title="Every new physics problem starts from zero" lede="Solvers are slow, and a neural surrogate is retrained for every new equation." />,
  notes: `Solvers are accurate but slow. Neural operators make each run nearly free, but only after hundreds of training simulations, and new physics means starting over.`,
} satisfies SlideDef

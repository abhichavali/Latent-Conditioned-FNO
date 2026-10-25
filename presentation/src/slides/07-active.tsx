import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'

export default {
  title: 'Spend simulations where the model is unsure',
  tone: 'latent',
  section: 'Method',
  Component: () => <Frame title="Spend simulations where the model is unsure" lede="Sample, score disagreement, cluster, simulate, update." />,
  notes: `Active few-shot loop: Bayesian optimal experimental design on a fixed budget.`,
} satisfies SlideDef

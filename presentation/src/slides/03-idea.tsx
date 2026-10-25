import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'

export default {
  title: "Don't retrain the operator. Infer the physics.",
  tone: 'latent',
  section: 'Idea',
  Component: () => <Frame title="Don’t retrain the operator. Infer the physics." lede="A pretrained operator, an equation-graph prior, and active few-shot learning." />,
  notes: `Three pieces: a pretrained FNO, a prior over its latent code read from the equation itself, and a small, well-spent simulation budget.`,
} satisfies SlideDef

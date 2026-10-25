import type { SlideDef } from '../deck/SlideFrame'
import { Section } from '../components/layouts'

export default {
  title: 'Results',
  tone: 'sea',
  section: 'Results',
  Component: () => (
    <Section mark="Results" title="How many simulations does new physics cost?"
      lede="First the from-scratch baseline, then where the few-shot model has to land." />
  ),
  notes: `Now the numbers. First, what you pay today with a from-scratch FNO; then the bar for our model.`,
} satisfies SlideDef

import type { SlideDef } from '../deck/SlideFrame'
import { Title } from '../components/layouts'

export default {
  title: 'PDE Foundation Model',
  chrome: false,
  Component: () => (
    <Title
      title="PDE Foundation Model"
      subtitle="A pretrained neural operator that adapts to new physics from a handful of simulations."
      meta={['Abhinav Chavali', 'HackPSU 2026']}
    />
  ),
  notes: `Hi, we built a foundation model for PDEs.
One pretrained Fourier Neural Operator, steered by a small latent code z, that adapts to a new equation from a handful of simulations instead of hundreds.
We'll show it on two held-out problems: a coastal tsunami and spiral waves in a scarred heart.`,
} satisfies SlideDef

import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'

export default {
  title: 'One frozen operator, steered by a latent code',
  tone: 'latent',
  section: 'Method',
  Component: () => <Frame title="One frozen operator, steered by a latent code" lede="Graph encoder to prior over z to a conditioned FNO." />,
  notes: `The FNO is frozen after pretraining; the equation only enters through z.`,
} satisfies SlideDef

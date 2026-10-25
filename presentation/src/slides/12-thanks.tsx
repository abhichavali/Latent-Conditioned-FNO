import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'

export default {
  title: 'Thank you',
  tone: 'sea',
  section: 'Takeaways',
  Component: () => <Frame title="Thank you" lede="One operator, many physics, a handful of simulations." />,
  notes: `Takeaways and thanks.`,
} satisfies SlideDef

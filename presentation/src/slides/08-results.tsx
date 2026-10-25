import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'

export default {
  title: 'Results',
  tone: 'sea',
  section: 'Results',
  Component: () => <Frame title="Results" lede="What a from-scratch FNO needs, and what we're aiming to beat." />,
  notes: `Section divider.`,
} satisfies SlideDef

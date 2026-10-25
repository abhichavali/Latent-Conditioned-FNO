import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'

export default {
  title: 'An equation is a graph',
  tone: 'latent',
  section: 'Method',
  Component: () => <Frame title="An equation is a graph" lede="The encoder reads the symbolic expression DAG, not the data." />,
  notes: `Every right-hand side becomes an expression DAG with shared subexpressions.`,
} satisfies SlideDef

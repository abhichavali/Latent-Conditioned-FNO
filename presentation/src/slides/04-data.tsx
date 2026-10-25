import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'

export default {
  title: '5,120 synthetic PDEs, built from physical building blocks',
  tone: 'sea',
  section: 'Data',
  Component: () => <Frame title="5,120 synthetic PDEs, built from physical building blocks" lede="Two datasets of equations sampled from a term library near shallow water and Aliev–Panfilov." />,
  notes: `Dataset A: 1,024 equations x 73 ICs. Dataset B: 4,096 equations x 30 ICs.`,
} satisfies SlideDef

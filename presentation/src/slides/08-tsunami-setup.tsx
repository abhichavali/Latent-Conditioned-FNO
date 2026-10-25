import type { SlideDef } from '../deck/SlideFrame'
import { Setup } from '../components/Setup'

export default {
  title: 'Tsunami forecasting',
  section: 'Applications',
  tone: 'sea',
  steps: 1,
  Component: () => <Setup n={2} title="Tsunami forecasting" tone="sea" kind="tsunami" points={[
      <>An undersea earthquake lifts the seafloor and sends a wave toward the coast</>,
      <>A 4 km stretch of coast, an island and a sloping seabed</>,
      <><b>The question:</b> where does it flood, and how badly?</>,
    ]} />,
  notes: `Set up the scenario in one breath each: what happens, where, and the question.
(click) The physics: the shallow-water equations. The sea surface and the current, pushed by gravity, shaped by the seabed.`,
} satisfies SlideDef

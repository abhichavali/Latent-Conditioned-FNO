import type { SlideDef } from '../deck/SlideFrame'
import { Setup } from '../components/Setup'

export default {
  title: 'Wildfire spread',
  section: 'Applications',
  tone: 'ember',
  steps: 1,
  Component: () => <Setup n={3} title="Wildfire spread" tone="ember" kind="fire" points={[
      <>A fire burns through fuel, pushed by the wind</>,
      <>Burned ground can't burn again; lakes and roads have no fuel</>,
      <><b>The question:</b> from where, and how fast, does it reach the town?</>,
    ]} />,
  notes: `Set up the scenario in one breath each: what happens, where, and the question.
(click) The physics: heat spreads and ignites, the fuel burns down, the wind carries it.`,
} satisfies SlideDef

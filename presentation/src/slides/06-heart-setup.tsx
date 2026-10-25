import type { SlideDef } from '../deck/SlideFrame'
import { Setup } from '../components/Setup'

export default {
  title: 'Cardiac arrhythmia',
  section: 'Applications',
  tone: 'ember',
  steps: 1,
  Component: () => <Setup n={1} title="Cardiac arrhythmia" tone="ember" kind="heart" points={[
      <>Electrical waves sweep across heart tissue to trigger every beat</>,
      <>Scar tissue conducts slowly, and a wave can curl into a self-sustaining spiral: an arrhythmia</>,
      <><b>The question:</b> does this scar trigger an arrhythmia?</>,
    ]} />,
  notes: `Set up the scenario in one breath each: what happens, where, and the question.
(click) The physics: the Aliev–Panfilov model. Voltage spreads through the tissue; the recovery variable makes it refractory; the scar lowers the conductivity.`,
} satisfies SlideDef

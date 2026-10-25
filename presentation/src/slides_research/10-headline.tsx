import type { SlideDef } from '../deck/SlideFrame'
import { Frame, TwoCol, build } from '../components/layouts'
import { Chart } from '../components/Chart'
import { Stat } from '../components/Stat'
import data from '../data/baseline.json'

// PLACEHOLDER. Only the from-scratch curves are real (baseline_FNO/RESULTS.md). The few-shot model has no numbers yet:
// when they land, add a series here and replace the null stats. Never fill these in by hand.
const scratch = (rows: typeof data.swe) => rows.map(r => ({ x: r.n, y: r.rollout_mean[0] }))

export default {
  title: 'Headline result (placeholder)',
  tone: 'ember',
  section: 'Results',
  steps: 2,
  Component: () => (
    <Frame title="Goal: from-scratch accuracy at a fraction of the simulations">
      <TwoCol split="980px 1fr" gap={72} left={
        <Chart width={980} height={500} logX logY xDomain={[3.2, 1300]} yDomain={[0.02, 1]} xTicks={[4, 16, 64, 256, 1024]}
          xLabel="simulations of the new equation" yLabel="19-step rollout nRMSE" axisStep={0}
          band={{ from: 3.2, to: 24, label: 'few-shot model: pending', step: 1 }}
          series={[
            { name: 'from-scratch FNO, shallow water', points: scratch(data.swe), color: 'var(--sea)', step: 0 },
            { name: 'from-scratch FNO, Aliev–Panfilov', points: scratch(data.ap), color: 'var(--ember)', step: 0 },
          ]} />
      } right={
        <div style={{ display: 'grid', gap: 44, alignContent: 'start' }}>
          <div {...build(0)}><span className="pending-stamp">Placeholder — results pending</span></div>
          <Stat value={null} step={2} unit="×" label="fewer simulations to match the from-scratch FNO" />
          <Stat value={null} step={2} label="rollout nRMSE with a 16-simulation budget" />
          <p className="micro" {...build(2)}>Only the from-scratch curves are measured so far. Nothing on this slide is a few-shot result yet.</p>
        </div>
      } />
    </Frame>
  ),
  notes: `PLACEHOLDER: results pending. Do not quote numbers from this slide.
The real curves are the from-scratch FNO rollout errors from the baseline.
(click) The hatched region is where the few-shot model has to land: from-scratch accuracy with a budget of a few to a couple dozen simulations.
(click) These two numbers fill in from the few-shot evaluation.`,
} satisfies SlideDef

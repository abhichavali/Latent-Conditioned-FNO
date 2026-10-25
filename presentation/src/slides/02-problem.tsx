import type { SlideDef } from '../deck/SlideFrame'
import { Frame, Points, TwoCol, build } from '../components/layouts'
import { Stat } from '../components/Stat'

export default {
  title: 'Every new physics problem starts from zero',
  tone: 'sea',
  section: 'Motivation',
  steps: 3,
  Component: () => (
    <Frame title="Every new physics problem starts from zero">
      <TwoCol split="1.2fr 0.8fr" align="center" left={
        <Points items={[
          <><b>Solvers are accurate but slow.</b> A coastal hazard map or a patient-specific heart model needs many runs, not one.</>,
          <><b>Neural operators make each run nearly free,</b> but only after training on hundreds of solver runs.</>,
          <><b>New physics means starting over.</b> A new coastline or a scarred heart is a new equation, and the surrogate is retrained from scratch.</>,
        ]} />
      } right={
        <div style={{ display: 'grid', gap: 22 }}>
          <Stat value={512} step={2} label="solver runs before a from-scratch FNO stops improving on our tsunami test" />
          <p className="micro" {...build(2, 'fade', 0.4)}>Rollout nRMSE 0.068 at 512 runs, 0.065 at 1,024 (baseline_FNO/RESULTS.md).</p>
        </div>
      } />
    </Frame>
  ),
  notes: `PDE solvers are the workhorse of science and engineering, but they are slow, and real questions need many runs: every fault scenario for a coastline, every pacing site for a heart.
(click) Neural operators like the FNO fix the speed: one forward pass per run. But they need lots of solver data to train.
(click) Our own baseline: a from-scratch FNO on the tsunami keeps improving until about 512 simulations; 512 → 1,024 barely moves it (0.068 → 0.065).
(click) And all of that is for one equation. Change the coastline or put a scar in the heart and you pay again.`,
} satisfies SlideDef

import type { SlideDef } from '../deck/SlideFrame'
import { Frame, Points, Quote, TwoCol, build } from '../components/layouts'

export default {
  title: 'Takeaways',
  tone: 'sea',
  section: 'Takeaways',
  steps: 4,
  Component: () => (
    <Frame title="Takeaways">
      <TwoCol split="1fr 0.9fr" gap={120} align="start" left={
        <Points items={[
          <><b>One frozen operator</b> covers a family of PDEs; the equation enters only through a 32-number latent.</>,
          <><b>The symbolic equation is data.</b> Its graph gives a prior before a single simulation.</>,
          <><b>Active learning spends the budget</b> where plausible physics disagree, not at random.</>,
        ]} />
      } right={
        <div style={{ display: 'grid', gap: 64 }}>
          <Quote step={4} big>One operator, many physics, a handful of simulations.</Quote>
          <p style={{ font: '300 64px/1 var(--serif)' }} {...build(4, 'fade', 0.4)}>Thank you.</p>
        </div>
      } />
    </Frame>
  ),
  notes: `(click) One frozen operator; the equation only enters through z.
(click) The equation's symbolic form is itself informative: it gives a prior before any simulation.
(click) Active learning: simulate where the model is unsure.
(click) Thank you! Happy to take questions. The appendix has the full baseline figure.`,
} satisfies SlideDef

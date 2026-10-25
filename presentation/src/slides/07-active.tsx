import { useRef } from 'react'
import gsap from 'gsap'
import { useGSAP } from '@gsap/react'
import { useBuild, useMode, type SlideDef } from '../deck/SlideFrame'
import { reduced } from '../deck/motion'
import { Frame, Points, TwoCol, build } from '../components/layouts'
import { Diagram, type DNode, type DEdge } from '../components/Diagram'

// five stages on a circle, clockwise from the top
const C = { x: 440, y: 350 }, R = 262
const stages = ['Sample z', 'Score disagreement', 'Cluster', 'Simulate', 'Update posterior']
const tones: DNode['tone'][] = ['latent', 'latent', 'sea', 'ember', 'latent']
const nodes: DNode[] = stages.map((label, k) => {
  const a = (-90 + 72 * k) * (Math.PI / 180)
  return { id: `n${k}`, x: C.x + R * Math.cos(a), y: C.y + R * Math.sin(a), w: 236, h: 76, label, tone: tones[k], step: k + 1, shape: 'pill', size: 24 }
})
const edges: DEdge[] = stages.map((_, k) => ({ from: `n${k}`, to: `n${(k + 1) % 5}`, bend: -0.2, tone: 'ink', step: k === 4 ? 5 : k + 1, delay: k === 4 ? 0.3 : 0.45 }))

// belief over z: a wide prior that collapses to a narrow posterior (MorphSVG) at step 6
const bell = (sigma: number, amp: number) => {
  let d = ''
  for (let i = 0; i <= 60; i++) {
    const x = -150 + i * 5
    d += `${i ? 'L' : 'M'}${C.x + x} ${(C.y + 60 - amp * Math.exp(-(x * x) / (2 * sigma * sigma))).toFixed(1)}`
  }
  return d
}
const PRIOR = bell(58, 70), POSTERIOR = bell(16, 150)

function Belief() {
  const curve = useRef<SVGPathElement>(null), orbit = useRef<SVGGElement>(null), live = useMode() === 'live'
  useBuild((k, tl, at) => {
    if (k === 6) tl.to(curve.current, { morphSVG: POSTERIOR, stroke: 'var(--sea)', duration: 1.4, ease: 'power3.inOut' }, at)
  })
  useGSAP(() => { // a pulse that keeps travelling the loop once it is closed
    if (live && !reduced()) gsap.to(orbit.current, { rotation: 360, svgOrigin: `${C.x} ${C.y}`, duration: 7, ease: 'none', repeat: -1 })
  })
  return (
    <>
      <line x1={C.x - 160} x2={C.x + 160} y1={C.y + 60} y2={C.y + 60} stroke="var(--line-2)" {...build(0, 'appear')} />
      <path ref={curve} d={PRIOR} fill="none" stroke="var(--latent)" strokeWidth={3.5} {...build(0, 'draw', 0.4)} />
      <g {...build(6, 'appear', 0.8)}>
        <g ref={orbit}><circle cx={C.x} cy={C.y - R} r={9} fill="var(--sea)" style={{ filter: 'drop-shadow(0 0 10px var(--sea))' }} /></g>
      </g>
    </>
  )
}

export default {
  title: 'Spend simulations where the model is unsure',
  tone: 'latent',
  section: 'Method',
  steps: 6,
  Component: () => (
    <Frame title="Spend simulations where the model is unsure">
      <TwoCol split="880px 1fr" gap={72} left={
        <div style={{ position: 'relative', marginTop: -40 }}>
          <Diagram nodes={nodes} edges={edges} width={880} height={680}><Belief /></Diagram>
          <div className="micro" style={{ position: 'absolute', left: C.x - 200, width: 400, top: C.y + 78, textAlign: 'center' }}>
            <div {...build(0, 'appear')}><div {...build(6, 'out')}>prior over z, from the equation</div></div>
            <div style={{ position: 'absolute', inset: 0 }} {...build(6, 'appear', 0.6)}>posterior after a few simulations</div>
          </div>
        </div>
      } right={
        <Points items={[
          <><b>Sample</b> plausible latents z from the current belief.</>,
          <><b>Score</b> candidate initial conditions by how much those latents' predictions disagree.</>,
          <><b>Cluster</b> the top candidates (k-means on their knobs) so picks stay diverse.</>,
          <><b>Simulate</b> only the chosen few with the real solver.</>,
          <><b>Update</b> the posterior over z, and repeat until the budget is spent.</>,
        ]} />
      } />
    </Frame>
  ),
  notes: `This is the active few-shot loop, Bayesian optimal experimental design on a fixed simulation budget.
(click) Sample latents from the current belief, starting with the encoder's prior.
(click) For many candidate initial conditions (fault sources, broken wave fronts, stimuli), run the frozen FNO under each sampled z and score how much they disagree. Disagreement = information.
(click) Cluster the high-scoring candidates with k-means on their knob features so we don't spend the budget on near-duplicates.
(click) Simulate just those with the real solver.
(click) Update the posterior over z with the new trajectories, and go around again.
(click) The belief collapses from the wide prior to a sharp posterior, without touching the network weights.`,
} satisfies SlideDef

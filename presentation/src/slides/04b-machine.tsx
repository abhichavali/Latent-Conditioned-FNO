import type { SlideDef } from '../deck/SlideFrame'
import { build } from '../components/layouts'
import { Tex } from '../components/Tex'

// The machine: an unseen equation goes in, its graph is read into a latent code (a first guess), a handful of
// simulations sharpen the code, and the neural operator turns the code into a forecast of the dynamics.
const N = 16
const wave = (k: number, p: number) => Array.from({ length: N }, (_, i) => 0.25 + 0.7 * Math.abs(Math.sin(i * k + p)))
const GUESS = wave(0.9, 0.4), SHARP = wave(1.7, 1.1)
const RING = [10, 18, 27, 36, 46] // forecast frames: the fire front's radius (%)

const Arrow = ({ step, delay = 0 }: { step: number; delay?: number }) => (
  <svg viewBox="0 0 70 24" className="m-arrow"><path d="M2 12 H62 M52 3 L66 12 L52 21" {...build(step, 'draw', delay)} /></svg>
)

export default {
  title: 'How Simula predicts a new equation',
  tone: 'latent',
  section: 'How it works',
  steps: 3,
  build: (k, tl, at) => {
    if (k === 1) tl.fromTo('.zbar', { scaleY: 0 }, { scaleY: i => GUESS[i], duration: 0.9, ease: 'power3.out', stagger: 0.03 }, `${at}+=0.6`)
    if (k === 2) tl.to('.m-state .s1', { autoAlpha: 0, duration: 0.3 }, at)
    if (k === 2) tl.to('.zbar', { scaleY: i => SHARP[i], duration: 1.1, ease: 'power2.inOut', stagger: 0.02 }, `${at}+=0.5`)
    if (k === 3) tl.fromTo('.m-frame', { autoAlpha: 0, x: -16 }, { autoAlpha: 1, x: 0, duration: 0.5, stagger: 0.18 }, `${at}+=0.9`)
  },
  Component: () => (
    <div className="frame machine">
      <h2 className="slide-head" {...build(0, 'words')}>From equation to forecast</h2>
      <div className="m-row">
        <div className="m-card glass" {...build(0, 'fade', 0.4)}>
          <span className="kicker c-ember">Unseen equation</span>
          <Tex block>{'\\partial_t u = \\nabla^2 u + k\\,u(1-u)(u-a) - \\gamma u(1-f) - \\mathbf{w}\\!\\cdot\\!\\nabla u'}</Tex>
          <Tex block>{'\\partial_t f = -\\beta\\,u\\,f'}</Tex>
          <small>wildfire: never seen in training</small>
        </div>
        <Arrow step={1} />
        <div className="m-box" {...build(1, 'pop')}><b>Graph encoder</b><span>reads the equation</span></div>
        <Arrow step={1} delay={0.3} />
        <div className="m-latent" {...build(1, 'fade', 0.4)}>
          <span className="kicker">Latent code z</span>
          <svg viewBox={`0 0 ${N * 14} 120`} className="zbars">
            {Array.from({ length: N }, (_, i) => <rect key={i} className="zbar" x={i * 14 + 2} y={10} width={10} height={110} rx={3} />)}
          </svg>
          <span className="m-state">
            <span {...build(1, 'appear', 0.8)} className="s1">first guess, from the equation</span>
            <span {...build(2, 'appear', 0.4)} className="s2">sharpened by a few simulations</span>
          </span>
        </div>
        <Arrow step={3} />
        <div className="m-box sea" {...build(3, 'pop', 0.2)}><b>Neural operator</b><span>runs the physics the code describes</span></div>
        <Arrow step={3} delay={0.4} />
        <div className="m-out" {...build(3, 'fade', 0.6)}>
          <span className="kicker c-sea">Forecast</span>
          <div className="m-frames">
            {RING.map(r => <i key={r} className="m-frame" style={{ background: `radial-gradient(circle at 38% 55%, #3a2a20 0 ${r - 5}%, #ffd27a ${r - 2}%, #ff5a1f ${r}%, #4f7a2a ${r + 3}%, #34561d 100%)` }} />)}
          </div>
          <small>u and f over time</small>
        </div>
      </div>
      <div className="m-sims glass" {...build(2, 'fade')}>
        <span className="kicker">+ a handful of simulations</span>
        <div className="m-simrow">{[0, 1, 2, 3, 4].map(i => <i key={i} style={{ background: `radial-gradient(circle at ${30 + i * 9}% ${60 - i * 6}%, #ffb070 0 ${8 + i * 3}%, #ff5a1f ${12 + i * 3}%, #34561d ${16 + i * 4}%)` }} />)}</div>
        <small>5 runs of the real simulator update the code, so the forecast fits this problem</small>
      </div>
    </div>
  ),
  notes: `(~15 s) This is the machine.
An equation it has never seen: here, wildfire. (click) The graph encoder reads its graph and writes a latent code: a compact description of how this system behaves. On its own that's only a first guess.
(click) Then we give it a handful of real simulations, five or so, and the code sharpens to fit this problem. That's where the "12× fewer simulations" comes from.
(click) The neural operator takes the code and the current state and runs the physics forward: that's the forecast.
If asked: today the gain mostly comes from the simulations step (we also fine-tune the operator on them); the equation-only guess is rough on brand-new physics. Our next version learns the update from simulations directly.`,
} satisfies SlideDef

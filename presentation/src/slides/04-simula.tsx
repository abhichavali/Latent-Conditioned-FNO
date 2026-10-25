import gsap from 'gsap'
import type { SlideDef } from '../deck/SlideFrame'
import { build } from '../components/layouts'
import { Tex } from '../components/Tex'
import { useAmbient } from '../components/useAmbient'

// 1. Learned physics: many kinds of waves, folded into one model.
function Learned() {
  const ref = useAmbient(() => gsap.to('.wave', { x: -120, duration: 3, ease: 'none', repeat: -1 }))
  const wave = (k: number, a: number, y: number) =>
    Array.from({ length: 81 }, (_, i) => `${i ? "L" : "M"}${i * 6 - 60} ${(y + a * Math.sin((i * 6 * k * Math.PI) / 120)).toFixed(1)}`).join('')
  return (
    <svg ref={ref} viewBox="0 0 240 240" className="icon">
      <defs><clipPath id="disc"><circle cx="120" cy="120" r="96" /></clipPath></defs>
      <circle cx="120" cy="120" r="104" className="halo" {...build(0, 'draw', 0.2)} />
      <g clipPath="url(#disc)">
        <path className="wave" d={wave(2, 18, 82)} stroke="#7fd4ff" />
        <path className="wave" d={wave(4, 12, 120)} stroke="#b9a6ff" />
        <path className="wave" d={wave(6, 8, 156)} stroke="#ff7a45" />
      </g>
    </svg>
  )
}

// 3. Learns from a few runs: fine-tunes on a handful of simulations of the new problem.
const DOTS = Array.from({ length: 49 }, (_, i) => ({ x: 48 + (i % 7) * 24, y: 48 + Math.floor(i / 7) * 24 }))
const PICKED = [9, 25, 39]
function Picks({ step }: { step: number }) {
  return (
    <svg viewBox="0 0 240 240" className="icon">
      <circle cx="120" cy="120" r="104" className="halo" {...build(step, 'draw', 0.2)} />
      {DOTS.map((d, i) => <circle key={i} cx={d.x} cy={d.y} r="3.5" fill="#4b5567" />)}
      {PICKED.map((i, n) => (
        <g key={i} {...build(step, 'pop', 0.6 + n * 0.25)}>
          <circle cx={DOTS[i].x} cy={DOTS[i].y} r="13" fill="none" stroke="#7fd4ff" strokeWidth="3" />
          <circle cx={DOTS[i].x} cy={DOTS[i].y} r="5.5" fill="#7fd4ff" />
        </g>
      ))}
    </svg>
  )
}

export default {
  title: 'Simula',
  section: 'Simula',
  steps: 3,
  Component: () => (
    <div className="frame simula">
      <h2 className="slide-head" {...build(0, 'words')}>Simula</h2>
      <div className="steps3">
        <div className="how" {...build(0, 'fade', 0.3)}>
          <Learned />
          <h3>Learned physics</h3>
          <p>Learned from <b>5,000+ kinds of physics</b> before seeing your problem</p>
          <span className="micro">5,120 equations in pretraining</span>
        </div>
        <svg className="arrow" viewBox="0 0 80 24" {...build(1, 'fade')}><path d="M0 12 H72 M60 2 L74 12 L60 22" /></svg>
        <div className="how" {...build(1, 'fade', 0.2)}>
          <div className="icon eq">
            <svg viewBox="0 0 240 240" className="icon"><circle cx="120" cy="120" r="104" className="halo" {...build(1, 'draw', 0.3)} /></svg>
            <Tex>{String.raw`\partial_t u = \mathcal{N}(u)`}</Tex>
            <i className="scan" />
          </div>
          <h3>Reads your equation</h3>
          <p>Reads your equation and starts with an <b>educated guess</b></p>
        </div>
        <svg className="arrow" viewBox="0 0 80 24" {...build(2, 'fade')}><path d="M0 12 H72 M60 2 L74 12 L60 22" /></svg>
        <div className="how" {...build(2, 'fade', 0.2)}>
          <Picks step={2} />
          <h3>Learns from a few runs</h3>
          <p>Needs only a <b>handful</b> of simulations of your problem</p>
        </div>
      </div>
      <p className="punch" {...build(3, 'words')}>A handful of simulations instead of thousands.</p>
    </div>
  ),
  notes: `(1:10–1:40)

Simula learned from 5,000+ kinds of physics before it ever sees your problem. (5,120 equations in pretraining.)
(click) It reads your equation and starts with an educated guess.
(click) Then it learns your problem from just a handful of simulations. (We tested smart picking vs random picking: random was as good or better, so we don't claim smart picking.)
(click) Punchline: "A handful of simulations instead of thousands."`,
} satisfies SlideDef

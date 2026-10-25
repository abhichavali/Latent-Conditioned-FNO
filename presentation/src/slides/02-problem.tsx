import gsap from 'gsap'
import type { SlideDef } from '../deck/SlideFrame'
import { build } from '../components/layouts'
import { useAmbient } from '../components/useAmbient'

// An earthquake offshore, its wave rings heading for a town on the coast, and a clock that won't stop.
function Quake() {
  const ref = useAmbient(() => {
    gsap.fromTo('.ring', { scale: 0.15, opacity: 0.9, transformOrigin: '50% 50%' },
      { scale: 1, opacity: 0, duration: 3.6, ease: 'none', repeat: -1, stagger: { each: 1.2 } })
    gsap.to('.hand', { rotation: 360, svgOrigin: '612 74', duration: 4, ease: 'none', repeat: -1 })
  })
  const shore = 'C495 60 540 110 505 165 C480 205 520 250 498 300 C485 330 505 360 492 380'
  return (
    <svg ref={ref} viewBox="0 0 740 380" className="pic">
      <defs>
        <linearGradient id="ocean" x1="0" x2="1"><stop offset="0" stopColor="#06233a" /><stop offset="1" stopColor="#0e4a6b" /></linearGradient>
        <clipPath id="sea"><path d={`M0 0 L520 0 ${shore} L0 380 Z`} /></clipPath>
      </defs>
      <rect width="740" height="380" fill="url(#ocean)" />
      <g clipPath="url(#sea)">
        {[0, 1, 2].map(i => <circle key={i} className="ring" cx="170" cy="190" r="420" fill="none" stroke="#7fd4ff" strokeWidth="5" />)}
      </g>
      <path d={`M520 0 ${shore} L740 380 L740 0 Z`} fill="#1d2a22" stroke="#e8d6ad" strokeWidth="3" />
      <ellipse cx="330" cy="130" rx="34" ry="22" fill="#2a3a2c" stroke="#e8d6ad" strokeWidth="2.5" />
      {Array.from({ length: 14 }, (_, i) => (
        <rect key={i} x={548 + (i % 4) * 34 + (i % 3) * 5} y={150 + Math.floor(i / 4) * 58 + (i % 2) * 16} width="14" height="14" rx="2" fill="#ffd9a3" opacity="0.85" />
      ))}
      <circle cx="170" cy="190" r="9" fill="#ffc477" />
      <g transform="translate(612 74)">
        <circle r="38" fill="rgba(5,7,11,0.7)" stroke="#e9eef5" strokeWidth="3" />
        {Array.from({ length: 12 }, (_, i) => <line key={i} y1="-30" y2="-25" stroke="#8c97a8" strokeWidth="2" transform={`rotate(${i * 30})`} />)}
      </g>
      <line className="hand" x1="612" y1="74" x2="612" y2="46" stroke="#ff7a45" strokeWidth="4" strokeLinecap="round" />
    </svg>
  )
}

// A heart with an arrhythmia spiral and dozens of candidate ablation sites.
const heartPt = (t: number, r = 1): [number, number] => [ // classic parametric heart; r < 1 shrinks it toward (370, 160)
  370 + r * 11 * 16 * Math.sin(t) ** 3,
  160 - r * 11 * (13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)),
]
const HEART = Array.from({ length: 121 }, (_, i) => heartPt((i / 120) * 2 * Math.PI)).map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join('') + 'Z'
const SITES = Array.from({ length: 30 }, (_, i) => heartPt(i * 2.39996, 0.25 + 0.6 * Math.sqrt((i * 0.7548) % 1)))
const SPIRAL = Array.from({ length: 90 }, (_, i) => { const a = i * 0.17, r = 4 + i * 0.9; return `${i ? 'L' : 'M'}${(330 + r * Math.cos(a)).toFixed(1)} ${(205 + r * Math.sin(a)).toFixed(1)}` }).join('')

function Heart() {
  const ref = useAmbient(() => {
    gsap.to('.spiral', { rotation: -360, svgOrigin: '330 205', duration: 3.2, ease: 'none', repeat: -1 })
    gsap.fromTo('.site', { scale: 0, transformOrigin: '50% 50%' }, { scale: 1, duration: 0.5, ease: 'back.out(3)', stagger: 0.09, delay: 0.6 })
  })
  return (
    <svg ref={ref} viewBox="0 0 740 380" className="pic">
      <defs>
        <radialGradient id="tissue" cx="0.45" cy="0.45"><stop offset="0" stopColor="#7a1a22" /><stop offset="1" stopColor="#2a070c" /></radialGradient>
        <clipPath id="in-heart"><path d={HEART} /></clipPath>
      </defs>
      <rect width="740" height="380" fill="#0a0507" />
      <path d={HEART} fill="url(#tissue)" stroke="#ff7a45" strokeOpacity="0.5" strokeWidth="2.5" />
      <g clipPath="url(#in-heart)">
        <path className="spiral" d={SPIRAL} fill="none" stroke="#ffb27a" strokeWidth="9" strokeLinecap="round" opacity="0.9" />
      </g>
      {SITES.map(([x, y], i) => <circle key={i} className="site" cx={x} cy={y} r="7" fill="none" stroke="#e9eef5" strokeWidth="2.5" opacity="0.85" />)}
    </svg>
  )
}

export default {
  title: 'The problem',
  section: 'The problem',
  tone: 'ember',
  steps: 2,
  Component: () => (
    <div className="frame">
      <h2 className="kicker-title" {...build(0, 'fade')}>Two decisions that can't wait</h2>
      <div className="stakes">
        <figure className="stake glass t-sea" {...build(0, 'fade', 0.2)}>
          <Quake />
          <figcaption><span className="kicker">Tsunami</span>An earthquake hits. Forecasters have <b>minutes</b> to decide who evacuates.</figcaption>
        </figure>
        <figure className="stake glass t-ember" {...build(1)}>
          <Heart />
          <figcaption><span className="kicker">Heart</span>A surgeon has <b>dozens</b> of possible ablation sites, and each one needs a simulation.</figcaption>
        </figure>
      </div>
      <p className="punch wide" {...build(2, 'words')}>Simulation is too slow for decisions that can't wait.</p>
    </div>
  ),
  notes: `(0:20–0:50)

Tsunami: an earthquake hits, and forecasters have minutes to decide who evacuates.
(click) Heart: a surgeon has dozens of possible ablation sites, and each one needs a simulation.
(click) Punchline: "Simulation is too slow for decisions that can't wait."`,
} satisfies SlideDef

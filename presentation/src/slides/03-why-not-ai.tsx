import { useRef } from 'react'
import type { SlideDef } from '../deck/SlideFrame'
import { useBuild } from '../deck/SlideFrame'
import { build } from '../components/layouts'

// Three heaps of "simulations needed", one per new problem. Each tile is one simulation (illustrative, not a count).
const PILES = [
  { label: 'Tsunami', color: '#7fd4ff' },
  { label: 'Heart', color: '#ff7a45' },
  { label: 'Your next problem', color: '#b9a6ff' },
]
const T = 17, P = 20, ROWS = 13, BASE = 25, W = 1648, H = 330 // tile, pitch, rows, bottom-row tiles
const hash = (n: number) => { const s = Math.sin(n * 91.345) * 47453.53; return s - Math.floor(s) }
const TILES = PILES.flatMap((pile, p) => {
  const cx = (W / 3) * (p + 0.5), out = []
  for (let r = 0; r < ROWS; r++) {
    const n = BASE - 2 * r + (r % 2)
    for (let i = 0; i < n; i++) {
      const k = p * 1000 + r * 50 + i
      out.push({ x: cx + (i - (n - 1) / 2) * P - T / 2 + (hash(k) - 0.5) * 4, y: H - (r + 1) * (P - 1) + (hash(k + 7) - 0.5) * 3,
        rot: (hash(k + 3) - 0.5) * 14, o: 0.45 + 0.55 * hash(k + 5), color: pile.color, r })
    }
  }
  return out
}).sort((a, b) => a.r - b.r) // bottom rows land first

function Piles() {
  const ref = useRef<SVGSVGElement>(null)
  useBuild((k, tl, at) => {
    if (k !== 1 || !ref.current) return
    tl.fromTo(ref.current.querySelectorAll('.tile'), { y: -620, autoAlpha: 0 },
      { y: 0, autoAlpha: 1, duration: 0.9, ease: 'bounce.out', stagger: { each: 0.0045 } }, at)
  })
  return (
    <svg ref={ref} viewBox={`0 0 ${W} ${H + 70}`} width={W} height={H + 70} className="piles" aria-hidden>
      {TILES.map((t, i) => (
        <g key={i} transform={`rotate(${t.rot.toFixed(1)} ${(t.x + T / 2).toFixed(1)} ${(t.y + T / 2).toFixed(1)})`}>
          <rect className="tile" x={t.x} y={t.y} width={T} height={T} rx="3.5" fill={t.color} fillOpacity={t.o} />
        </g>
      ))}
      {PILES.map((p, i) => (
        <text key={p.label} x={(W / 3) * (i + 0.5)} y={H + 52} textAnchor="middle" className="pile-label" {...build(1, 'fade', 0.4)}>{p.label}</text>
      ))}
    </svg>
  )
}

export default {
  title: "Why AI hasn't fixed it",
  section: "Why AI hasn't fixed it",
  tone: 'latent',
  steps: 2,
  Component: () => (
    <div className="frame why">
      <p className="big-line" {...build(0, 'words')}>AI shortcuts are fast…</p>
      <p className="big-line second" {...build(1, 'words')}>…but they need <b>thousands</b> of simulations to learn <em>each</em> new problem.</p>
      <Piles />
      <p className="footnote" {...build(1, 'fade', 2)}>Our standard-AI baseline: trained from scratch on up to 1,024 simulations per problem.</p>
      <p className="punch" {...build(2, 'words')}>The training cost just moves the bottleneck.</p>
    </div>
  ),
  notes: `(0:50–1:10)

"AI shortcuts are fast..."
(click) "...but they need thousands of simulations to learn each new problem." (the piles drop: simulations needed for every new physics problem)
(click) Punchline: "The training cost just moves the bottleneck."

Real number if asked: our from-scratch FNO baseline was trained on up to 1,024 simulations per problem (baseline_FNO/RESULTS.md).`,
} satisfies SlideDef

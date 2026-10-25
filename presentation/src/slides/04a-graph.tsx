import type { SlideDef } from '../deck/SlideFrame'
import { build } from '../components/layouts'
import { Tex } from '../components/Tex'
import { Diagram, type DNode, type DEdge } from '../components/Diagram'

// The Aliev–Panfilov u-equation as its expression graph (same as the research deck's), then one training example
// (graph + simulations), then a wall of them: the pretraining set.
const leaf = (id: string, y: number, label: string, tone: DNode['tone'] = 'ink', x = 60): DNode =>
  ({ id, x, y, label: <Tex>{label}</Tex>, shape: 'circle', tone, step: 1, w: id === 'u' ? 92 : 76, size: id.startsWith('c') ? 22 : 30 })
const op = (id: string, x: number, y: number, label: string, step: number): DNode =>
  ({ id, x, y, label: <Tex>{label}</Tex>, shape: 'circle', tone: 'latent', step, w: 76, size: 30 })
const nodes: DNode[] = [
  leaf('c1', 165, '1'), leaf('u', 300, 'u', 'ember'), leaf('c015', 425, '0.15'), leaf('cm1', 530, '-1'), leaf('v', 630, 'v', 'ember'),
  leaf('c8', 50, '8', 'ink', 290),
  op('sub1', 260, 150, '-', 1), op('lap', 260, 290, '\\nabla^2', 1), op('sub2', 260, 430, '-', 1),
  op('mulc', 480, 230, '\\times', 2), op('muluv', 480, 560, '\\times', 2), op('add', 680, 380, '+', 2),
  { id: 'root', x: 840, y: 380, label: <Tex>{'\\partial_t u'}</Tex>, shape: 'pill', tone: 'sea', step: 2, w: 120, h: 76, size: 30 },
]
const e = (from: string, to: string, bend = 0): DEdge => ({ from, to, bend })
const edges: DEdge[] = [
  e('c1', 'sub1'), e('u', 'sub1'), e('u', 'lap'), e('u', 'sub2'), e('c015', 'sub2'),
  e('c8', 'mulc'), e('u', 'mulc', -0.2), e('sub1', 'mulc'), e('sub2', 'mulc'),
  e('cm1', 'muluv'), e('u', 'muluv', 0.25), e('v', 'muluv'),
  e('lap', 'add'), e('mulc', 'add'), e('muluv', 'add'),
  { from: 'add', to: 'root', tone: 'sea' },
]

// deterministic pseudo-random, so the wall looks the same every time
const rnd = (seed: number) => { let s = Math.imul(seed ^ 0x9e3779b9, 0x85ebca6b) >>> 0; return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 2 ** 32 } }
const HUES = ['#7fd4ff', '#ff7a45', '#b9a6ff', '#ffd27a']

function MiniGraph({ seed, size = 1 }: { seed: number; size?: number }) {
  const r = rnd(seed), leaves = 3 + Math.floor(r() * 3), ops = 2 + Math.floor(r() * 2)
  const L = Array.from({ length: leaves }, (_, i) => [8, 6 + (i * 36) / (leaves - 1)])
  const O = Array.from({ length: ops }, (_, i) => [32, 10 + (i * 28) / Math.max(ops - 1, 1)])
  const root = [56, 24]
  return (
    <svg viewBox="0 0 64 48" width={64 * size} height={48 * size} className="mini-graph">
      {L.map(([x, y], i) => <line key={`l${i}`} x1={x} y1={y} x2={O[i % ops][0]} y2={O[i % ops][1]} />)}
      {O.map(([x, y], i) => <line key={`o${i}`} x1={x} y1={y} x2={root[0]} y2={root[1]} />)}
      {L.map(([x, y], i) => <circle key={`L${i}`} cx={x} cy={y} r={3.2} className={r() < 0.4 ? 'f' : 'c'} />)}
      {O.map(([x, y], i) => <circle key={`O${i}`} cx={x} cy={y} r={4} className="o" />)}
      <circle cx={root[0]} cy={root[1]} r={5} className="r" />
    </svg>
  )
}

function Frames({ seed, n }: { seed: number; n: number }) {
  const r = rnd(seed * 7 + 3), hue = HUES[Math.floor(r() * HUES.length)]
  return (
    <div className="frames">
      {Array.from({ length: n }, (_, i) => {
        const x = 20 + r() * 60, y = 20 + r() * 60, s = 30 + i * 12
        return <i key={i} style={{ background: `radial-gradient(circle at ${x}% ${y}%, ${hue} 0, ${hue}55 ${s * 0.4}%, #0b1220 ${s}%)` }} />
      })}
    </div>
  )
}

const WALL = Array.from({ length: 32 }, (_, i) => i + 11)
const GPUS = ['RTX 5090', 'RTX 5080', 'RTX PRO 4000', 'RTX PRO 4000']
const Gpu = () => (
  <svg viewBox="0 0 120 56" className="gpu-icon">
    <rect x="2" y="6" width="116" height="40" rx="6" />
    <circle cx="36" cy="26" r="13" /><circle cx="84" cy="26" r="13" />
    <path d="M36 15 v22 M25 26 h22 M84 15 v22 M73 26 h22" />
    <path d="M14 46 v8 M24 46 v8 M34 46 v8 M44 46 v8 M54 46 v8 M64 46 v8" />
  </svg>
)

export default {
  title: 'Every equation becomes a graph',
  tone: 'latent',
  section: 'How it works',
  steps: 4,
  build: (k, tl, at) => {
    if (k === 4) tl.fromTo('.wall .mini', { autoAlpha: 0, scale: 0.6 }, { autoAlpha: 1, scale: 1, duration: 0.45, ease: 'back.out(1.6)', stagger: 0.025 }, at)
  },
  Component: () => (
    <div className="frame graph-slide">
      <h2 className="slide-head" {...build(0, 'words')}>Every equation becomes a graph</h2>
      <div className="graph-body">
        <div className="graph-left">
          <div className="graph-eq" {...build(0, 'fade', 0.4)}><Tex block>{'\\partial_t u = \\nabla^2 u + 8u(1-u)(u-0.15) - uv'}</Tex></div>
          <div className="graph-diagram"><Diagram nodes={nodes} edges={edges} width={900} height={690} /></div>
        </div>
        <div className="graph-right">
          <div className="example glass" {...build(3, 'pop')}>
            <span className="kicker">One training example</span>
            <div className="example-row">
              <MiniGraph seed={3} size={1.6} />
              <span className="plus">+</span>
              <Frames seed={3} n={4} />
            </div>
            <small>the equation's graph + simulations of it from different starting states</small>
          </div>
          <div className="wall">
            {WALL.map(s => <div key={s} className="mini"><MiniGraph seed={s} size={0.62} /><Frames seed={s} n={2} /></div>)}
          </div>
          <p className="wall-count" {...build(4, 'fade', 0.9)}><b>5,120</b> equations · <b>197,632</b> simulations</p>
          <div className="gpus" {...build(4, 'fade', 1.4)}>
            <span className="kicker">Generated and trained on rented GPUs</span>
            <div className="gpu-row">{GPUS.map(g => <div key={g} className="gpu"><Gpu /><span>{g}</span></div>)}</div>
          </div>
        </div>
      </div>
    </div>
  ),
  notes: `(~15 s) How Simula sees an equation.
Here's the heart equation. (click) Simula doesn't read it as text: the variables and constants become nodes... (click) ...and every operation joins them, up to the time derivative. Same variable, one node.
(click) Pair that graph with simulations of the equation and you have one training example.
(click) We generated 5,120 equations this way, about 200,000 simulations, with tsunamis, hearts and wildfires deliberately left out. That's what Simula learned physics from: all generated and trained on four rented GPUs (an RTX 5090, an RTX 5080 and two RTX PRO 4000s).`,
} satisfies SlideDef

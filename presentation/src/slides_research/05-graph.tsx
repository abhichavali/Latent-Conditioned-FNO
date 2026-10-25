import type { SlideDef } from '../deck/SlideFrame'
import { Frame, TwoCol, build } from '../components/layouts'
import { Tex } from '../components/Tex'
import { Code } from '../components/Code'
import { Diagram, type DNode, type DEdge } from '../components/Diagram'

// Aliev–Panfilov u-equation (data_collection/demos.py) as its expression DAG, leaves left, root right.
// Simplified view: sympy stores a - b as a + (-1)*b; here subtraction is one node.
const leaf = (id: string, y: number, label: string, tone: DNode['tone'] = 'ink', x = 60): DNode =>
  ({ id, x, y, label: <Tex>{label}</Tex>, shape: 'circle', tone, step: 1, w: id === 'u' ? 92 : 76, size: id.startsWith('c') ? 22 : 30 })
const op = (id: string, x: number, y: number, label: string, step: number): DNode =>
  ({ id, x, y, label: <Tex>{label}</Tex>, shape: 'circle', tone: 'latent', step, w: 76, size: 30 })

const nodes: DNode[] = [
  leaf('c1', 165, '1'), leaf('u', 300, 'u', 'ember'), leaf('c015', 425, '0.15'), leaf('cm1', 530, '-1'), leaf('v', 630, 'v', 'ember'),
  leaf('c8', 50, '8', 'ink', 290),
  op('sub1', 260, 150, '-', 2), op('lap', 260, 290, '\\nabla^2', 2), op('sub2', 260, 430, '-', 2),
  op('mulc', 480, 230, '\\times', 3), op('muluv', 480, 560, '\\times', 3),
  op('add', 680, 380, '+', 4),
  { id: 'root', x: 840, y: 380, label: <Tex>{'\\partial_t u'}</Tex>, shape: 'pill', tone: 'sea', step: 4, w: 120, h: 76, size: 30 },
  { id: 'ring', x: 60, y: 300, label: '', shape: 'circle', className: 'ghost', tone: 'ember', step: 5, w: 132 }, // highlights the shared leaf
]
const e = (from: string, to: string, bend = 0): DEdge => ({ from, to, bend })
const edges: DEdge[] = [
  e('c1', 'sub1'), e('u', 'sub1'), e('u', 'lap'), e('u', 'sub2'), e('c015', 'sub2'),
  e('c8', 'mulc'), e('u', 'mulc', -0.2), e('sub1', 'mulc'), e('sub2', 'mulc'),
  e('cm1', 'muluv'), e('u', 'muluv', 0.25), e('v', 'muluv'),
  e('lap', 'add'), e('mulc', 'add'), e('muluv', 'add'),
  { from: 'add', to: 'root', tone: 'sea' },
]

const spec = `
{
  "fields": ["u", "v"],
  "L": 64.0, "N": 128,
  "rhs": {
    "u": "1.0*lap(u) + 8*u*(1 - u)*(u - 0.15) - u*v",
    "v": "0.05*lap(v) + (0.01 + 0.2*v/(u + 0.3))*…"
  }
}`

export default {
  title: 'An equation is a graph',
  tone: 'latent',
  section: 'Method',
  steps: 5,
  Component: () => (
    <Frame title="An equation is a graph" lede="The encoder reads the symbolic right-hand side, not simulation data.">
      <TwoCol split="684px 900px" gap={64} left={
        <div style={{ display: 'grid', gap: 40 }}>
          <div style={{ fontSize: 40 }} {...build(0, 'fade', 0.4)}>
            <Tex block>{'\\partial_t u = \\nabla^2 u + 8u(1-u)(u-0.15) - uv'}</Tex>
          </div>
          <Code code={spec} size={18} step={0} focus={{ 1: [5], 5: [] }} />
          <p className="small" {...build(5)}>
            <span className="c-ember">u</span> feeds five operators but is stored once: identical subexpressions are shared, so it is a DAG,
            and every one of the 5,120 training equations and both held-out targets uses this format.
          </p>
        </div>
      } right={<Diagram nodes={nodes} edges={edges} width={900} height={690} />} />
    </Frame>
  ),
  notes: `Every PDE we generate, and both demo targets, is a JSON spec with one sympy right-hand side per field.
We turn that into a graph: fields, static fields and constants are leaves, operators are internal nodes, and each field gets a time-derivative root.
(click) leaves  (click) first operators  (click) products  (click) the sum and the root.
(click) Note u: one node, five parents. Shared subexpressions make it a DAG.
The graph encoder reads exactly this, so it can say something about an equation it has never simulated.
Simplified for the slide: sympy writes a - b as a + (-1)·b.`,
} satisfies SlideDef

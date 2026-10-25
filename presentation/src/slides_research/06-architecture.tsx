import type { SlideDef } from '../deck/SlideFrame'
import { Frame, build } from '../components/layouts'
import { Tex } from '../components/Tex'
import { Diagram, type DNode, type DEdge } from '../components/Diagram'

const nodes: DNode[] = [
  { id: 'G', x: 150, y: 150, w: 250, h: 116, label: 'Equation graph', sub: 'symbolic DAG G', tone: 'latent', step: 1 },
  { id: 'enc', x: 520, y: 150, w: 250, h: 116, label: 'Graph encoder', sub: <Tex>{'p_\\phi(z \\mid G)'}</Tex>, tone: 'latent', step: 1 },
  { id: 'prior', x: 890, y: 150, w: 250, h: 116, label: 'Prior over z', sub: <Tex>{'\\mathcal N(\\mu_\\phi,\\, \\mathrm{diag}\\,\\sigma_\\phi^2)'}</Tex>, tone: 'latent', step: 2 },
  { id: 'z', x: 1220, y: 150, w: 112, label: <Tex>z</Tex>, sub: '32-d', shape: 'circle', tone: 'latent', step: 2, size: 40 },
  { id: 'u0', x: 890, y: 450, w: 250, h: 116, label: 'Initial condition', sub: <Tex>{'u_0(x),\\; t'}</Tex>, tone: 'sea', step: 3 },
  { id: 'fno', x: 1220, y: 450, w: 290, h: 136, label: 'FNO, frozen', sub: 'FiLM + mixture of spectral filter banks', tone: 'sea', step: 3 },
  { id: 'out', x: 1540, y: 450, w: 190, h: 116, label: 'Prediction', sub: <Tex>{'\\hat u(x, t)'}</Tex>, tone: 'sea', step: 4 },
  { id: 'sim', x: 1540, y: 150, w: 190, h: 116, label: 'Solver', sub: 'a few runs', tone: 'ember', step: 5 },
]
const edges: DEdge[] = [
  { from: 'G', to: 'enc', tone: 'latent' },
  { from: 'enc', to: 'prior', tone: 'latent' },
  { from: 'prior', to: 'z', tone: 'latent', label: 'sample' },
  { from: 'z', to: 'fno', tone: 'latent', label: 'conditions' },
  { from: 'u0', to: 'fno', tone: 'sea' },
  { from: 'fno', to: 'out', tone: 'sea' },
  { from: 'out', to: 'sim', dashed: true, tone: 'ember', label: 'compare' },
  { from: 'sim', to: 'prior', bend: 0.3, dashed: true, tone: 'ember', label: 'update the posterior over z' },
]

export default {
  title: 'One frozen operator, steered by a latent code',
  tone: 'latent',
  section: 'Method',
  steps: 5,
  Component: () => (
    <Frame title="One frozen operator, steered by a latent code">
      <Diagram nodes={nodes} edges={edges} width={1648} height={560} />
      <p className="small" style={{ maxWidth: '78ch', marginTop: 8 }} {...build(5, 'fade', 1)}>
        Pretraining learns the FNO, one latent per training equation, and an encoder that predicts it from the graph.
        On a new equation the network stays fixed; only the belief about z moves.
      </p>
    </Frame>
  ),
  notes: `The architecture, from model/README.md.
(click) The equation graph goes through a graph encoder...
(click) ...which outputs a Gaussian prior over a 32-dimensional latent z. We sample from it.
(click) z conditions a Fourier Neural Operator through FiLM and a softmax mixture of spectral filter banks. The FNO takes the initial condition and a query time.
(click) Out comes the predicted field at time t.
(click) The key: after pretraining the FNO is frozen. On a new PDE, a handful of simulations only update our belief about z.
Pretraining: a latent table with one z per training equation, the encoder trained to predict it (NLL), rehearsal splits that hold out whole structures.`,
} satisfies SlideDef

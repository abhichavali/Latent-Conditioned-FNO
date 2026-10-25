import type { SlideDef } from '../deck/SlideFrame'
import { Frame, build } from '../components/layouts'
import { Stat } from '../components/Stat'

const chips = (xs: string[]) => xs.map(x => <span key={x} className="chip">{x}</span>)

export default {
  title: '5,120 synthetic PDEs',
  tone: 'sea',
  section: 'Data',
  steps: 3,
  Component: () => (
    <Frame title="5,120 synthetic PDEs, built from physical building blocks">
      <div className="row" style={{ gap: 72 }}>
        <Stat value={5120} label="equations: dataset A (1,024) and B (4,096)" />
        <Stat value={197632} label="trajectories: 1,024 × 73 + 4,096 × 30 initial conditions" />
        <Stat value={128} unit="²" label="periodic grid, 15 frames per trajectory" />
        <Stat value={7} label="initial-condition pattern families" />
      </div>
      <div className="panels" style={{ marginTop: 64 }}>
        <div className="glass panel" {...build(1)}>
          <h3>Building blocks</h3>
          <p>Transport equations mix shallow-water pieces {chips(['flux', 'adv', 'grad', 'bathy'])}; excitable ones mix Aliev–Panfilov pieces {chips(['cubic', 'uv', 'gate'])}.</p>
        </div>
        <div className="glass panel" {...build(2)} data-tone="ember">
          <h3>Held out by construction</h3>
          <p>The full cores {chips(['flux+adv+grad', 'cubic+uv+gate'])} are never generated, so both demo equations are unseen.</p>
        </div>
        <div className="glass panel" {...build(3)}>
          <h3>Max-entropy initial conditions</h3>
          <p>Greedy log-det selection over spectral, amplitude and rate features keeps 18 of 384 candidate windows per equation.</p>
        </div>
      </div>
    </Frame>
  ),
  notes: `Pretraining data, from data_collection/README.md.
Every equation is a JSON spec: fields on a periodic square, one sympy right-hand side per field, solved with a generic spectral (ETDRK) solver.
Dataset A: 1,024 equations × 73 initial conditions. Dataset B: 4,096 × 30 (18 selected plus 12 from a supplement pass). That's ~198k trajectory windows before de-duplication and non-finite filtering.
(click) Equations are sampled from families near our targets: transport and waves near shallow water, excitable media near Aliev–Panfilov, plus pattern formation and free composition. Each building block appears alone and in pairs.
(click) The full SWE and AP cores are never emitted, so the demos are genuinely out of distribution.
(click) Initial conditions are chosen for diversity: greedy log-det (max entropy) over spectral, amplitude and rate features, capped at 25% per IC family.`,
} satisfies SlideDef

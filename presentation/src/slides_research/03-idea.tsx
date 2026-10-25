import type { SlideDef } from '../deck/SlideFrame'
import { Quote, build } from '../components/layouts'
import { Tex } from '../components/Tex'

const pillars = [
  ['Pretrained operator', 'One FNO trained on 5,120 synthetic PDEs, then frozen. A 32-number latent code z selects the physics.'],
  ['Equation-graph prior', "A graph encoder reads the equation's symbolic form and predicts a distribution over z before any simulation."],
  ['Active few-shot learning', 'Spend a tiny simulation budget where plausible values of z disagree most, then update the posterior.'],
]

export default {
  title: "Don't retrain the operator. Infer the physics.",
  tone: 'latent',
  section: 'Idea',
  steps: 4,
  Component: () => (
    <div className="frame">
      <Quote big step={0}>Don’t retrain the operator. Infer the physics.</Quote>
      <div className="panels" style={{ marginTop: 72 }}>
        {pillars.map(([h, p], i) => (
          <div key={h} className="glass panel" {...build(i + 1)}>
            <h3>{h}</h3>
            <p>{p}</p>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 56, fontSize: 40, display: 'flex', gap: 96, alignItems: 'baseline' }} {...build(4)}>
        <Tex>{'\\hat u(\\cdot, t) = \\mathcal{F}_\\theta\\big(u_0, t;\\, z\\big)'}</Tex>
        <Tex>{'z \\sim p_\\phi(z \\mid G) \\;\\longrightarrow\\; p(z \\mid G, \\mathcal{D}_{\\text{few}})'}</Tex>
      </div>
    </div>
  ),
  notes: `The pitch in one line: keep the operator, infer the physics.
(click) One FNO, pretrained on thousands of synthetic equations and then frozen. What changes between equations is a small latent code z.
(click) We read z's prior off the equation itself: a graph encoder over the symbolic right-hand side.
(click) Then we spend a tiny simulation budget, chosen where the prior's plausible z's disagree, to sharpen it into a posterior.
(click) Formally: the frozen operator maps an initial condition and a time to the field, given z; the encoder gives p(z | G), and a few simulations turn it into p(z | G, D).`,
} satisfies SlideDef

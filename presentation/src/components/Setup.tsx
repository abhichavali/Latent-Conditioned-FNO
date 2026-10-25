// An application's setup slide: what the scenario is, the question it answers, and the PDE behind it.
import type { ReactNode } from 'react'
import { build } from './layouts'
import { Tex } from './Tex'

/** The governing equations behind each demo: what the simulator solves and Simula learned to predict. */
export const EQUATIONS = {
  tsunami: { name: 'Shallow-water equations', eqs: [
    String.raw`\partial_t \eta + \nabla\!\cdot\!\big[(\eta - b)\,\mathbf{u}\big] = \nu\nabla^2\eta - s\,\eta`,
    String.raw`\partial_t \mathbf{u} + (\mathbf{u}\!\cdot\!\nabla)\,\mathbf{u} = -g\,\nabla\eta + \nu\nabla^2\mathbf{u} - s\,\mathbf{u}`],
    legend: 'η sea surface · u current · b seabed · s open-ocean boundary' },
  heart: { name: 'Aliev–Panfilov cardiac model', eqs: [
    String.raw`\partial_t u = \nabla\!\cdot\!\big(D(\mathbf{x})\,\nabla u\big) + k\,u\,(1-u)(u-a) - u\,v`,
    String.raw`\partial_t v = \delta\,\nabla^2 v + \varepsilon(u,v)\,\big[-v - k\,u\,(u-a-1)\big]`],
    legend: 'u voltage · v recovery · D conductivity (low inside the scar)' },
  fire: { name: 'Wildfire spread with fuel', eqs: [
    String.raw`\partial_t u = \nabla^2 u + k\,u\,(1-u)(u-a) - \gamma\,u\,(1-f) - \mathbf{w}\!\cdot\!\nabla u`,
    String.raw`\partial_t f = -\beta\,u\,f`],
    legend: 'u heat · f fuel · w wind' },
}


export function Setup({ n, title, tone, points, kind }: {
  n: number; title: string; tone: 'sea' | 'ember'; points: ReactNode[]; kind: keyof typeof EQUATIONS
}) {
  const e = EQUATIONS[kind]
  return (
    <div className="frame setup">
      <span className={`kicker c-${tone}`} {...build(0, 'fade')}>Application {n}</span>
      <h2 className="slide-head" {...build(0, 'words')}>{title}</h2>
      <div className="setup-body">
        <ul className="setup-points">{points.map((p, i) => <li key={i} {...build(0, 'fade', 0.4 + i * 0.2)}>{p}</li>)}</ul>
        <div className="setup-eqs glass" {...build(1, 'fade')}>
          <span className="kicker">Governing equations · {e.name}</span>
          {e.eqs.map(q => <Tex key={q} block>{q}</Tex>)}
          <small>{e.legend}</small>
        </div>
      </div>
    </div>
  )
}

// Overlays shared by the hook and the demo: the solver-vs-Simula race card, posters, bank status.
import { useEffect, useRef, type ReactNode } from 'react'
import { Missing, ratio, type timing } from './bank'
import { Tex } from '../components/Tex'

type T = ReturnType<typeof timing>

/** The race, to scale but without seconds: the solver's bar fills over SLOW x its measured time, Simula finishes at SLOW x its own. run = restart key, 0 = idle. */
export const SLOW = 20
export function SolverCard({ t, run, who = 'Simula', className = '' }: { t: T; run: number; who?: string; className?: string }) {
  const clock = useRef<HTMLSpanElement>(null), bar = useRef<HTMLElement>(null), state = useRef<HTMLSpanElement>(null), sim = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    let id = 0
    const t0 = performance.now()
    const tick = () => {
      const s = run ? (performance.now() - t0) / 1000 / SLOW : 0, p = Math.min(s / t.solver, 1)
      if (!clock.current || !bar.current || !state.current || !sim.current) return // unmounting
      clock.current.textContent = !run ? '' : p < 1 ? `${Math.floor(p * 100)}%` : 'done'
      bar.current.style.transform = `scaleX(${p})`
      state.current.textContent = !run ? 'waiting' : p < 1 ? 'still computing…' : 'done'
      sim.current.textContent = !run ? '—' : s < t.simula ? 'computing…' : `✓ ${ratio(t)}× faster`
      if (run && p < 1) id = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(id)
  }, [run, t.solver, t.simula])
  return (
    <div className={`race glass ${className}`}>
      <div className="race-row"><b>Numerical solver</b><span ref={clock} className="clock" /></div>
      <div className="meter"><i ref={bar} /></div>
      <span ref={state} className="state" />
      <div className="race-row win"><b>{who}</b><span ref={sim} className="clock" /></div>
      <span className="ph">relative speed, to scale{t.placeholder ? ' · placeholder timing' : ''}</span>
    </div>
  )
}

/** Stand-in for a live 3D view in thumbnails, the presenter window and print. */
export const Poster = ({ label, tone = 'sea' }: { label: ReactNode; tone?: 'sea' | 'ember' }) => (
  <div className={`poster t-${tone}`}><span>{label}</span></div>
)

export function BankStatus({ kind, error }: { kind: 'tsunami' | 'heart' | 'fire'; error?: Error }) {
  if (!error) return <div className="bank-status"><span className="spinner" /> Loading scenarios…</div>
  return (
    <div className="bank-status">
      <strong>{error instanceof Missing ? 'No scenario bank yet' : 'Scenario bank error'}</strong>
      <p>{error.message}</p>
      {error instanceof Missing && <p>For rehearsal, build the mock bank: <code>npm run mock-bank</code>. For the talk, copy the real bank into <code>public/demo/{kind}/</code>.</p>}
    </div>
  )
}

export const MockBadge = ({ what }: { what: string }) => <span className="mock-badge" title={what}>Mock data · {what}</span>

/** The governing equations behind each demo: what the simulator solves and Simula learned to predict. */
const EQUATIONS = {
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

export function Equations({ kind }: { kind: keyof typeof EQUATIONS }) {
  const e = EQUATIONS[kind]
  return (
    <div className="eqs glass">
      <span className="kicker">Governing equations · {e.name}</span>
      {e.eqs.map(q => <Tex key={q} block>{q}</Tex>)}
      <small>{e.legend}</small>
    </div>
  )
}

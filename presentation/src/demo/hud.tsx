// Overlays shared by the hook and the demo: the solver-vs-Simula race card, posters, bank status.
import { useEffect, useRef, type ReactNode } from 'react'
import { Missing, ratio, type timing } from './bank'

type T = ReturnType<typeof timing>

/** Races are drawn to scale but stretched: the solver's side always takes RACE seconds on screen, Simula's its measured share of that. */
export const RACE = 8
export const slow = (t: { solver: number }) => RACE / t.solver

/** The race card. run = restart key, 0 = idle. */
export function SolverCard({ t, run, who = 'Simula', className = '' }: { t: T; run: number; who?: string; className?: string }) {
  const clock = useRef<HTMLSpanElement>(null), bar = useRef<HTMLElement>(null), state = useRef<HTMLSpanElement>(null), sim = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    let id = 0
    const t0 = performance.now()
    const tick = () => {
      const s = run ? (performance.now() - t0) / 1000 / slow(t) : 0, p = Math.min(s / t.solver, 1)
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


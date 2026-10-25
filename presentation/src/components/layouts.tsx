import { useRef, type CSSProperties, type ReactNode } from 'react'
import gsap from 'gsap'
import { useGSAP } from '@gsap/react'
import { useMode } from '../deck/SlideFrame'
import { reduced } from '../deck/motion'

/** Build attributes: `<p {...build(2)}>` appears at build step 2. anim names live in deck/motion.ts. */
export const build = (step?: number, anim?: string, delay?: number) =>
  step === undefined ? {} : { 'data-step': step, 'data-anim': anim, 'data-delay': delay }

/** Standard content slide: heading (+ optional lede) and a body. */
export function Frame({ title, lede, children }: { title: ReactNode; lede?: ReactNode; children?: ReactNode }) {
  return (
    <div className="frame">
      <header>
        <h2 {...build(0, 'words')}>{title}</h2>
        {lede && <p className="lede" {...build(0, 'fade', 0.3)}>{lede}</p>}
      </header>
      <div className="frame-body">{children}</div>
    </div>
  )
}

export function Title({ title, subtitle, meta }: { title: ReactNode; subtitle?: ReactNode; meta?: ReactNode[] }) {
  return (
    <div className="title-slide frame">
      <Modes />
      <h1 className="hero" {...build(0, 'words')}>{title}</h1>
      {subtitle && <p className="sub" {...build(0, 'fade', 0.6)}>{subtitle}</p>}
      {meta && <div className="meta" {...build(0, 'fade', 0.8)}>{meta.map((m, i) => <span key={i}>{m}</span>)}</div>}
    </div>
  )
}

// Fourier modes drifting along the bottom of the title slide: the FNO's native basis.
const MODES = [
  { n: 1, a: 92, y: 300, c: 'var(--sea)', w: 2.5, o: 0.55, t: 46 },
  { n: 2, a: 58, y: 330, c: 'var(--latent)', w: 2, o: 0.5, t: 34 },
  { n: 3, a: 40, y: 360, c: 'var(--sea)', w: 1.5, o: 0.35, t: 28 },
  { n: 5, a: 26, y: 390, c: 'var(--ember)', w: 1.5, o: 0.45, t: 22 },
  { n: 8, a: 14, y: 420, c: 'var(--ink)', w: 1, o: 0.25, t: 18 },
]
function Modes() {
  const ref = useRef<SVGSVGElement>(null), mode = useMode()
  useGSAP(() => {
    if (mode !== 'live' || reduced()) return
    ref.current!.querySelectorAll('path').forEach((p, i) =>
      gsap.to(p, { x: -1920 / MODES[i].n, duration: MODES[i].t, ease: 'none', repeat: -1 }))
  }, { scope: ref })
  return (
    <svg ref={ref} className="modes" viewBox="0 0 1920 520" aria-hidden>
      {MODES.map(({ n, a, y, c, w, o }, i) => {
        const lam = 1920 / n
        let d = ''
        for (let x = 0; x <= 1920 + lam; x += 6) {
          const k = (2 * Math.PI * x) / lam
          d += `${x ? 'L' : 'M'}${x} ${(y + a * Math.sin(k + i) + a * 0.25 * Math.sin(2 * k)).toFixed(1)}`
        }
        return <path key={n} d={d} stroke={c} strokeWidth={w} opacity={o} {...build(0, 'draw', 0.2 + i * 0.12)} />
      })}
    </svg>
  )
}

export function Section({ mark, title, lede }: { mark?: ReactNode; title: ReactNode; lede?: ReactNode }) {
  return (
    <div className="section-slide frame">
      {mark && <div className="mark" {...build(0, 'fade')}>{mark}</div>}
      <h2 {...build(0, 'words')}>{title}</h2>
      <div className="rule" {...build(0, 'wipe', 0.3)} />
      {lede && <p className="lede" {...build(0, 'fade', 0.5)}>{lede}</p>}
    </div>
  )
}

export function TwoCol({ left, right, split = '1fr 1fr', align = 'start', gap }: {
  left: ReactNode; right: ReactNode; split?: string; align?: CSSProperties['alignItems']; gap?: number
}) {
  return (
    <div className="cols" style={{ gridTemplateColumns: split, alignItems: align, gap }}>
      <div>{left}</div>
      <div>{right}</div>
    </div>
  )
}

export function Quote({ children, by, step, big }: { children: ReactNode; by?: ReactNode; step?: number; big?: boolean }) {
  return (
    <blockquote className={`quote${big ? ' big' : ''}`} {...build(step)}>
      <p>{children}</p>
      {by && <cite>{by}</cite>}
    </blockquote>
  )
}

/** List whose items build one per step, starting at `from` (null = all visible at once). */
export function Points({ items, from = 1 }: { items: ReactNode[]; from?: number | null }) {
  return (
    <ul className="points">
      {items.map((it, i) => <li key={i} {...build(from === null ? undefined : from + i)}>{it}</li>)}
    </ul>
  )
}

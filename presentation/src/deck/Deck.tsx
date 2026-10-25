import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import gsap from 'gsap'
import { useGSAP } from '@gsap/react'
import { Fit } from './Fit'
import { SlideFrame, type SlideDef } from './SlideFrame'
import { Presenter } from './Presenter'
import { advance, clamp, format, parse, progress, type Pos } from './nav'
import { reduced } from './motion'

const channel = new BroadcastChannel('pde-deck') // keeps the audience and presenter windows in step
const params = new URLSearchParams(location.search)
const isPresenter = params.has('presenter')
const isPrint = params.has('print') // every slide at its last build, one per page: print to PDF
const pad = (n: number) => String(n).padStart(2, '0')

export function Deck({ slides }: { slides: SlideDef[] }) {
  const steps = useMemo(() => slides.map(s => s.steps ?? 0), [slides])
  const [pos, setPos] = useState<Pos>(() => parse(location.hash, steps))
  const [overview, setOverview] = useState(false)
  const remote = useRef('')

  const go = useCallback((p: Pos) => setPos(clamp(steps, p.i, p.s)), [steps])
  const move = useCallback((d: 1 | -1) => setPos(p => advance(steps, p, d)), [steps])

  // URL hash (refresh keeps your place) + broadcast to the other window, unless the change came from it
  useEffect(() => {
    const key = format(pos)
    history.replaceState(null, '', key)
    if (key === remote.current) remote.current = ''
    else channel.postMessage(pos)
  }, [pos])

  useEffect(() => {
    const onMsg = (e: MessageEvent<Pos>) => { remote.current = format(e.data); go(e.data) }
    const onHash = () => go(parse(location.hash, steps))
    channel.addEventListener('message', onMsg)
    addEventListener('hashchange', onHash)
    return () => { channel.removeEventListener('message', onMsg); removeEventListener('hashchange', onHash) }
  }, [go, steps])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if ((e.target as Element).closest?.('input, textarea, select, [contenteditable]')) return
      const k = e.key
      if (k === 'ArrowRight' || k === 'ArrowDown' || k === 'PageDown' || (k === ' ' && !e.shiftKey)) move(1)
      else if (k === 'ArrowLeft' || k === 'ArrowUp' || k === 'PageUp' || k === ' ') move(-1)
      else if (k === 'Home') go({ i: 0, s: 0 })
      else if (k === 'End') go({ i: slides.length - 1, s: steps[slides.length - 1] })
      else if (isPresenter) return
      else if (k === 'o' || k === 'O') setOverview(o => !o)
      else if ((k === 'Escape' || k === 'Enter') && overview) setOverview(false)
      else if (k === 'f' || k === 'F') {
        if (document.fullscreenElement) document.exitFullscreen()
        else document.documentElement.requestFullscreen().catch(() => {})
      } else if (k === 'p' || k === 'P') {
        const u = new URL(location.href)
        u.searchParams.set('presenter', '')
        window.open(u, 'pde-presenter', 'popup,width=1400,height=860')
      } else return
      e.preventDefault()
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [move, go, overview, slides.length, steps])

  if (isPrint) return (
    <div className="print">
      {slides.map((s, i) => <div key={i} className="page" data-tone={s.tone ?? 'sea'}><div className="stage"><SlideFrame def={s} step={s.steps ?? 0} mode="thumb" /></div></div>)}
    </div>
  )
  if (isPresenter) return <Presenter slides={slides} steps={steps} pos={pos} go={go} />

  return (
    <>
      <Backdrop tone={slides[pos.i].tone ?? 'sea'} />
      <Stage slides={slides} pos={pos} move={move} />
      <div className="progress" style={{ transform: `scaleX(${progress(steps, pos)})` }} />
      {overview && <Overview slides={slides} current={pos.i} pick={i => { go({ i, s: 0 }); setOverview(false) }} />}
    </>
  )
}

/** The live stage: the current slide plus, during a transition, the one leaving. Click / tap / swipe to navigate. */
function Stage({ slides, pos, move }: { slides: SlideDef[]; pos: Pos; move: (d: 1 | -1) => void }) {
  const [view, setView] = useState({ cur: pos, dir: 0, out: null as null | (Pos & { dir: number }) })
  if (view.cur !== pos) { // derive the transition during render so the leaving slide is never remounted
    const changed = view.cur.i !== pos.i
    const dir = changed ? Math.sign(pos.i - view.cur.i) : view.dir
    setView({ cur: pos, dir, out: changed && !reduced() ? { ...view.cur, dir } : changed ? null : view.out })
  }
  const out = view.out && view.out.i !== pos.i ? view.out : null
  const down = useRef<{ x: number; y: number } | null>(null)
  const def = slides[pos.i]

  return (
    <Fit className="main">
      <div
        className="hit"
        onPointerDown={e => { if (e.button === 0) down.current = { x: e.clientX, y: e.clientY } }}
        onPointerUp={e => {
          const d = down.current
          down.current = null
          if (!d || (e.target as Element).closest('a, button, input, select, textarea, iframe, [data-no-nav]')) return
          if (getSelection()?.toString()) return
          const dx = e.clientX - d.x, dy = e.clientY - d.y
          if (Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy)) move(dx < 0 ? 1 : -1) // swipe
          else if (Math.hypot(dx, dy) < 10) move(e.clientX < innerWidth * 0.25 ? -1 : 1) // tap: left quarter goes back
        }}
      >
        {out && (
          <SlideFrame key={out.i} def={slides[out.i]} step={out.s} role="out" dir={out.dir}
            onOut={() => setView(v => (v.out?.i === out.i ? { ...v, out: null } : v))} />
        )}
        <SlideFrame key={pos.i} def={def} step={pos.s} role="in" dir={view.dir} />
      </div>
      {def.chrome !== false && (
        <footer className="chrome">
          <span>{def.section ?? 'PDE Foundation Model'}</span>
          <span className="num">{pad(pos.i + 1)} <i>/ {pad(slides.length)}</i></span>
        </footer>
      )}
    </Fit>
  )
}

/** Fixed full-viewport backdrop: accent glow that follows the slide's tone, grain, vignette. */
function Backdrop({ tone }: { tone: string }) {
  return (
    <div className="backdrop" data-tone={tone} aria-hidden>
      <div className="glow" />
      <div className="glow two" />
      <div className="grain" />
      <div className="vignette" />
    </div>
  )
}

function Overview({ slides, current, pick }: { slides: SlideDef[]; current: number; pick: (i: number) => void }) {
  const ref = useRef<HTMLDivElement>(null)
  useGSAP(() => {
    if (!reduced()) gsap.from('.thumb', { autoAlpha: 0, y: 30, scale: 0.96, stagger: 0.025, duration: 0.6 })
    ref.current?.querySelector('.thumb.on')?.scrollIntoView({ block: 'center' })
  }, { scope: ref })
  return (
    <div ref={ref} className="overview" role="dialog" aria-label="All slides">
      {slides.map((s, i) => (
        <button key={i} className={`thumb${i === current ? ' on' : ''}`} onClick={() => pick(i)}>
          <Fit><SlideFrame def={s} step={s.steps ?? 0} mode="thumb" /></Fit>
          <span><b>{pad(i + 1)}</b> {s.title}</span>
        </button>
      ))}
    </div>
  )
}

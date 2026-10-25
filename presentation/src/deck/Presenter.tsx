import { useEffect, useState } from 'react'
import { Fit } from './Fit'
import { SlideFrame, type SlideDef } from './SlideFrame'
import { advance, type Pos } from './nav'

const mmss = (ms: number) => {
  const s = Math.floor(ms / 1000)
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

/** Speaker view (second window, opened with `p`): current + next position, notes, timer. Keys navigate both windows. */
export function Presenter({ slides, steps, pos, go }: { slides: SlideDef[]; steps: number[]; pos: Pos; go: (p: Pos) => void }) {
  const [t0, setT0] = useState(() => Date.now())
  const [paused, setPaused] = useState<number | null>(null) // elapsed ms while paused
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), 250); return () => clearInterval(id) }, [])
  useEffect(() => { if (!document.title.startsWith('Presenter')) document.title = `Presenter · ${document.title}` }, [])

  const elapsed = paused ?? now - t0
  const next = advance(steps, pos, 1)
  const end = next.i === pos.i && next.s === pos.s
  const def = slides[pos.i]

  return (
    <div className="presenter">
      <section className="p-current">
        <Fit><SlideFrame key={pos.i} def={def} step={pos.s} mode="thumb" /></Fit>
      </section>
      <aside className="p-side">
        <div className="p-next">
          <h3>{end ? 'End of deck' : next.i === pos.i ? `Next: build ${next.s} of ${steps[pos.i]}` : `Next: ${slides[next.i].title}`}</h3>
          {!end && <Fit><SlideFrame key={next.i} def={slides[next.i]} step={next.s} mode="thumb" /></Fit>}
        </div>
        <div className="p-bar">
          <span className="p-timer" data-paused={paused !== null}>{mmss(elapsed)}</span>
          <button onClick={() => (paused === null ? setPaused(now - t0) : (setT0(Date.now() - paused), setPaused(null)))}>
            {paused === null ? 'Pause' : 'Resume'}
          </button>
          <button onClick={() => { setT0(Date.now()); setPaused(paused === null ? null : 0) }}>Reset</button>
          <span className="p-clock">{new Date(now).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
      </aside>
      <section className="p-notes">
        <h3>
          {pos.i + 1} / {slides.length} · {def.title}
          {steps[pos.i] > 0 && <small> · build {pos.s} / {steps[pos.i]}</small>}
        </h3>
        <div className="notes">{def.notes ?? 'No notes for this slide.'}</div>
        <nav>
          <button onClick={() => go(advance(steps, pos, -1))}>Back</button>
          <button onClick={() => go(next)}>Next</button>
        </nav>
      </section>
    </div>
  )
}

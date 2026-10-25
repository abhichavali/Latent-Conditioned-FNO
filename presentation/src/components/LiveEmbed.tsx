import { useEffect, useState } from 'react'
import { useMode } from '../deck/SlideFrame'
import { build } from './layouts'

// Where the repo's viz/ pages live. Override with ?viz=<url> or VITE_VIZ_BASE.
// dev: `python3 -m http.server 8000` at the repo root; build: dist/ served from the repo root, so ../../viz/.
export const VIZ_BASE: string = new URLSearchParams(location.search).get('viz')
  ?? import.meta.env.VITE_VIZ_BASE
  ?? (import.meta.env.DEV ? 'http://localhost:8000/viz/' : '../../viz/')

type Variant = { label: string; src: string } // src relative to VIZ_BASE, e.g. 'tsunami/?data=…'

/**
 * iframes a viz page. Only on the live stage (thumbnails get a card). The iframe ignores the pointer until
 * "Interact" is pressed, so clicks and keys keep driving the deck. Shows how to serve it when unreachable.
 */
export function LiveEmbed({ title, variants, step, height = 640 }: { title: string; variants: Variant[]; step?: number; height?: number }) {
  const live = useMode() === 'live'
  const [v, setV] = useState(0)
  const [state, setState] = useState<'probe' | 'ok' | 'down'>('probe')
  const [tries, setTries] = useState(0)
  const [interact, setInteract] = useState(false)
  const src = new URL(variants[v].src, new URL(VIZ_BASE, location.href)).href

  useEffect(() => {
    if (!live) return
    let alive = true
    setState('probe')
    // no-cors: a cross-origin server answers with an opaque response, a dead one rejects
    fetch(src, { mode: 'no-cors', cache: 'no-store' })
      .then(r => alive && setState(r.type === 'opaque' || r.ok ? 'ok' : 'down'), () => alive && setState('down'))
    return () => { alive = false }
  }, [src, live, tries])

  return (
    <div className={`embed glass${interact ? ' interactive' : ''}`} style={{ height }} data-no-nav {...build(step)}>
      <div className="bar">
        <b>{title}</b>
        {variants.length > 1 && variants.map((x, i) => (
          <button key={x.label} aria-pressed={i === v} onClick={() => setV(i)}>{x.label}</button>
        ))}
        {live && state === 'ok' && (
          <button aria-pressed={interact} onClick={() => setInteract(i => !i)}>{interact ? 'Done' : 'Interact'}</button>
        )}
      </div>
      {live && state === 'ok' ? (
        <iframe key={src} src={src} title={title} allow="fullscreen" />
      ) : (
        <div className="placeholder">
          {!live ? <strong>Live three.js view</strong> : state === 'probe' ? <strong>Connecting…</strong> : (
            <>
              <strong>Live view isn't reachable</strong>
              <p>
                The deck looks for the viz pages at <code>{new URL(VIZ_BASE, location.href).href}</code>. From the repo root
                run <code>python3 -m http.server 8000</code>, then open <code>localhost:8000/presentation/dist/</code>
                (or keep <code>npm run dev</code> running alongside it).
              </p>
              <div><button className="btn" onClick={() => setTries(t => t + 1)}>Try again</button></div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

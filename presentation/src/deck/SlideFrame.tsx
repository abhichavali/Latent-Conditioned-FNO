import { createContext, useContext, useId, useRef, type ComponentType } from 'react'
import gsap from 'gsap'
import { useGSAP } from '@gsap/react'
import { buildTimeline, reduced, EASE, type Builder } from './motion'

export type Tone = 'sea' | 'ember' | 'latent'

/** One slide = one module in src/slides/ whose default export is a SlideDef. */
export type SlideDef = {
  title: string // overview, presenter, footer
  Component: ComponentType
  steps?: number // last build step; 0 (default) = no builds
  notes?: string // speaker notes, newlines kept
  tone?: Tone // accent colour for the slide and the backdrop glow
  section?: string // small footer label
  chrome?: boolean // false hides the footer
  build?: Builder // bespoke tweens per step, on top of data-step builds
}

/** 'live' = the slide on stage (animated); 'thumb' = overview / presenter preview (seeks, no iframes, no loops). */
export type Mode = 'live' | 'thumb'
const ModeCtx = createContext<Mode>('live')
const BuildCtx = createContext<Map<string, Builder> | null>(null)
export const useMode = () => useContext(ModeCtx)

/** Register per-step tweens from inside a component; runs once when the slide's timeline is built. */
export function useBuild(fn: Builder) {
  const reg = useContext(BuildCtx), id = useId()
  reg?.set(id, fn) // idempotent, so safe during render
}

type Props = {
  def: SlideDef
  step: number
  mode?: Mode
  role?: 'in' | 'out' // entering/current vs leaving (live only)
  dir?: number // -1 back, 1 forward, 0 jump/first load
  onOut?: () => void
}

export function SlideFrame({ def, step, mode = 'live', role = 'in', dir = 0, onOut }: Props) {
  const root = useRef<HTMLDivElement>(null)
  const tl = useRef<gsap.core.Timeline>(null)
  const playing = useRef<gsap.core.Tween>(null)
  const shown = useRef(-1)
  const builders = useRef(new Map<string, Builder>())

  useGSAP(() => {
    const t = gsap.timeline({ paused: true })
    buildTimeline(t, root.current!, def.steps ?? 0, [...builders.current.values(), ...(def.build ? [def.build] : [])])
    tl.current = t
  }, { scope: root })

  // build steps: tween the timeline's playhead to the step's end label; going back reverses it
  useGSAP(() => {
    const t = tl.current!, label = `s${step}`
    playing.current?.kill()
    const first = shown.current < 0
    shown.current = step
    if (mode !== 'live' || reduced() || role === 'out' || (first && step > 0)) { t.seek(label, false); return }
    if (first) t.seek(0)
    const back = t.labels[label] < t.time()
    playing.current = t.tweenTo(label, { duration: Math.abs(t.labels[label] - t.time()) * (back ? 0.5 : 1), ease: 'none' })
  }, { dependencies: [step], scope: root })

  // slide transition
  useGSAP(() => {
    if (mode !== 'live') return
    const el = root.current!
    if (reduced()) { gsap.set(el, { autoAlpha: role === 'in' ? 1 : 0 }); if (role === 'out') onOut?.(); return }
    if (role === 'out')
      gsap.to(el, { autoAlpha: 0, x: -dir * 140, scale: 0.97, duration: 0.55, ease: 'power2.in', overwrite: true, onComplete: onOut })
    else
      gsap.fromTo(el, { autoAlpha: 0, x: dir * 140, scale: dir ? 1 : 0.985 },
        { autoAlpha: 1, x: 0, scale: 1, duration: 0.9, ease: EASE, overwrite: true, delay: dir ? 0.12 : 0 })
  }, { dependencies: [role], scope: root })

  const C = def.Component
  return (
    <div ref={root} className="slide" data-tone={def.tone ?? 'sea'} aria-hidden={role === 'out' || mode === 'thumb'}>
      <ModeCtx.Provider value={mode}>
        <BuildCtx.Provider value={builders.current}>
          <C />
        </BuildCtx.Provider>
      </ModeCtx.Provider>
    </div>
  )
}

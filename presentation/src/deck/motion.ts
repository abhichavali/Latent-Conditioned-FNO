import gsap from 'gsap'
import { useGSAP } from '@gsap/react'
import { SplitText } from 'gsap/SplitText'
import { DrawSVGPlugin } from 'gsap/DrawSVGPlugin'
import { MorphSVGPlugin } from 'gsap/MorphSVGPlugin'

gsap.registerPlugin(useGSAP, SplitText, DrawSVGPlugin, MorphSVGPlugin)

// Motion tokens. Mirrored in theme.css (--ease, --dur) for CSS transitions.
export const EASE = 'expo.out'
export const EASE_IN_OUT = 'power3.inOut'
export const DUR = 0.8
export const STAGGER = 0.07
gsap.defaults({ ease: EASE, duration: DUR })

export const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches

type El = HTMLElement | SVGElement
type Tl = gsap.core.Timeline
/** Adds one step's extra tweens at position `at` (the step's start label). Called for every step k = 0..steps. */
export type Builder = (k: number, tl: Tl, at: string) => void

const fmt = (v: number, dec: number) =>
  v.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec })

/**
 * Declarative builds: any element with data-step="k" animates in at build step k using data-anim
 * (default "fade"). data-delay adds seconds. Add a new kind here and use it by name.
 */
export const ANIMS: Record<string, (tl: Tl, el: El, at: string) => void> = {
  fade: (tl, el, at) => tl.fromTo(el, { autoAlpha: 0, y: 28 }, { autoAlpha: 1, y: 0 }, at),
  appear: (tl, el, at) => tl.fromTo(el, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.5 }, at),
  pop: (tl, el, at) =>
    tl.fromTo(el, { autoAlpha: 0, scale: 0.7, transformOrigin: '50% 50%' },
      { autoAlpha: 1, scale: 1, ease: 'back.out(1.7)', duration: 0.6 }, at),
  draw: (tl, el, at) =>
    tl.fromTo(el, { drawSVG: '0%' }, { drawSVG: '100%', duration: 1.1, ease: EASE_IN_OUT }, at),
  wipe: (tl, el, at) =>
    tl.fromTo(el, { clipPath: 'inset(0% 100% 0% 0%)' }, { clipPath: 'inset(0% 0% 0% 0%)', duration: 1.2, ease: EASE_IN_OUT }, at),
  words: (tl, el, at) => {
    const split = SplitText.create(el, { type: 'words' }) // no mask: masks clip display-serif ascenders
    tl.fromTo(split.words, { yPercent: 55, autoAlpha: 0 }, { yPercent: 0, autoAlpha: 1, stagger: 0.06, duration: 1.1 }, at)
  },
  count: (tl, el, at) => {
    const to = Number(el.dataset.to), dec = Number(el.dataset.decimals ?? 0), o = { v: 0 }
    tl.fromTo(el, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.4 }, at)
    tl.fromTo(o, { v: 0 }, { v: to, duration: 1.8, ease: 'power3.out', onUpdate: () => { el.textContent = fmt(o.v, dec) } }, at)
  },
  dim: (tl, el, at) => tl.to(el, { opacity: 0.28, duration: 0.5 }, at),
  out: (tl, el, at) => tl.to(el, { autoAlpha: 0, duration: 0.4 }, at), // leaves at its step
}

/** One paused timeline per slide with labels b{k} (step k starts) and s{k} (step k done). */
export function buildTimeline(tl: Tl, root: Element, steps: number, builders: Builder[]) {
  for (let k = 0; k <= steps; k++) {
    const at = `b${k}`
    tl.addLabel(at, k === 0 ? 0.15 : tl.duration())
    root.querySelectorAll<El>(`[data-step="${k}"]`).forEach((el, n) => {
      const anim = ANIMS[el.dataset.anim ?? 'fade']
      if (import.meta.env.DEV && !anim) console.warn(`unknown data-anim "${el.dataset.anim}"`, el)
      ;(anim ?? ANIMS.fade)(tl, el, `${at}+=${n * STAGGER + Number(el.dataset.delay ?? 0)}`)
    })
    builders.forEach(b => b(k, tl, at))
    tl.addLabel(`s${k}`, tl.duration())
  }
  if (import.meta.env.DEV) {
    const max = Math.max(0, ...[...root.querySelectorAll<El>('[data-step]')].map(e => Number(e.dataset.step)))
    if (max > steps) console.warn(`slide uses data-step=${max} but declares steps: ${steps}`, root)
  }
}

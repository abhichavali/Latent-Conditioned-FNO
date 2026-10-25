import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import { useMode } from '../deck/SlideFrame'
import { reduced } from '../deck/motion'

/** Endless ambient tweens, scoped to the returned ref; only on the live stage and without reduced motion. */
export function useAmbient<T extends Element = SVGSVGElement>(fn: () => void) {
  const ref = useRef<T>(null), live = useMode() === 'live'
  useGSAP(() => { if (live && !reduced()) fn() }, { scope: ref })
  return ref
}

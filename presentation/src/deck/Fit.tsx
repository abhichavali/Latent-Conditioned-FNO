import { useLayoutEffect, useRef, type ReactNode } from 'react'

export const W = 1920, H = 1080

/** Scales a fixed 1920×1080 stage to fit (contain) its box. */
export function Fit({ children, className = '' }: { children: ReactNode; className?: string }) {
  const box = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const el = box.current!
    const fit = (w: number, h: number) => el.style.setProperty('--fit', String(Math.min(w / W, h / H)))
    const r = el.getBoundingClientRect()
    fit(r.width, r.height)
    const ro = new ResizeObserver(([e]) => fit(e.contentRect.width, e.contentRect.height))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return (
    <div ref={box} className={`fit ${className}`}>
      <div className="stage">{children}</div>
    </div>
  )
}

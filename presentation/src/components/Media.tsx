import type { ReactNode } from 'react'
import { useMode } from '../deck/SlideFrame'
import { build } from './layouts'

type Common = { caption?: ReactNode; step?: number; framed?: boolean; className?: string }

/** Image, e.g. `import fig from '../assets/fig.png'` then `<Img src={fig} alt="…" />`. framed = white mat for plots. */
export function Img({ src, alt, caption, step, framed, className = '' }: Common & { src: string; alt: string }) {
  return (
    <figure className={`media${framed ? ' framed' : ''} ${className}`} {...build(step)}>
      <img src={src} alt={alt} />
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  )
}

/** Muted looping video; only plays on the live stage (thumbnails show the poster). */
export function Video({ src, poster, caption, step, framed, className = '', loop = true }: Common & { src: string; poster?: string; loop?: boolean }) {
  const live = useMode() === 'live'
  return (
    <figure className={`media${framed ? ' framed' : ''} ${className}`} {...build(step)}>
      <video src={src} poster={poster} muted playsInline loop={loop} autoPlay={live} preload={live ? 'auto' : 'none'} />
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  )
}

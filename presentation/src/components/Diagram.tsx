import type { ReactNode } from 'react'
import type { Tone } from '../deck/SlideFrame'
import { build } from './layouts'

type NodeTone = Tone | 'ink'
export type DNode = {
  id: string
  x: number // centre, in diagram px
  y: number
  label: ReactNode
  sub?: ReactNode
  w?: number
  h?: number
  step?: number // build step it pops in at (omit = always visible)
  tone?: NodeTone
  shape?: 'box' | 'circle' | 'pill'
  className?: string // e.g. 'ghost' (transparent, dashed)
  size?: number // label font size
}
export type DEdge = {
  from: string
  to: string
  step?: number // default: when both ends are visible
  bend?: number // control-point offset as a fraction of the chord; sign picks the side
  label?: ReactNode
  dashed?: boolean
  tone?: NodeTone
  delay?: number
}

const dims = (n: DNode) => {
  const w = n.w ?? (n.shape === 'circle' ? 72 : 220)
  return { w, h: n.shape === 'circle' ? w : n.h ?? 96 }
}

// boundary point of n in direction (dx, dy), pushed out by gap
function clip(n: DNode, dx: number, dy: number, gap: number): [number, number] {
  const len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len, { w, h } = dims(n)
  const t = n.shape === 'circle' ? w / 2 : Math.min(Math.abs(ux) > 1e-6 ? w / 2 / Math.abs(ux) : 1e9, Math.abs(uy) > 1e-6 ? h / 2 / Math.abs(uy) : 1e9)
  return [n.x + ux * (t + gap), n.y + uy * (t + gap)]
}

/**
 * Nodes (HTML, so labels can hold KaTeX) and directed edges (SVG) that build in step by step:
 * nodes pop, edges draw (DrawSVG), arrowheads and labels follow. `children` render into the SVG underlay.
 */
export function Diagram({ nodes, edges, width, height, children, className = '' }: {
  nodes: DNode[]; edges: DEdge[]; width: number; height: number; children?: ReactNode; className?: string
}) {
  const byId = new Map(nodes.map(n => [n.id, n]))
  const geo = edges.map(e => {
    const a = byId.get(e.from), b = byId.get(e.to)
    if (!a || !b) throw new Error(`Diagram edge ${e.from} → ${e.to}: unknown node`)
    const L = Math.hypot(b.x - a.x, b.y - a.y) || 1, bend = e.bend ?? 0
    const cx = (a.x + b.x) / 2 - ((b.y - a.y) / L) * bend * L, cy = (a.y + b.y) / 2 + ((b.x - a.x) / L) * bend * L
    const [sx, sy] = clip(a, cx - a.x, cy - a.y, 6)
    const [tx, ty] = clip(b, cx - b.x, cy - b.y, 8)
    const ang = Math.atan2(ty - cy, tx - cx)
    const ex = tx - Math.cos(ang) * 10, ey = ty - Math.sin(ang) * 10 // line stops under the arrowhead
    const steps = [a.step, b.step].filter((s): s is number => s !== undefined)
    const step = e.step ?? (steps.length ? Math.max(...steps) : undefined)
    return {
      e, step, tx, ty, deg: (ang * 180) / Math.PI,
      d: bend ? `M${sx} ${sy} Q${cx} ${cy} ${ex} ${ey}` : `M${sx} ${sy} L${ex} ${ey}`,
      mx: 0.25 * sx + 0.5 * cx + 0.25 * ex, my: 0.25 * sy + 0.5 * cy + 0.25 * ey,
      delay: e.delay ?? 0.25,
    }
  })

  return (
    <div className={`diagram ${className}`} style={{ width, height }}>
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} aria-hidden>
        {children}
        {geo.map(({ e, step, d, tx, ty, deg, delay }, i) => (
          <g key={i} className={`dedge t-${e.tone ?? 'ink'}`}>
            <path d={d} className={e.dashed ? 'dashed' : undefined} {...build(step, e.dashed ? 'appear' : 'draw', delay)} />
            <g transform={`translate(${tx} ${ty}) rotate(${deg})`}>
              <path className="tip" d="M0 0 L-17 -8 L-12 0 L-17 8 Z" {...build(step, 'appear', delay + 0.8)} />
            </g>
          </g>
        ))}
      </svg>
      {geo.map(({ e, step, mx, my, delay }, i) =>
        e.label ? <div key={i} className="dlabel" style={{ left: mx, top: my }} {...build(step, 'appear', delay + 0.6)}>{e.label}</div> : null)}
      {nodes.map(n => {
        const { w, h } = dims(n)
        return (
          <div key={n.id} className={`dnode ${n.shape ?? 'box'} t-${n.tone ?? 'ink'} ${n.className ?? ''}`}
            style={{ left: n.x - w / 2, top: n.y - h / 2, width: w, height: h, fontSize: n.size }} {...build(n.step, 'pop')}>
            <div>{n.label}</div>
            {n.sub && <div className="sub">{n.sub}</div>}
          </div>
        )
      })}
    </div>
  )
}

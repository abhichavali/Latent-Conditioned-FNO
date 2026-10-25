import { useId, type ReactNode } from 'react'
import { build } from './layouts'

export type Point = { x: number; y: number; err?: number }
export type Series = { name: string; points: Point[]; color?: string; dashed?: boolean; step?: number }

type Props = {
  series: Series[]
  width: number
  height: number
  logX?: boolean
  logY?: boolean
  xDomain?: [number, number]
  yDomain?: [number, number]
  xTicks?: number[]
  yTicks?: number[]
  xLabel?: string
  yLabel?: string
  title?: ReactNode
  axisStep?: number // when the axes appear (default: always)
  band?: { from: number; to: number; label: string; step?: number } // hatched x-range, e.g. pending results
  legend?: boolean
}

const M = { l: 92, r: 24, t: 20, b: 72 }

function niceTicks(a: number, b: number, n = 5) {
  let step = 10 ** Math.floor(Math.log10((b - a) / n))
  const err = (b - a) / n / step
  step *= err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1
  const out: number[] = []
  for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(+v.toPrecision(12))
  return out
}
const decades = (a: number, b: number) => {
  const out: number[] = []
  for (let e = Math.ceil(Math.log10(a) - 1e-9); 10 ** e <= b * (1 + 1e-9); e++) out.push(10 ** e)
  return out
}
const minors = (a: number, b: number) => {
  const out: number[] = []
  for (let e = Math.floor(Math.log10(a)); 10 ** e <= b; e++) for (let k = 2; k < 10; k++) if (k * 10 ** e >= a && k * 10 ** e <= b) out.push(k * 10 ** e)
  return out
}
const label = (v: number) => (Math.abs(v) >= 1000 ? v.toLocaleString('en-US') : String(+v.toPrecision(3)))

/** Line chart from data (log or linear axes, error bars, legend). Each series draws in at its build step. */
export function Chart({ series, width, height, logX, logY, xDomain, yDomain, xTicks, yTicks, xLabel, yLabel, title, axisStep, band, legend = true }: Props) {
  const hatch = useId().replace(/:/g, '')
  const xs = series.flatMap(s => s.points.map(p => p.x)), ys = series.flatMap(s => s.points.flatMap(p => [p.y - (p.err ?? 0), p.y + (p.err ?? 0)]))
  const [x0, x1] = xDomain ?? [Math.min(...xs), Math.max(...xs)]
  const [y0, y1] = yDomain ?? (logY ? [Math.min(...ys) * 0.8, Math.max(...ys) * 1.25] : [Math.min(0, ...ys), Math.max(...ys) * 1.1])
  const tx = logX ? Math.log10 : (v: number) => v, ty = logY ? Math.log10 : (v: number) => v
  const iw = width - M.l - M.r, ih = height - M.t - M.b
  const sx = (v: number) => M.l + ((tx(v) - tx(x0)) / (tx(x1) - tx(x0))) * iw
  const sy = (v: number) => M.t + ih - ((ty(Math.max(v, y0)) - ty(y0)) / (ty(y1) - ty(y0))) * ih
  const xt = xTicks ?? (logX ? decades(x0, x1) : niceTicks(x0, x1))
  const yt = yTicks ?? (logY ? decades(y0, y1) : niceTicks(y0, y1))

  return (
    <div className="chart" style={{ width }}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={typeof title === 'string' ? title : 'chart'}>
        <defs>
          <pattern id={hatch} width="14" height="14" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="14" stroke="rgba(255,122,69,0.35)" strokeWidth="3" />
          </pattern>
        </defs>
        <g {...build(axisStep, 'appear')}>
          <g className="grid">
            {logY && minors(y0, y1).map(v => <line key={`m${v}`} className="minor" x1={M.l} x2={M.l + iw} y1={sy(v)} y2={sy(v)} />)}
            {yt.map(v => <line key={v} x1={M.l} x2={M.l + iw} y1={sy(v)} y2={sy(v)} />)}
            {xt.map(v => <line key={v} x1={sx(v)} x2={sx(v)} y1={M.t} y2={M.t + ih} />)}
          </g>
          <g className="axis">
            <line x1={M.l} x2={M.l + iw} y1={M.t + ih} y2={M.t + ih} />
            <line x1={M.l} x2={M.l} y1={M.t} y2={M.t + ih} />
          </g>
          {xt.map(v => <text key={v} className="tick" x={sx(v)} y={M.t + ih + 30} textAnchor="middle">{label(v)}</text>)}
          {yt.map(v => <text key={v} className="tick" x={M.l - 14} y={sy(v) + 6} textAnchor="end">{label(v)}</text>)}
          {xLabel && <text className="axis-label" x={M.l + iw / 2} y={height - 8} textAnchor="middle">{xLabel}</text>}
          {yLabel && <text className="axis-label" transform={`translate(20 ${M.t + ih / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>}
        </g>
        {band && (
          <g {...build(band.step, 'appear')}>
            <rect x={sx(band.from)} y={M.t} width={sx(band.to) - sx(band.from)} height={ih} fill={`url(#${hatch})`} stroke="rgba(255,122,69,0.6)" strokeDasharray="6 6" />
            <text className="band-label" x={(sx(band.from) + sx(band.to)) / 2} y={M.t + 40} textAnchor="middle">{band.label}</text>
          </g>
        )}
        {series.map(s => {
          const c = s.color ?? 'var(--accent)'
          return (
            <g key={s.name} className="series">
              <path d={s.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x)} ${sy(p.y)}`).join('')} stroke={c}
                strokeDasharray={s.dashed ? '10 10' : undefined} {...build(s.step, s.dashed ? 'appear' : 'draw')} />
              <g {...build(s.step, 'appear', 0.6)}>
                {s.points.map((p, i) => p.err ? <line key={`e${i}`} x1={sx(p.x)} x2={sx(p.x)} y1={sy(p.y - p.err)} y2={sy(p.y + p.err)} stroke={c} /> : null)}
                {s.points.map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={5.5} stroke={c} />)}
              </g>
            </g>
          )
        })}
      </svg>
      {title && <div className="micro" style={{ position: 'absolute', top: 4, left: M.l + 20, color: 'var(--ink)', fontSize: 24 }}>{title}</div>}
      {legend && (
        <ul className="legend" style={{ marginLeft: M.l }}>
          {series.map(s => (
            <li key={s.name} {...build(s.step)}><i className={s.dashed ? 'dashed' : undefined} style={{ borderColor: s.color ?? 'var(--accent)' }} />{s.name}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

import type { ReactNode } from 'react'
import { build } from './layouts'

const fmt = (v: number, dec: number) => v.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec })

/** Big number that counts up when its step arrives. value={null} renders a pending dash (no invented numbers). */
export function Stat({ value, label, decimals = 0, prefix, unit, step = 0 }: {
  value: number | null; label: ReactNode; decimals?: number; prefix?: ReactNode; unit?: ReactNode; step?: number
}) {
  return (
    <div className={`stat${value === null ? ' pending' : ''}`} {...build(step)}>
      <div className="stat-num">
        {prefix}
        {value === null ? '—' : <span {...build(step, 'count')} data-to={value} data-decimals={decimals}>{fmt(value, decimals)}</span>}
        {unit && <span className="unit">{unit}</span>}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

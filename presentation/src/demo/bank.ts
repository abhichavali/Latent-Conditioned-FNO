// The demo's data: the scenario bank (public/demo/, fetched at runtime) and the measured numbers (src/data/results.json).
// Schema: README.md, "Scenario bank".
import { useEffect, useState } from 'react'
import { parseNpy } from '@viz/common.js'
import raw from '../data/results.json'

// ---------- results.json ----------
type Timing = { simula_seconds: number | null; solver_seconds: number | null }
type Point = { sims: number; error: number }
export type Results = {
  placeholder: boolean
  fewer_simulations: number | null
  timing: { tsunami: Timing; heart: Timing; fire: Timing; batch: Timing & { scenarios: number }; sequential: Timing & { scenarios: number } }
  rehearsal: Record<Kind, { simula_seconds: number; solver_seconds: number }>
  chart: { title: string; x_label: string; y_label: string; series: { id: string; label: string; points: Point[] }[] }
}
export const results = raw as Results
type Kind = 'tsunami' | 'heart' | 'fire' | 'batch' | 'sequential'

/** Measured timings, or the rehearsal durations (placeholder = true) while they're null. */
export function timing(kind: Kind) {
  const m = results.timing[kind], r = results.rehearsal[kind]
  return {
    simula: m.simula_seconds ?? r.simula_seconds,
    solver: m.solver_seconds ?? r.solver_seconds,
    placeholder: m.simula_seconds === null || m.solver_seconds === null,
  }
}

/** solver time / Simula time, one decimal: the measured multiple shown on screen */
export const ratio = (t: { simula: number; solver: number }) => (t.solver / t.simula).toFixed(1)

export const fmtDuration = (s: number) =>
  s < 1 ? `${+s.toPrecision(2)} s` : s < 120 ? `${+s.toFixed(1)} s` : s < 7200 ? `${+(s / 60).toFixed(1)} min` : `${+(s / 3600).toFixed(1)} h`
export const fmtClock = (s: number) =>
  `${String(Math.floor(s / 60)).padStart(2, '0')}:${(s % 60).toFixed(1).padStart(4, '0')}`

// ---------- scenario bank ----------
type XY = [number, number] // grid fractions: x = column / (nx - 1), y = row / (ny - 1)
export type TsunamiScenario = { id: string; x: number; y: number; frames: string; solver?: string }
export type TsunamiBank = {
  version: 1; mock?: boolean; note?: string
  grid: { nx: number; ny: number; lx: number; ly: number }
  bed: string; dt: number; time_unit?: string
  view?: { vexag?: number; wexag?: number }
  default?: string
  scenarios: TsunamiScenario[]
  risk?: { frames: string; count: number; unit?: string }
}
export type Ablation = { id: string; label?: string; line: [XY, XY]; frames: string; stops_arrhythmia: boolean } // fire: a firebreak, true = town safe
export type Scar = { id: string; x: number; y: number; r: number; frames: string; ablations: Ablation[] }
export type HeartBank = {
  version: 1; mock?: boolean; note?: string
  grid: { nx: number; ny: number; lx?: number; ly?: number }
  geometry?: 'sheet' | 'ventricle'; dt: number; time_unit?: string; range?: [number, number]
  asset?: { x: number; y: number; r: number; label: string } // fire: the town
  scars: Scar[]
}
type Banks = { tsunami: TsunamiBank; heart: HeartBank; fire: HeartBank } // fire reuses the heart schema: scars = ignitions, ablations = firebreaks

const ROOT = `${import.meta.env.BASE_URL}demo/`
const cache = new Map<string, Promise<unknown>>()
function once<T>(key: string, load: () => Promise<T>): Promise<T> {
  if (!cache.has(key)) cache.set(key, load().catch(e => { cache.delete(key); throw e })) // failures retry next time
  return cache.get(key) as Promise<T>
}

export class Missing extends Error {}

export function loadBank<K extends keyof Banks>(kind: K): Promise<Banks[K]> {
  return once(kind, async () => {
    const r = await fetch(`${ROOT}${kind}/index.json`, { cache: 'no-cache' })
    // a dev server answers a missing file with index.html, so a parse failure also means "not there"
    const bank = r.ok ? await r.json().catch(() => null) : null
    if (!bank) throw new Missing(`no scenario bank at demo/${kind}/index.json`)
    const list = kind === 'tsunami' ? bank.scenarios : bank.scars
    if (!bank.grid?.nx || !bank.grid?.ny || !Array.isArray(list) || !list.length)
      throw new Error(`demo/${kind}/index.json: needs grid.nx, grid.ny and a non-empty ${kind === 'tsunami' ? 'scenarios' : 'scars'} list`)
    return bank
  })
}

/** One .npy from a bank, split into frames. 2D arrays come back as a single frame. Checked against the grid. */
export function loadFrames(kind: keyof Banks, file: string, grid: { nx: number; ny: number }): Promise<Float32Array[]> {
  return once(`${kind}/${file}`, async () => {
    const r = await fetch(`${ROOT}${kind}/${file}`)
    if (!r.ok) throw new Error(`demo/${kind}/${file}: HTTP ${r.status}`)
    const { data, shape } = parseNpy(await r.arrayBuffer())
    const [ny, nx] = shape.slice(-2), n = nx * ny
    if (nx !== grid.nx || ny !== grid.ny) throw new Error(`demo/${kind}/${file}: frames are ${ny}×${nx}, grid says ${grid.ny}×${grid.nx}`)
    return Array.from({ length: data.length / n }, (_, t) => data.subarray(t * n, (t + 1) * n))
  })
}

export function nearest<T extends { x: number; y: number }>(items: T[], x: number, y: number): T {
  let best = items[0], d = Infinity
  for (const it of items) { const e = Math.hypot(it.x - x, it.y - y); if (e < d) { d = e; best = it } }
  return best
}

/** Promise -> state, re-run when deps change. */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[]): { value?: T; error?: Error } {
  const [s, set] = useState<{ value?: T; error?: Error }>({})
  useEffect(() => {
    let alive = true
    set({})
    load().then(value => alive && set({ value }), error => alive && set({ error }))
    return () => { alive = false }
  }, deps) // eslint-disable-line react-hooks/exhaustive-deps
  return s
}

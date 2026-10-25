// Pure navigation math. A position is (slide index, build step); steps[i] is slide i's last build step.
export type Pos = { i: number; s: number }

export function advance(steps: number[], p: Pos, d: 1 | -1): Pos {
  if (d > 0) return p.s < steps[p.i] ? { i: p.i, s: p.s + 1 } : p.i < steps.length - 1 ? { i: p.i + 1, s: 0 } : p
  return p.s > 0 ? { i: p.i, s: p.s - 1 } : p.i > 0 ? { i: p.i - 1, s: steps[p.i - 1] } : p
}

export function clamp(steps: number[], i: number, s: number): Pos {
  const j = Math.min(Math.max(Math.trunc(i) || 0, 0), steps.length - 1)
  return { i: j, s: Math.min(Math.max(Math.trunc(s) || 0, 0), steps[j]) }
}

// #/3/2 = third slide (1-based), build step 2. #/3 = step 0.
export function parse(hash: string, steps: number[]): Pos {
  const m = hash.match(/^#\/(\d+)(?:\/(\d+))?/)
  return m ? clamp(steps, +m[1] - 1, +(m[2] ?? 0)) : { i: 0, s: 0 }
}

export const format = (p: Pos) => `#/${p.i + 1}` + (p.s ? `/${p.s}` : '')

// fraction of all positions passed, 0 at the first, 1 at the last
export function progress(steps: number[], p: Pos): number {
  let before = 0, all = 0
  steps.forEach((n, i) => { all += n + 1; if (i < p.i) before += n + 1 })
  return all > 1 ? (before + p.s) / (all - 1) : 1
}

// MOCK scenario bank for rehearsing the demo slide: `npm run mock-bank`.
// Derived from the baseline FNO rollouts in ../baseline_FNO/results/viz (solver truth vs a from-scratch FNO), NOT Simula.
// Every index.json it writes says "mock": true, and the deck shows a MOCK badge for it.
// The real bank (same format, see README.md "Scenario bank") replaces public/demo/{tsunami,heart}/.
import fs from 'node:fs'
import path from 'node:path'

const here = path.dirname(new URL(import.meta.url).pathname)
const SRC = path.resolve(here, '../../baseline_FNO/results/viz')
const OUT = path.resolve(here, '../public/demo')

function readNpy(file) {
  const buf = fs.readFileSync(file)
  const hlen = buf.readUInt16LE(8), header = buf.toString('latin1', 10, 10 + hlen)
  if (!/'descr':\s*'<f4'/.test(header) || /True/.test(header)) throw new Error(`${file}: expected C-order float32`)
  const shape = header.match(/'shape':\s*\(([^)]*)\)/)[1].split(',').filter(s => s.trim()).map(Number)
  const ab = buf.buffer.slice(buf.byteOffset + 10 + hlen, buf.byteOffset + buf.length)
  return { data: new Float32Array(ab), shape }
}

function writeNpy(file, data, shape) {
  const descr = data instanceof Float16Array ? '<f2' : '<f4'
  let h = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`
  h += ' '.repeat(63 - ((10 + h.length) % 64)) + '\n' // header padded to a multiple of 64 bytes
  const pre = Buffer.alloc(10)
  pre.write('\x93NUMPY', 0, 'latin1'); pre[6] = 1; pre[7] = 0; pre.writeUInt16LE(h.length, 8)
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, Buffer.concat([pre, Buffer.from(h, 'latin1'), Buffer.from(data.buffer, data.byteOffset, data.byteLength)]))
}

const frames = ({ data, shape: [T, ny, nx] }, ts) => ts.filter(t => t < T).map(t => data.subarray(t * ny * nx, (t + 1) * ny * nx))
const stack = fs => { const n = fs[0].length, out = new Float32Array(fs.length * n); fs.forEach((f, k) => out.set(f, k * n)); return out }
const range = (a, b, s) => Array.from({ length: Math.ceil((b - a) / s) }, (_, k) => a + k * s)
const roll = (f, nx, ny, dx, dy) => { // periodic shift by (dx, dy) cells
  const out = new Float32Array(f.length)
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) out[((j + dy + ny) % ny) * nx + ((i + dx + nx) % nx)] = f[j * nx + i]
  return out
}
const json = (file, o) => fs.writeFileSync(file, JSON.stringify(o, null, 1) + '\n')

// ---------- tsunami ----------
// Epicentres along the coast: the rollout shifted in y (the solver grid is periodic). The bed stays put, so the
// shifted wave is carried as a free surface eta = b + h and re-cut against the bed (dry land carries eta = 0).
function tsunami() {
  const dir = `${OUT}/tsunami`, b = readNpy(`${SRC}/swe_truth/b.npy`).data
  const [ny, nx] = readNpy(`${SRC}/swe_truth/b.npy`).shape
  const ts = range(0, 142, 2) // every other frame of the action (dt 2 s -> 4 s); the engine interpolates between them
  const truth = frames(readNpy(`${SRC}/swe_truth/h.npy`), ts), fno = frames(readNpy(`${SRC}/swe_fno/h.npy`), ts)
  let i0 = 0, j0 = 0, top = -Infinity // epicentre of the source rollout: highest initial surface
  for (let k = 0; k < nx * ny; k++) if (b[k] < 0 && truth[0][k] + b[k] > top) { top = truth[0][k] + b[k]; i0 = k % nx; j0 = Math.floor(k / nx) }
  const shift = (h, dy) => {
    const eta = h.map((v, k) => (v > 1e-3 ? v + b[k] : 0)), e = roll(eta, nx, ny, 0, dy)
    return e.map((v, k) => Math.max(v - b[k], 0))
  }
  fs.rmSync(dir, { recursive: true, force: true })
  writeNpy(`${dir}/b.npy`, b, [ny, nx])
  const scenarios = [], risk = [], acc = new Float32Array(nx * ny)
  for (const [n, dy] of [0, -20, -40, -60].entries()) {
    const id = `e${n}`, sim = fno.map(h => shift(h, dy))
    writeNpy(`${dir}/${id}/h.npy`, stack(sim), [sim.length, ny, nx])
    writeNpy(`${dir}/${id}/h_solver.npy`, stack(truth.map(h => shift(h, dy))), [sim.length, ny, nx])
    scenarios.push({ id, x: +(i0 / (nx - 1)).toFixed(4), y: +(((j0 + dy + ny) % ny) / (ny - 1)).toFixed(4), frames: `${id}/h.npy`, solver: `${id}/h_solver.npy` })
    for (const h of sim) for (let k = 0; k < h.length; k++) if (b[k] > 0 && h[k] > acc[k]) acc[k] = h[k]
    risk.push(acc.slice()) // cumulative max inundation after each scenario
  }
  writeNpy(`${dir}/risk_map.npy`, stack(risk), [risk.length, ny, nx])
  json(`${dir}/index.json`, {
    version: 1, mock: true,
    note: `MOCK: baseline FNO (swe_fno) rollout shifted along the coast; solver = shifted swe_truth; risk map = max over these ${scenarios.length} scenarios. Not Simula.`,
    grid: { nx, ny, lx: 4000, ly: 4000 }, bed: 'b.npy', dt: 4, time_unit: 's', view: { vexag: 4, wexag: 6 },
    default: 'e0', scenarios,
    risk: { frames: 'risk_map.npy', count: scenarios.length, unit: 'm' },
  })
  console.log(`tsunami: ${scenarios.length} scenarios × ${ts.length} frames, epicentre (${i0}, ${j0}) -> ${dir}`)
}

// ---------- heart ----------
// Scars: the rollout shifted (Aliev-Panfilov on a periodic grid is shift-invariant), with the scar disc held at rest.
// Ablations: the ablated line held at rest; the one that "stops" the arrhythmia fades to rest. Purely illustrative.
function heart() {
  const dir = `${OUT}/heart`, src = readNpy(`${SRC}/ap_fno/u.npy`), [, ny, nx] = src.shape
  const ts = range(0, 286, 2), base = frames(src, ts), T = base.length
  const at = (x, y) => [Math.round(x * (nx - 1)), Math.round(y * (ny - 1))]
  const disc = (f, x, y, r) => { const [ci, cj] = at(x, y), R = r * (nx - 1); for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (Math.hypot(i - ci, j - cj) <= R) f[j * nx + i] = 0 }
  const line = (f, [[x0, y0], [x1, y1]]) => {
    const [a, c] = at(x0, y0), [b2, d] = at(x1, y1), n = Math.ceil(Math.hypot(b2 - a, d - c) * 2)
    for (let s = 0; s <= n; s++) {
      const i = Math.round(a + ((b2 - a) * s) / n), j = Math.round(c + ((d - c) * s) / n)
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) f[((j + dj + ny) % ny) * nx + ((i + di + nx) % nx)] = 0
    }
  }
  const f16 = fs => { const out = new Float16Array(fs.length * nx * ny); fs.forEach((f, k) => out.set(f, k * nx * ny)); return out }
  // scar centres sit where the source rollout's spiral turns; the second scar is the same rollout shifted
  const placements = [{ id: 's0', x: 0.55, y: 0.32, dx: 0, dy: 0 }, { id: 's1', x: 0.24, y: 0.71, dx: -40, dy: 50 }]
  const r = 0.06, tStop = Math.round(T * 0.35), fade = Math.round(T * 0.2)
  fs.rmSync(dir, { recursive: true, force: true })
  const scars = placements.map(({ id, x, y, dx, dy }) => {
    const scar = base.map(f => { const g = roll(f, nx, ny, dx, dy); disc(g, x, y, r); return g })
    writeNpy(`${dir}/${id}/u.npy`, f16(scar), [T, ny, nx])
    const lines = [ // A: scar to the edge (the one that works in this mock), B and C: elsewhere
      [[x + r, y], [Math.min(x + 0.4, 0.98), y]],
      [[x - 0.05, y + r + 0.08], [x - 0.05, Math.min(y + r + 0.3, 0.98)]],
      [[Math.max(x - 0.35, 0.02), y - r - 0.06], [x - 0.08, y - r - 0.06]],
    ].map(l => l.map(([a, b]) => [+a.toFixed(3), +b.toFixed(3)]))
    const ablations = lines.map((l, k) => {
      const aid = 'abc'[k], stops = k === 0
      const fr = scar.slice(tStop).map((f, t) => { // starts from the established arrhythmia, line ablated at frame 0
        const g = f.slice()
        line(g, l)
        if (stops) { const s = Math.min(Math.max((t - 8) / fade, 0), 1); for (let q = 0; q < g.length; q++) g[q] *= 1 - s * s * (3 - 2 * s) }
        return g
      })
      writeNpy(`${dir}/${id}/${aid}.npy`, f16(fr), [fr.length, ny, nx])
      return { id: aid, label: `Line ${aid.toUpperCase()}`, line: l, frames: `${id}/${aid}.npy`, stops_arrhythmia: stops }
    })
    return { id, x, y, r, frames: `${id}/u.npy`, ablations }
  })
  json(`${dir}/index.json`, {
    version: 1, mock: true,
    note: 'MOCK: baseline FNO (ap_fno) rollout shifted per scar; ablation lines and the stopping fade are drawn in, not simulated. Not Simula.',
    grid: { nx, ny }, geometry: 'sheet', dt: 12.9, time_unit: 'ms', range: [0, 1], scars,
  })
  console.log(`heart: ${scars.length} scars × ${scars[0].ablations.length + 1} rollouts × ${T} frames -> ${dir}`)
}

tsunami()
heart()

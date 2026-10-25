// The repo's three.js engines (../viz), driven by the deck: mount/teardown, frame playback, picking.
import * as THREE from 'three'
import { TsunamiEngine } from '@viz/tsunami/engine.js'
import { HeartEngine } from '@viz/heart/engine.js'
import type { TsunamiBank } from './bank'

// Cap on the 3D views' pixel ratio. 1 keeps two live engines (the hook) at 60 fps on a retina laptop; a 1080p
// projector renders at 1 anyway. ?dpr=2 for crisper views on a strong GPU, ?dpr=0.75 if the venue machine stutters.
const MAX_DPR = Number(new URLSearchParams(location.search).get('dpr')) || 1

/** Creates an engine in `el` with orbiting off and the pixel ratio matched to the scaled stage. Returns [engine, dispose]. */
export function mount<E extends TsunamiEngine | HeartEngine>(Make: new (el: HTMLElement) => E, el: HTMLElement): [E, () => void] {
  const e = new Make(el)
  e.controls.enabled = false
  let timer = 0
  const fit = () => { // the stage is CSS-scaled: render at the size it's shown, not its 1920-wide layout size
    const scale = el.getBoundingClientRect().width / (el.clientWidth || 1)
    e.renderer.setPixelRatio(Math.min(MAX_DPR, devicePixelRatio * scale) || 1)
    e.resize()
  }
  const onResize = () => { clearTimeout(timer); timer = window.setTimeout(fit, 120) } // after the stage re-fits
  fit()
  addEventListener('resize', onResize)
  return [e, () => {
    clearTimeout(timer)
    removeEventListener('resize', onResize)
    e.onFrame = undefined
    e.renderer.setAnimationLoop(null)
    e.controls.dispose()
    e.renderer.dispose()
    e.renderer.forceContextLoss()
    e.renderer.domElement.remove()
  }]
}

/** Frame playback: holds frame 0 for `wait` s (the model "computing"), plays at fps, then loops or holds the last frame. */
export class Clip {
  frames: Float32Array[] = []
  t = 0
  wait = 0
  rest = 0
  constructor(public fps = 12, public loop = true, public hold = 1.2) {}
  play(frames: Float32Array[], wait = 0) { this.frames = frames; this.t = 0; this.wait = wait; this.rest = 0 }
  get progress() { return this.frames.length > 1 ? this.t / (this.frames.length - 1) : 1 }
  tick(dt: number, set: (a: Float32Array, b: Float32Array, alpha: number) => void) {
    const last = this.frames.length - 1
    if (last < 0) return
    if (this.wait > 0) this.wait -= dt
    else if (this.t < last) this.t = Math.min(last, this.t + dt * this.fps)
    else if (this.loop && (this.rest += dt) > this.hold) { this.t = 0; this.rest = 0 }
    const i = Math.floor(this.t)
    set(this.frames[i], this.frames[Math.min(i + 1, last)], this.t - i)
  }
}

// ---------- tsunami ----------
export function tsunami(el: HTMLElement, bank: TsunamiBank, bed: Float32Array) {
  const [engine, dispose] = mount(TsunamiEngine, el)
  const { nx, ny, lx, ly } = bank.grid
  engine.setBathymetry(bed, nx, ny, { lx, ly })
  Object.assign(engine.params, { vexag: bank.view?.vexag ?? engine.autoVexag, wexag: bank.view?.wexag ?? 1 })
  engine.applyParams()
  engine.updateSun()
  // bed and sun never move and the water only receives shadows: render the shadow map once, not every frame
  engine.renderer.shadowMap.autoUpdate = false
  engine.renderer.shadowMap.needsUpdate = true
  const S = 100 / Math.max(lx, ly) // the engine's metres -> scene units
  const sea = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
  return {
    engine, dispose,
    /** grid fractions -> point on the sea surface */
    toWorld: (x: number, y: number, out = new THREE.Vector3()) => out.set((x - 0.5) * lx * S, 0, (y - 0.5) * ly * S),
    /** pointer -> grid fractions on the sea surface, or null off the map */
    pick(e: { clientX: number; clientY: number }): [number, number] | null {
      const p = rayAt(e, engine.camera, engine.renderer.domElement).ray.intersectPlane(sea, new THREE.Vector3())
      if (!p) return null
      const x = p.x / (lx * S) + 0.5, y = p.z / (ly * S) + 0.5
      return x >= 0 && x <= 1 && y >= 0 && y <= 1 ? [x, y] : null
    },
  }
}

/** Highest offshore surface over the frames: the engine scales its crest glow and colour map by it. */
export function etaMax(bed: Float32Array, frames: Float32Array[]) {
  let m = 0
  for (const h of frames) for (let k = 0; k < h.length; k++) if (bed[k] < 0 && h[k] > 0 && bed[k] + h[k] > m) m = bed[k] + h[k]
  return m
}

const ray = new THREE.Raycaster(), ndc = new THREE.Vector2()
export function rayAt(e: { clientX: number; clientY: number }, camera: THREE.Camera, canvas: HTMLElement) {
  const r = canvas.getBoundingClientRect()
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
  ray.setFromCamera(ndc, camera)
  return ray
}

// ---------- heart ----------
/**
 * Maps grid fractions to the tissue surface (in the heart group's frame), using the mesh's own aGrid attribute,
 * so it works for both the sheet and the ventricle. The first vertex with a given grid point is the top/outer
 * surface (the engine builds that part first).
 */
export function surface(mesh: THREE.Mesh) {
  const attr = (k: string) => mesh.geometry.getAttribute(k) as THREE.BufferAttribute
  const grid = attr('aGrid'), pos = attr('position'), nrm = attr('normal')
  const tri = new THREE.Triangle(), bary = [new THREE.Vector2(), new THREE.Vector2(), new THREE.Vector2()]
  return {
    at(x: number, y: number, lift = 0.012) {
      let best = 0, d = Infinity
      for (let k = 0; k < grid.count; k++) {
        const e = (grid.getX(k) - x) ** 2 + (grid.getY(k) - y) ** 2
        if (e < d) { d = e; best = k }
      }
      const n = new THREE.Vector3().fromBufferAttribute(nrm, best)
      return { p: new THREE.Vector3().fromBufferAttribute(pos, best).addScaledVector(n, lift), n }
    },
    /** pointer -> grid fractions on the tissue, or null */
    pick(e: { clientX: number; clientY: number }, camera: THREE.Camera, canvas: HTMLElement): [number, number] | null {
      const hit = rayAt(e, camera, canvas).intersectObject(mesh)[0]
      if (!hit?.face) return null
      const { a, b, c } = hit.face, local = mesh.worldToLocal(hit.point.clone())
      tri.setFromAttributeAndIndices(pos, a, b, c)
      ;[a, b, c].forEach((k, i) => bary[i].fromBufferAttribute(grid, k))
      const uv = THREE.Triangle.getInterpolation(local, tri.a, tri.b, tri.c, bary[0], bary[1], bary[2], new THREE.Vector2())
      return uv ? [uv.x, uv.y] : null
    },
  }
}

export { HeartEngine }

/**
 * Flood-risk heat map draped over the land (metres, so it goes in the engine's world group). paint(r, top) colours
 * each land cell by r / top, yellow -> ember -> red, transparent where dry or offshore.
 */
export function riskLayer(bank: TsunamiBank, bed: Float32Array) {
  const { nx, ny, lx, ly } = bank.grid, n = nx * ny
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 4), index: number[] = []
  for (let j = 0, k = 0; j < ny; j++) for (let i = 0; i < nx; i++, k++)
    pos.set([(i / (nx - 1) - 0.5) * lx, Math.max(bed[k], 0) + 0.4, (j / (ny - 1) - 0.5) * ly], k * 3)
  for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) { const a = j * nx + i, c = a + nx; index.push(a, c, a + 1, a + 1, c, c + 1) }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  geo.setAttribute('color', new THREE.BufferAttribute(col, 4))
  geo.setIndex(index)
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    vertexColors: true, transparent: true, depthWrite: false, toneMapped: false, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
  }))
  mesh.renderOrder = 5
  const ramp = [new THREE.Color('#ffe08a'), new THREE.Color('#ff7a45'), new THREE.Color('#e8263f')], c = new THREE.Color()
  return {
    mesh,
    paint(r: Float32Array, top: number) {
      for (let k = 0; k < n; k++) {
        const t = bed[k] > 0 ? Math.min(r[k] / top, 1) : 0
        if (t < 0.004) { col[k * 4 + 3] = 0; continue }
        c.lerpColors(ramp[t < 0.5 ? 0 : 1], ramp[t < 0.5 ? 1 : 2], t < 0.5 ? t * 2 : t * 2 - 1).multiplyScalar(1.25)
        col.set([c.r, c.g, c.b, 0.6 + 0.4 * Math.sqrt(t)], k * 4)
      }
      geo.attributes.color.needsUpdate = true
    },
  }
}

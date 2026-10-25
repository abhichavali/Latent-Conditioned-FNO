// Slide 5, heart and wildfire tabs (same bank schema): click to place a scar / start a fire, then try ablation lines / firebreaks.
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { loadBank, loadFrames, nearest, ratio, timing, useAsync, type HeartBank, type Scar } from './bank'
import { Clip, HeartEngine, mount, surface } from './engines'
import { BankStatus, MockBadge, SolverCard, slow } from './hud'
import { RaceView, type Setup } from './Hook'

type Kind = 'heart' | 'fire'
const TEXT = {
  heart: { kicker: (abl: boolean) => abl ? 'Heart · ablation planning' : 'Heart · scar risk', place: 'Click the tissue to place a scar',
    formed: (abl: boolean) => abl ? 'An arrhythmia forms. Try an ablation line' : 'An arrhythmia forms around the scar', ask: 'Does this line stop it?',
    panel: 'Ablation sites', ok: 'stops it', no: 'keeps going', yes: '✓ Arrhythmia stopped', nope: '✗ Still spiralling', screen: 'scar layouts' },
  fire: { kicker: (abl: boolean) => abl ? 'Wildfire · firebreak planning' : 'Wildfire · spread forecast', place: 'Click the forest to start a fire',
    formed: (abl: boolean) => abl ? 'The fire heads for the town. Try a firebreak' : 'The wind drives the fire toward the town', ask: 'Does this firebreak save the town?',
    panel: 'Firebreaks', ok: 'saves it', no: 'town burns', yes: '✓ Town saved', nope: '✗ Town burns', screen: 'ignition scenarios' },
}

export const HeartDemo = () => <SheetDemo kind="heart" />
export const FireDemo = () => <SheetDemo kind="fire" />

function SheetDemo({ kind }: { kind: Kind }) {
  const bank = useAsync(() => loadBank(kind), [kind])
  return bank.value ? <HeartLive bank={bank.value} kind={kind} /> : <BankStatus kind={kind} error={bank.error} />
}

// fire palette over the bank's display value d (see the fire bank's note): forest green -> charcoal (burned) ->
// slate (lake, bare ground) -> fire red -> orange -> white-hot; values > 1 glow through the bloom
const FIRE_STOPS: [number, number, number, number][] = [
  [0, 0.08, 0.13, 0.05], [0.05, 0.12, 0.19, 0.07], [0.2, 0.25, 0.42, 0.12], [0.26, 0.14, 0.14, 0.11], [0.32, 0.05, 0.045, 0.04],
  [0.38, 0.09, 0.08, 0.07], [0.42, 0.16, 0.3, 0.4], [0.48, 0.16, 0.3, 0.4], [0.52, 0.2, 0.1, 0.04], [0.55, 1.4, 0.16, 0.02],
  [0.75, 3.2, 0.95, 0.1], [1, 5, 3.6, 1.2],
]
function fireLut() {
  const n = 256, d = new Uint16Array(n * 4)
  for (let i = 0; i < n; i++) {
    const v = i / (n - 1), k = Math.max(1, FIRE_STOPS.findIndex(s => s[0] >= v)), [a, b] = [FIRE_STOPS[k - 1], FIRE_STOPS[k]]
    const w = Math.min(1, Math.max(0, (v - a[0]) / (b[0] - a[0] || 1)))
    for (let c = 0; c < 3; c++) d[i * 4 + c] = THREE.DataUtils.toHalfFloat(a[c + 1] + (b[c + 1] - a[c + 1]) * w)
    d[i * 4 + 3] = THREE.DataUtils.toHalfFloat(1)
  }
  const t = new THREE.DataTexture(d, n, 1, THREE.RGBAFormat, THREE.HalfFloatType)
  t.magFilter = t.minFilter = THREE.LinearFilter
  t.needsUpdate = true
  return t
}

type XY = [number, number]
const segDist = ([px, py]: XY, [[ax, ay], [bx, by]]: [XY, XY]) => {
  const dx = bx - ax, dy = by - ay, k = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy || 1)))
  return Math.hypot(px - ax - k * dx, py - ay - k * dy)
}
const mat = (c: string, k = 1, opacity = 1) =>
  new THREE.MeshBasicMaterial({ color: new THREE.Color(c).multiplyScalar(k), transparent: true, opacity, toneMapped: false, side: THREE.DoubleSide, depthWrite: false })
const COLORS = { idle: '#e9eef5', on: '#ffe2c2', stops: '#7fd4ff', fails: '#ff7a45' }

function label(text: string) {
  const c = Object.assign(document.createElement('canvas'), { width: 96, height: 96 }), g = c.getContext('2d')!
  g.fillStyle = 'rgba(5,7,11,0.8)'; g.beginPath(); g.arc(48, 48, 40, 0, 7); g.fill()
  g.fillStyle = '#e9eef5'; g.font = '500 52px "IBM Plex Sans", sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(text, 48, 52)
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }))
  s.scale.setScalar(0.22); s.renderOrder = 12
  return s
}

/** Grid, value range and (for fire) the forest palette, shared by the demo view and the side-by-side race. */
function prepSheet(engine: HeartEngine, bank: HeartBank, kind: Kind) {
  const { nx, ny } = bank.grid
  engine.params.autorotate = false
  engine.setGrid(nx, ny, { geometry: bank.geometry ?? 'sheet', lx: bank.grid.lx ?? nx, ly: bank.grid.ly ?? ny })
  engine.range = bank.range ?? [0, 1]
  if (kind === 'fire') { // forest floor instead of tissue; the LUT path ('clinical') carries the fire palette
    const e = engine as unknown as { luts: Record<string, THREE.Texture>; material: THREE.MeshPhysicalMaterial; params: Record<string, unknown>; applyParams(): void }
    e.luts.clinical = fireLut()
    Object.assign(e.params, { cmap: 'clinical', contract: 0, glow: 1 })
    e.material.color.set('#3c4a2c'); e.material.clearcoat = 0.1; e.material.sheen = 0
    e.applyParams()
  }
}

/** Side by side: the solver's frames at its measured speed vs Simula's forecast for the bank's first scenario. */
export function SheetRace({ kind }: { kind: Kind }) {
  const bank = useAsync(() => loadBank(kind), [kind])
  const setup = useMemo<Setup | null>(() => {
    const b = bank.value
    if (!b) return null
    const sc = b.scars[0]
    return (left, right) => Promise.all([loadFrames(kind, sc.frames, b.grid), sc.solver ? loadFrames(kind, sc.solver, b.grid) : null]).then(([sim, solved]) => {
      const [L, dl] = mount(HeartEngine, left), [R, dr] = mount(HeartEngine, right)
      for (const e of [L, R]) { prepSheet(e, b, kind); e.camera.position.set(0, 0.5, 6.2); e.controls.target.set(0, 0, 0) }
      return { L, R, sim, solver: solved ?? sim, dispose: () => { dl(); dr() } }
    })
  }, [bank.value, kind])
  if (!bank.value || !setup) return <BankStatus kind={kind} error={bank.error} />
  return <RaceView t={timing(kind)} setup={setup} mock={bank.value.mock} fps={kind === 'heart' ? 12 : 10} />
}

type Ctl = { click(e: { clientX: number; clientY: number }): void; ablate(id: string): void }

function HeartLive({ bank, kind }: { bank: HeartBank; kind: Kind }) {
  const T = TEXT[kind]
  const box = useRef<HTMLDivElement>(null), ctl = useRef<Ctl | null>(null), down = useRef<XY | null>(null)
  const [scar, setScar] = useState<Scar | null>(null)
  const [abl, setAbl] = useState<string | null>(null)
  const [verdicts, setVerdicts] = useState<Record<string, boolean>>({}) // `${scar}/${ablation}` -> stops?
  const [run, setRun] = useState(0)
  const [error, setError] = useState<string>()
  const t = timing(kind)

  useEffect(() => {
    const { nx, ny } = bank.grid, range = bank.range ?? [0, 1]
    const [engine, dispose] = mount(HeartEngine, box.current!)
    prepSheet(engine, bank, kind)
    engine.camera.position.set(0.7, 0.5, 6.8) // closer than the engine default, shifted so the side panel doesn't cover the tissue
    engine.controls.target.set(0.7, 0, 0)
    for (const s of bank.scars) for (const f of [s.frames, ...s.ablations.map(a => a.frames)]) loadFrames(kind, f, bank.grid).catch(() => {}) // prefetch

    const surf = surface(engine.mesh), up = new THREE.Vector3(0, 0, 1)
    const unit = surf.at(0.45, 0.5).p.distanceTo(surf.at(0.55, 0.5).p) / 0.1 // local units per grid fraction
    const onSurface = (m: THREE.Object3D, x: number, y: number, lift = 0.012) => {
      const { p, n } = surf.at(x, y, lift)
      m.position.copy(p); m.quaternion.setFromUnitVectors(up, n)
      return m
    }
    // every scar site as a faint ring; the placed scar as a solid disc
    for (const s of bank.scars) engine.heart.add(onSurface(new THREE.Mesh(new THREE.RingGeometry(s.r * unit * 0.85, s.r * unit, 48), mat(COLORS.idle, 1, 0.35)), s.x, s.y))
    if (bank.asset) { // the town the fire must not reach
      const { x, y, r, label: name } = bank.asset
      engine.heart.add(onSurface(new THREE.Mesh(new THREE.RingGeometry(r * unit * 0.8, r * unit, 48), mat('#ffe2c2', 2, 0.9)), x, y))
      const tag = label(name.slice(0, 1)); tag.position.copy(surf.at(x, y, 0.12).p); engine.heart.add(tag)
    }
    const disc = new THREE.Mesh(new THREE.CircleGeometry(1, 48), mat(kind === 'fire' ? '#ffb070' : '#3a3f4a', kind === 'fire' ? 2 : 1, 0.95))
    disc.visible = false
    engine.heart.add(disc)
    let lines = new THREE.Group()
    engine.heart.add(lines)

    const rest = new Float32Array(nx * ny).fill(range[0]), clip = new Clip(24)
    let current: Scar | null = null, pending: { id: string; stops: boolean } | null = null, token = 0
    engine.onFrame = dt => {
      if (!clip.frames.length) return engine.setFrames(rest, rest, 0)
      clip.tick(dt, (a, b, w) => engine.setFrames(a, b, w))
      if (pending && clip.wait <= 0 && clip.progress > 0.45) { // reveal the verdict partway through the rollout
        const { id, stops } = pending, tube = lines.getObjectByName(id) as THREE.Mesh
        pending = null
        tube.material = mat(stops ? COLORS.stops : COLORS.fails, 2.6)
        setVerdicts(v => ({ ...v, [`${current!.id}/${id}`]: stops }))
      }
    }
    const play = (file: string, loop: boolean, then?: () => void) => {
      const me = ++token
      loadFrames(kind, file, bank.grid).then(fr => {
        if (me !== token) return
        clip.loop = loop
        clip.play(fr, t.simula * slow(t))
        then?.()
        setRun(r => r + 1)
      }, e => setError(e.message))
    }
    const paint = (id: string) => lines.children.forEach(o => {
      if (o instanceof THREE.Mesh) o.material = o.name === id ? mat(COLORS.on, 2.6) : mat(COLORS.idle, 1, 0.55)
    })
    const placeScar = (s: Scar) => {
      current = s; pending = null
      onSurface(disc, s.x, s.y, 0.016).scale.setScalar(s.r * unit)
      disc.visible = true
      engine.heart.remove(lines)
      lines = new THREE.Group()
      for (const a of s.ablations) {
        const [[x0, y0], [x1, y1]] = a.line
        const pts = Array.from({ length: 24 }, (_, i) => surf.at(x0 + ((x1 - x0) * i) / 23, y0 + ((y1 - y0) * i) / 23, 0.02).p)
        const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 64, 0.02, 8), mat(COLORS.idle, 1, 0.55))
        tube.name = a.id; tube.renderOrder = 11
        const tag = label((a.label ?? a.id).replace(/^Line\s*/i, ''))
        tag.position.copy(surf.at(x1, y1, 0.12).p)
        lines.add(tube, tag)
      }
      engine.heart.add(lines)
      setScar(s); setAbl(null)
      play(s.frames, true)
    }
    const ablate = (id: string) => {
      const a = current?.ablations.find(x => x.id === id)
      if (!a) return
      pending = null // a verdict still waiting belongs to the previous line
      paint(id)
      setAbl(id)
      play(a.frames, false, () => { pending = { id, stops: a.stops_arrhythmia } })
    }
    ctl.current = {
      click(e) {
        const xy = surf.pick(e, engine.camera, engine.renderer.domElement)
        if (!xy) return
        const near = current?.ablations.map(a => [segDist(xy, a.line), a] as const).sort((p, q) => p[0] - q[0])[0]
        if (near && near[0] < 0.045) ablate(near[1].id)
        else placeScar(nearest(bank.scars, xy[0], xy[1]))
      },
      ablate,
    }
    return () => { token++; ctl.current = null; dispose() }
  }, [bank, kind, t.simula])

  const verdict = scar && abl ? verdicts[`${scar.id}/${abl}`] : undefined
  return (
    <div className="demo-stage grab">
      <div ref={box} className="engine"
        onPointerDown={e => { down.current = [e.clientX, e.clientY] }}
        onPointerUp={e => {
          const d = down.current
          down.current = null
          if (d && Math.hypot(e.clientX - d[0], e.clientY - d[1]) < 8) ctl.current?.click(e)
        }} />
      <div className="demo-title">
        <span className="kicker c-ember">{T.kicker(bank.scars.some(s => s.ablations.length))}</span>
        <h3>{!scar ? T.place : !abl ? T.formed(scar.ablations.length > 0) : T.ask}</h3>
      </div>
      <SolverCard t={t} run={run} who={bank.mock ? 'Mock forecast' : 'Simula'} className="corner" />
      {scar && scar.ablations.length > 0 && (
        <div className="ablations glass">
          <span className="kicker">{T.panel}</span>
          {scar.ablations.map(a => {
            const v = verdicts[`${scar.id}/${a.id}`]
            return (
              <button key={a.id} className={`abl${a.id === abl ? ' on' : ''}`} onMouseDown={e => e.preventDefault()} onClick={() => ctl.current?.ablate(a.id)}>
                <span>{a.label ?? a.id}</span>
                <i className={v === undefined ? '' : v ? 'ok' : 'no'}>{v === undefined ? '' : v ? T.ok : T.no}</i>
              </button>
            )
          })}
        </div>
      )}
      {verdict !== undefined && <div className={`verdict ${verdict ? 'ok' : 'no'}`}>{verdict ? T.yes : T.nope}</div>}
      <p className="demo-hint">Every rollout {ratio(t)}× faster than the simulator · fast enough to test thousands of {T.screen}</p>
      {bank.mock && <MockBadge what="baseline FNO, drawn-in ablations, not Simula" />}
      {error && <div className="bank-status"><strong>Demo error</strong><p>{error}</p></div>}
    </div>
  )
}

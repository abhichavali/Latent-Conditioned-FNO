// Slide 5, heart tab: click the tissue to place a scar (the arrhythmia forms), then try ablation lines.
import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { loadBank, loadFrames, nearest, timing, useAsync, type HeartBank, type Scar } from './bank'
import { Clip, HeartEngine, mount, surface } from './engines'
import { BankStatus, MockBadge, SolverCard } from './hud'

export function HeartDemo() {
  const bank = useAsync(() => loadBank('heart'), [])
  return bank.value ? <HeartLive bank={bank.value} /> : <BankStatus kind="heart" error={bank.error} />
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

type Ctl = { click(e: { clientX: number; clientY: number }): void; ablate(id: string): void }

function HeartLive({ bank }: { bank: HeartBank }) {
  const box = useRef<HTMLDivElement>(null), ctl = useRef<Ctl | null>(null), down = useRef<XY | null>(null)
  const [scar, setScar] = useState<Scar | null>(null)
  const [abl, setAbl] = useState<string | null>(null)
  const [verdicts, setVerdicts] = useState<Record<string, boolean>>({}) // `${scar}/${ablation}` -> stops?
  const [run, setRun] = useState(0)
  const [error, setError] = useState<string>()
  const t = timing('heart')

  useEffect(() => {
    const { nx, ny } = bank.grid, range = bank.range ?? [0, 1]
    const [engine, dispose] = mount(HeartEngine, box.current!)
    engine.params.autorotate = false
    engine.setGrid(nx, ny, { geometry: bank.geometry ?? 'sheet', lx: bank.grid.lx ?? nx, ly: bank.grid.ly ?? ny })
    engine.range = range
    engine.camera.position.set(0.7, 0.5, 6.8) // closer than the engine default, shifted so the side panel doesn't cover the tissue
    engine.controls.target.set(0.7, 0, 0)
    for (const s of bank.scars) for (const f of [s.frames, ...s.ablations.map(a => a.frames)]) loadFrames('heart', f, bank.grid).catch(() => {}) // prefetch

    const surf = surface(engine.mesh), up = new THREE.Vector3(0, 0, 1)
    const unit = surf.at(0.45, 0.5).p.distanceTo(surf.at(0.55, 0.5).p) / 0.1 // local units per grid fraction
    const onSurface = (m: THREE.Object3D, x: number, y: number, lift = 0.012) => {
      const { p, n } = surf.at(x, y, lift)
      m.position.copy(p); m.quaternion.setFromUnitVectors(up, n)
      return m
    }
    // every scar site as a faint ring; the placed scar as a solid disc
    for (const s of bank.scars) engine.heart.add(onSurface(new THREE.Mesh(new THREE.RingGeometry(s.r * unit * 0.85, s.r * unit, 48), mat(COLORS.idle, 1, 0.35)), s.x, s.y))
    const disc = new THREE.Mesh(new THREE.CircleGeometry(1, 48), mat('#3a3f4a', 1, 0.95))
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
      loadFrames('heart', file, bank.grid).then(fr => {
        if (me !== token) return
        clip.loop = loop
        clip.play(fr, t.simula)
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
  }, [bank, t.simula])

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
        <span className="kicker c-ember">Heart · ablation planning</span>
        <h3>{!scar ? 'Click the tissue to place a scar' : !abl ? 'An arrhythmia forms. Try an ablation line' : 'Does this line stop it?'}</h3>
      </div>
      <SolverCard t={t} run={run} who={bank.mock ? 'Mock forecast' : 'Simula'} className="corner" />
      {scar && (
        <div className="ablations glass">
          <span className="kicker">Ablation sites</span>
          {scar.ablations.map(a => {
            const v = verdicts[`${scar.id}/${a.id}`]
            return (
              <button key={a.id} className={`abl${a.id === abl ? ' on' : ''}`} onMouseDown={e => e.preventDefault()} onClick={() => ctl.current?.ablate(a.id)}>
                <span>{a.label ?? a.id}</span>
                <i className={v === undefined ? '' : v ? 'ok' : 'no'}>{v === undefined ? '' : v ? 'stops it' : 'keeps going'}</i>
              </button>
            )
          })}
        </div>
      )}
      {verdict !== undefined && <div className={`verdict ${verdict ? 'ok' : 'no'}`}>{verdict ? '✓ Arrhythmia stopped' : '✗ Still spiralling'}</div>}
      {bank.mock && <MockBadge what="baseline FNO, drawn-in ablations, not Simula" />}
      {error && <div className="bank-status"><strong>Demo error</strong><p>{error}</p></div>}
    </div>
  )
}

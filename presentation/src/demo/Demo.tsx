// Slide 5: the live demo. Tabs follow the slide's build steps (→ / ←) and can also be clicked.
import { useEffect, useRef, useState, type PointerEvent as PE } from 'react'
import * as THREE from 'three'
import gsap from 'gsap'
import { useMode, useStep } from '../deck/SlideFrame'
import { reduced } from '../deck/motion'
import { fmtDuration, loadBank, loadFrames, nearest, ratio, results, timing, useAsync, type TsunamiBank } from './bank'
import { Clip, etaMax, riskLayer, tsunami } from './engines'
import { BankStatus, Equations, MockBadge, Poster, SLOW, SolverCard } from './hud'
import { FireDemo, HeartDemo } from './HeartDemo'

const TABS = ['Tsunami', 'Flood-risk map', 'Heart', 'Wildfire']
const noFocus = (e: { preventDefault(): void }) => e.preventDefault() // keep focus off buttons so Space still drives the deck

export function Demo() {
  const step = useStep(), live = useMode() === 'live'
  const [tab, setTab] = useState(step)
  const [backup, setBackup] = useState(false)
  useEffect(() => setTab(step), [step])
  return (
    <div className="demo" data-no-nav>
      <div className="demo-bar">
        <span className="kicker">Live demo</span>
        <nav className="tabs">
          {TABS.map((name, i) => live
            ? <button key={name} aria-pressed={tab === i} onMouseDown={noFocus} onClick={() => setTab(i)}>{name}</button>
            : <span key={name} aria-pressed={tab === i}>{name}</span>)}
        </nav>
        {live && <button className="btn ghost" onMouseDown={noFocus} onClick={() => setBackup(true)}>Backup: play video</button>}
      </div>
      <div className="demo-view glass">
        {!live ? <Poster label={TABS[tab]} tone={tab >= 2 ? 'ember' : 'sea'} /> : tab < 2 ? <TsunamiDemo risk={tab === 1} /> : tab === 2 ? <HeartDemo /> : <FireDemo />}
      </div>
      {backup && <Backup close={() => setBackup(false)} />}
    </div>
  )
}

function Backup({ close }: { close: () => void }) {
  const [missing, setMissing] = useState(false)
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    addEventListener('keydown', k)
    return () => removeEventListener('keydown', k)
  }, [close])
  return (
    <div className="backup" data-no-nav>
      {missing ? (
        <div className="bank-status">
          <strong>No backup video yet</strong>
          <p>Put a screen recording of the demo at <code>public/demo/backup.mp4</code> before building (or at <code>dist/demo/backup.mp4</code> in a build).</p>
        </div>
      ) : (
        <video src={`${import.meta.env.BASE_URL}demo/backup.mp4`} controls autoPlay playsInline onError={() => setMissing(true)} />
      )}
      <button className="btn" onClick={close}>Close (Esc)</button>
    </div>
  )
}

// ---------- tsunami: drag the epicentre, and the flood-risk map ----------
function TsunamiDemo({ risk }: { risk: boolean }) {
  const bank = useAsync(() => loadBank('tsunami'), [])
  return bank.value ? <TsunamiLive bank={bank.value} risk={risk} /> : <BankStatus kind="tsunami" error={bank.error} />
}

type Ctl = { drag(e: PE): void; drop(): void; setRisk(on: boolean): void; runRisk(): void }
const glow = (c: string, k = 2.4, opacity = 1) => // toneMapped off + colour > 1 lets the engine's bloom catch it
  new THREE.MeshBasicMaterial({ color: new THREE.Color(c).multiplyScalar(k), transparent: true, opacity, depthTest: false, toneMapped: false, side: THREE.DoubleSide })
const flat = (m: THREE.Mesh, order = 10) => { m.rotation.x = -Math.PI / 2; m.renderOrder = order; return m }

// camera poses: the engine's default, and a higher view from the sea onto the coast for the risk map
const RISK_VIEW = { pos: new THREE.Vector3(-78, 90, 42), target: new THREE.Vector3(10, 0, -2) }

function TsunamiLive({ bank, risk }: { bank: TsunamiBank; risk: boolean }) {
  const box = useRef<HTMLDivElement>(null), ctl = useRef<Ctl | null>(null), dragging = useRef(false)
  const count = useRef<HTMLSpanElement>(null)
  const [ready, setReady] = useState(false)
  const [run, setRun] = useState(0)
  const [sel, setSel] = useState('')
  const [riskState, setRiskState] = useState<'idle' | 'running' | 'done'>('idle')
  const [error, setError] = useState<string>()
  const t = timing('tsunami'), tb = timing('batch')

  useEffect(() => {
    let dead = false, dispose = () => {}
    loadFrames('tsunami', bank.bed, bank.grid).then(([bed]) => {
      if (dead) return
      const v = tsunami(box.current!, bank, bed), { engine } = v
      dispose = v.dispose
      for (const s of bank.scenarios) loadFrames('tsunami', s.frames, bank.grid).catch(() => {}) // prefetch: switching is instant
      if (bank.risk) loadFrames('tsunami', bank.risk.frames, bank.grid).catch(() => {})

      // epicentre marker (sonar ring + beam) and faint dots at every precomputed epicentre
      const marker = new THREE.Group(), ping = flat(new THREE.Mesh(new THREE.RingGeometry(2.4, 2.9, 64), glow('#ffc477')))
      const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 14, 12), glow('#ffc477', 2, 0.55))
      beam.position.y = 7; beam.renderOrder = 10
      marker.add(flat(new THREE.Mesh(new THREE.RingGeometry(1.5, 2.1, 64), glow('#ffd9a3'))), flat(new THREE.Mesh(new THREE.CircleGeometry(0.7, 32), glow('#fff4e2', 3))), ping, beam)
      const ghosts = new THREE.Group()
      for (const s of bank.scenarios) {
        const g = flat(new THREE.Mesh(new THREE.RingGeometry(0.7, 1.0, 32), glow('#e9eef5', 1, 0.45)), 9)
        v.toWorld(s.x, s.y, g.position).y = 0.3
        ghosts.add(g)
      }
      engine.scene.add(ghosts, marker)

      const clip = new Clip(12, true, 1.5), home = { pos: engine.camera.position.clone(), target: engine.controls.target.clone() }
      const calm = bed.map(b => Math.max(-b, 0)) // still water, dry land
      const heat = riskLayer(bank, bed), mix = new Float32Array(bed.length)
      heat.mesh.visible = false
      engine.world.add(heat.mesh)
      let current = '', mode: 'quake' | 'risk' = 'quake', riskFrames: Float32Array[] = [], riskTop = 1, riskT0 = -1, clock = 0

      const select = (id: string) => {
        const s = bank.scenarios.find(x => x.id === id)!
        current = id
        loadFrames('tsunami', s.frames, bank.grid).then(fr => {
          if (dead || current !== id || mode !== 'quake') return
          engine.etaMax = etaMax(bed, fr)
          clip.play(fr, t.simula * SLOW) // the forecast appears after Simula's measured time, on the race card's scale
          setSel(id); setRun(r => r + 1)
        }, e => !dead && setError(e.message))
      }
      const place = (id: string, animate: boolean) => {
        const s = bank.scenarios.find(x => x.id === id)!, to = v.toWorld(s.x, s.y)
        if (animate && !reduced()) gsap.to(marker.position, { x: to.x, z: to.z, duration: 0.5, ease: 'back.out(2)' })
        else marker.position.copy(to)
      }
      const fly = (to: { pos: THREE.Vector3; target: THREE.Vector3 }) => {
        const d = reduced() ? 0 : 1.6
        gsap.to(engine.camera.position, { x: to.pos.x, y: to.pos.y, z: to.pos.z, duration: d, ease: 'power3.inOut' })
        gsap.to(engine.controls.target, { x: to.target.x, y: to.target.y, z: to.target.z, duration: d, ease: 'power3.inOut' })
      }

      engine.onFrame = dt => {
        clock += dt
        const k = (clock % 1.6) / 1.6
        ping.scale.setScalar(1 + 2.2 * k)
        ;(ping.material as THREE.MeshBasicMaterial).opacity = 1 - k
        if (mode === 'quake') return clip.tick(dt, (a, b, w) => engine.setFrames(a, b, w))
        engine.setFrames(calm, calm, 0)
        if (riskT0 < 0) return
        // the map builds up through the cumulative snapshots over the measured batch time
        const p = Math.min((performance.now() - riskT0) / 1000 / tb.simula, 1), f = p * (riskFrames.length - 1), i = Math.floor(f)
        const a = riskFrames[i], b = riskFrames[Math.min(i + 1, riskFrames.length - 1)], w = f - i
        for (let k = 0; k < mix.length; k++) mix[k] = a[k] + (b[k] - a[k]) * w
        heat.paint(mix, riskTop)
        if (count.current) count.current.textContent = Math.round(p * (bank.risk?.count ?? 0)).toLocaleString('en-US')
        if (p === 1) { riskT0 = -1; setRiskState('done') }
      }

      ctl.current = {
        drag(e) {
          const xy = v.pick(e)
          if (!xy) return
          gsap.killTweensOf(marker.position)
          v.toWorld(xy[0], xy[1], marker.position)
          const s = nearest(bank.scenarios, xy[0], xy[1])
          if (s.id !== current) select(s.id)
        },
        drop() { place(current, true) },
        setRisk(on) {
          mode = on ? 'risk' : 'quake'
          marker.visible = ghosts.visible = !on
          heat.mesh.visible = false
          riskT0 = -1
          setRiskState('idle')
          fly(on ? RISK_VIEW : home)
          if (!on) select(current)
        },
        runRisk() {
          if (!bank.risk) return
          loadFrames('tsunami', bank.risk.frames, bank.grid).then(snaps => {
            if (dead || mode !== 'risk') return
            const last = snaps.at(-1)!
            let m = 0
            for (let k = 0; k < last.length; k++) if (bed[k] > 0 && last[k] > m) m = last[k]
            riskTop = Math.max(m / 2, 1e-3) // colour saturates at half the deepest flooding
            riskFrames = [new Float32Array(bed.length), ...snaps]
            heat.mesh.visible = true
            riskT0 = performance.now()
            setRiskState('running')
          }, e => !dead && setError(e.message))
        },
      }
      current = bank.default ?? bank.scenarios[0].id
      place(current, false)
      setReady(true)
    }, e => !dead && setError(e.message))
    return () => { dead = true; ctl.current = null; dispose() }
  }, [bank, t.simula, tb.simula])

  useEffect(() => { if (ready) ctl.current!.setRisk(risk) }, [risk, ready])

  const who = bank.mock ? 'Mock forecast' : 'Simula'
  const n = bank.risk?.count ?? 0, batch = results.timing.batch.scenarios
  return (
    <div className={`demo-stage${risk ? '' : ' grab'}`}>
      <div ref={box} className="engine"
        onPointerDown={e => { if (risk || !ctl.current) return; e.currentTarget.setPointerCapture(e.pointerId); dragging.current = true; ctl.current.drag(e) }}
        onPointerMove={e => { if (dragging.current) ctl.current?.drag(e) }}
        onPointerUp={() => { if (dragging.current) { dragging.current = false; ctl.current?.drop() } }} />
      {!risk ? (
        <>
          <div className="demo-title"><span className="kicker c-sea">Tsunami forecast</span><h3>Drag the epicentre</h3></div>
          <SolverCard t={t} run={run} who={who} className="corner" />
          <Equations kind="tsunami" />
          <p className="demo-hint">Every forecast {ratio(t)}× faster than the simulator · {bank.scenarios.length} precomputed Simula forecasts · wave height exaggerated for visibility{sel ? '' : ' · loading…'}</p>
        </>
      ) : (
        <>
          <div className="demo-title"><span className="kicker c-sea">Flood-risk map</span><h3>{n.toLocaleString('en-US')} earthquakes, every stretch of coast</h3></div>
          {bank.risk ? (
            <div className="risk glass corner">
              <div className="risk-count"><span ref={count}>{riskState === 'idle' ? '0' : n.toLocaleString('en-US')}</span><small>earthquakes</small></div>
              {riskState === 'idle' && <button className="btn big" onMouseDown={noFocus} onClick={() => ctl.current?.runRisk()}>Run {n.toLocaleString('en-US')} earthquakes</button>}
              {riskState === 'done' && (
                <p className="risk-time">
                  <b>{batch.toLocaleString('en-US')} forecasts in {fmtDuration(tb.simula)}</b>
                  <small>the whole coast, every earthquake{tb.placeholder ? ' · placeholder timing' : ''}</small>
                </p>
              )}
              <div className="legend-bar"><i /><span>shallow</span><span>deep flooding</span></div>
            </div>
          ) : <div className="bank-status corner"><strong>No risk map in this bank</strong><p>Add <code>risk</code> to demo/tsunami/index.json.</p></div>}
        </>
      )}
      {bank.mock && <MockBadge what="baseline FNO, not Simula" />}
      {error && <div className="bank-status"><strong>Demo error</strong><p>{error}</p></div>}
    </div>
  )
}

// Side-by-side races (slide 1 and each demo's first tab): the solver computing frame by frame at its measured speed vs
// Simula's forecast, both stretched by the same factor so the solver takes RACE seconds on screen.
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { useMode } from '../deck/SlideFrame'
import { loadBank, loadFrames, ratio, timing, useAsync, type TsunamiBank } from './bank'
import { Clip, etaMax, tsunami } from './engines'
import { BankStatus, Poster, slow } from './hud'

// closer than the engine's default: the island and the coast fill the frame
const VIEW = { pos: new THREE.Vector3(-58, 26, 58), target: new THREE.Vector3(-4, 0, 8) }

export function Hook() {
  const live = useMode() === 'live'
  const bank = useAsync(() => loadBank('tsunami'), [])
  if (!live) return <div className="hook"><Poster label="Numerical solver" tone="ember" /><Poster label="Simula" /></div>
  if (!bank.value) return <div className="hook"><BankStatus kind="tsunami" error={bank.error} /></div>
  return <TsunamiRace bank={bank.value} />
}

function TsunamiRace({ bank }: { bank: TsunamiBank }) {
  const sc = bank.scenarios.find(s => s.id === bank.default) ?? bank.scenarios[0]
  const setup = useMemo<Setup>(() => (left, right) => Promise.all([
    loadFrames('tsunami', bank.bed, bank.grid),
    loadFrames('tsunami', sc.frames, bank.grid),
    sc.solver ? loadFrames('tsunami', sc.solver, bank.grid) : null,
  ]).then(([[bed], sim, solved]) => {
    const L = tsunami(left, bank, bed), R = tsunami(right, bank, bed)
    L.engine.etaMax = R.engine.etaMax = etaMax(bed, sim)
    for (const v of [L, R]) { v.engine.camera.position.copy(VIEW.pos); v.engine.controls.target.copy(VIEW.target) }
    return { L: L.engine, R: R.engine, sim, solver: solved ?? sim, dispose: () => { L.dispose(); R.dispose() } }
  }), [bank, sc])
  return <RaceView t={timing('tsunami')} setup={setup} mock={bank.mock} fps={8} />
}

type Engine = { onFrame?: (dt: number) => void; setFrames(a: Float32Array, b: Float32Array, w: number): void }
export type Setup = (left: HTMLElement, right: HTMLElement) => Promise<{ L: Engine; R: Engine; sim: Float32Array[]; solver: Float32Array[]; dispose(): void }>

export function RaceView({ t, setup, mock, fps }: { t: ReturnType<typeof timing>; setup: Setup; mock?: boolean; fps: number }) {
  const left = useRef<HTMLDivElement>(null), right = useRef<HTMLDivElement>(null)
  const clock = useRef<HTMLSpanElement>(null), bar = useRef<HTMLElement>(null), lState = useRef<HTMLSpanElement>(null)
  const rClock = useRef<HTMLSpanElement>(null), rState = useRef<HTMLSpanElement>(null)
  const [error, setError] = useState<string>()
  const go = useRef<() => void>(() => {}), [started, setStarted] = useState(false)

  useEffect(() => {
    let dead = false, dispose = () => {}
    setup(left.current!, right.current!).then(({ L: Le, R: Re, sim, solver, dispose: d }) => {
      if (dead) return d()
      dispose = d
      const L = { engine: Le }, R = { engine: Re }
      const fast = new Clip(fps, true, 1.5), done = new Clip(fps, true, 1.5)
      let t0 = -1, finished = false
      go.current = () => { t0 = performance.now(); finished = false; fast.play(sim, t.simula * slow(t)) }
      R.engine.onFrame = dt => t0 < 0 ? R.engine.setFrames(sim[0], sim[0], 0) : fast.tick(dt, (a, b, k) => R.engine.setFrames(a, b, k))
      L.engine.onFrame = dt => {
        if (t0 < 0) return L.engine.setFrames(solver[0], solver[0], 0) // waiting for the button
        const s = (performance.now() - t0) / 1000 / slow(t), p = Math.min(s / t.solver, 1)
        if (p < 1) { const f = solver[Math.floor(p * (solver.length - 1))]; L.engine.setFrames(f, f, 0) } // one solved step at a time
        else {
          if (!finished) { finished = true; done.play(solver) }
          done.tick(dt, (a, b, k) => L.engine.setFrames(a, b, k))
        }
        if (!clock.current || !bar.current || !lState.current || !rClock.current || !rState.current) return // unmounting
        clock.current.textContent = p < 1 ? `${Math.floor(p * 100)}%` : 'done'
        bar.current.style.transform = `scaleX(${p})`
        lState.current.textContent = p < 1 ? 'still computing…' : 'done'
        rClock.current.textContent = s < t.simula ? '' : `${ratio(t)}×`
        rState.current.textContent = s < t.simula ? 'computing…' : `done, ${ratio(t)}× faster`
      }
    }, e => !dead && setError(e.message))
    return () => { dead = true; dispose() }
  }, [setup, t.simula, t.solver, fps])

  const tags = ['relative speed, to scale', t.placeholder && 'placeholder timing', mock && 'mock frames (baseline FNO, not Simula)'].filter(Boolean)
  return (
    <>
      <div className="hook">
        <div className="hook-panel t-ember">
          <div ref={left} className="engine" />
          <div className="hud-top">
            <b><i className="dot" />Numerical solver</b>
            <span ref={clock} className="clock" />
          </div>
          <div className="hud-bottom"><div className="meter"><i ref={bar} /></div><span ref={lState} className="state" /></div>
        </div>
        <div className="hook-panel t-sea">
          <div ref={right} className="engine" />
          <div className="hud-top">
            <b><i className="dot" />{mock ? 'Mock · baseline FNO' : 'Simula'}</b>
            <span ref={rClock} className="clock" />
          </div>
          <div className="hud-bottom"><span ref={rState} className="state" /></div>
        </div>
        {error && <div className="bank-status"><strong>Couldn't load the race</strong><p>{error}</p></div>}
        {!started && <button className="btn big hook-go" onMouseDown={e => e.preventDefault()} onClick={() => { setStarted(true); go.current() }}>▶ Run both</button>}
      </div>
      {tags.length > 0 && <p className="ph-line">{tags.join(' · ')}{t.placeholder || mock ? ' · rehearsal only' : ''}</p>}
    </>
  )
}

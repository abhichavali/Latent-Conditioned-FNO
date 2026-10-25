import type { SlideDef } from '../deck/SlideFrame'
import { Frame, build } from '../components/layouts'
import { Chart, type Series } from '../components/Chart'
import { Tex } from '../components/Tex'
import data from '../data/baseline.json'

type Row = (typeof data.swe)[number]
const pts = (rows: Row[], k: 'one_step' | 'rollout_mean' | 'rollout_final') =>
  rows.map(r => ({ x: r.n, y: r[k][0], err: r[k][1] }))
const series = (rows: Row[], c: string): Series[] => [
  { name: 'one step', points: pts(rows, 'one_step'), color: 'var(--ink-2)', step: 1 },
  { name: '19-step rollout, mean', points: pts(rows, 'rollout_mean'), color: c, step: 2 },
  { name: '19-step rollout, final', points: pts(rows, 'rollout_final'), color: c, dashed: true, step: 3 },
]
const common = { width: 780, height: 430, logX: true, logY: true, xDomain: [3.2, 1300] as [number, number], yDomain: [0.004, 1] as [number, number],
  xTicks: [4, 16, 64, 256, 1024], xLabel: 'training simulations (initial conditions)', yLabel: 'test nRMSE', axisStep: 0 }

export default {
  title: 'A from-scratch FNO needs hundreds of simulations',
  tone: 'sea',
  section: 'Results',
  steps: 4,
  Component: () => (
    <Frame title="A from-scratch FNO needs hundreds of simulations"
      lede="Plain FNO, no prior: N solver runs per target, tested on 128 unseen initial conditions.">
      <div className="row" style={{ gap: 64, marginTop: -8 }}>
        <Chart {...common} series={series(data.swe, 'var(--sea)')} title={<>Shallow water, <Tex>{'\\eta'}</Tex></>} />
        <Chart {...common} series={series(data.ap, 'var(--ember)')} title={<>Aliev–Panfilov, <Tex>u</Tex></>} legend={false} />
      </div>
      <p className="small" style={{ marginTop: 24, maxWidth: '80ch', color: 'var(--ink-2)' }} {...build(4)}>
        Rollout error falls as <Tex>{'N^{-0.41}'}</Tex> (tsunami) and <Tex>{'N^{-0.48}'}</Tex> (heart):
        halving it costs 4–5× more simulations, and both curves flatten past 512.
      </p>
    </Frame>
  ),
  notes: `This is the bar we have to beat, with real numbers from baseline_FNO/RESULTS.md.
A plain FNO (4 Fourier layers, width 64, 16 modes), trained from scratch on N simulations of each target.
(click) One-step error.  (click) 19-step autoregressive rollout, mean over the rollout.  (click) Error at the final step.
(click) Log-log slopes: rollout mean -0.41 for shallow water, -0.48 for Aliev–Panfilov, so halving the error takes 2^(1/0.41) ≈ 5.4× and 2^(1/0.48) ≈ 4.2× the data.
N = 512 and 1024 both hit the 6,000-step cap, so the top-end flattening is partly compute, not only data.
Error bars: std over 3 seeds, N ≤ 64.`,
} satisfies SlideDef

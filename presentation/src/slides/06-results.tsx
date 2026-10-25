import type { SlideDef } from '../deck/SlideFrame'
import { build } from '../components/layouts'
import { Chart } from '../components/Chart'
import { Stat } from '../components/Stat'
import { fmtDuration, results } from '../demo/bank'

// Every number here comes from src/data/results.json. null renders as a dash; placeholder: true shows "results pending".
const { batch } = results.timing
const speedup = batch.simula_seconds && batch.solver_seconds ? batch.solver_seconds / batch.simula_seconds : null
const dec = (v: number | null) => (v !== null && v < 10 ? 1 : 0)
const COLORS: Record<string, string> = { simula: 'var(--sea)', standard_ai: '#8c97a8' }
const series = results.chart.series.map(s => ({
  name: s.points.length ? s.label : `${s.label}: results pending`,
  points: s.points.map(p => ({ x: p.sims, y: p.error })),
  color: COLORS[s.id] ?? 'var(--latent)', step: 1,
}))
const xs = series.flatMap(s => s.points.map(p => p.x)), ys = series.flatMap(s => s.points.map(p => p.y))
const xDomain: [number, number] = [Math.min(...xs) / 1.4, Math.max(...xs) * 1.4]
const yDomain: [number, number] = [Math.min(...ys) / 1.5, Math.max(...ys) * 1.5]
const ticks = (lo: number, hi: number) => [1, 2, 5].flatMap(m => [-3, -2, -1, 0, 1, 2, 3, 4].map(e => m * 10 ** e)).filter(v => v >= lo && v <= hi).sort((a, b) => a - b)
const simulaPending = !results.chart.series.find(s => s.id === 'simula')?.points.length

export default {
  title: 'Results',
  section: 'Results',
  steps: 1,
  Component: () => (
    <div className="frame results">
      <header className="results-head">
        <h2 className="slide-head" {...build(0, 'words')}>Results</h2>
        {results.placeholder && <span className="pending-stamp" {...build(0, 'fade', 0.3)}>Results pending</span>}
      </header>
      <div className="results-body">
        <div className="giants">
          <Stat value={results.fewer_simulations} decimals={dec(results.fewer_simulations)} unit="×" step={0}
            label={<><b>fewer simulations</b> than a standard AI model to reach the same accuracy</>} />
          <Stat value={speedup} decimals={dec(speedup)} unit="×" step={0}
            label={<><b>faster</b> than the simulator ({batch.scenarios.toLocaleString('en-US')} scenarios: {batch.simula_seconds === null ? '—' : fmtDuration(batch.simula_seconds)} vs {batch.solver_seconds === null ? '—' : fmtDuration(batch.solver_seconds)})</>} />
        </div>
        <div className="chart-box" {...build(1, 'fade')}>
          <Chart width={880} height={470} logX logY xDomain={xDomain} yDomain={yDomain} xTicks={ticks(...xDomain)} yTicks={ticks(...yDomain)}
            xLabel={results.chart.x_label} yLabel={results.chart.y_label} title={results.chart.title} series={series} />
          {simulaPending && <div className="chart-pending" {...build(1, 'fade', 0.8)}>Simula curve: results pending</div>}
        </div>
      </div>
      <p className="footnote" {...build(1, 'fade', 1)}>Simula never trained on tsunamis or heart tissue.</p>
    </div>
  ),
  notes: `(3:10–3:35) All values come from src/data/results.json. While "Results pending" shows, don't quote numbers.

X× fewer simulations than a standard AI model to reach the same accuracy.
Y× faster than the simulator: 1,000 scenarios in Z seconds vs W minutes.
(click) The chart: error vs number of simulations, Simula vs standard AI. The standard-AI curve is measured (FNO trained from scratch, baseline_FNO).
Footnote: "And Simula never trained on tsunamis or heart tissue."`,
} satisfies SlideDef

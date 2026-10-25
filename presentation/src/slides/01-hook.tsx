import type { SlideDef } from '../deck/SlideFrame'
import { build } from '../components/layouts'
import { Hook } from '../demo/Hook'

export default {
  title: 'Simula: speeding up simulation with AI',
  chrome: false,
  Component: () => (
    <div className="hook-slide">
      <Hook />
      <div className="hook-title">
        <h1 className="hero" {...build(0, 'fade', 3)}>Simula</h1>
        <p className="sub" {...build(0, 'fade', 3.6)}>Speeding up simulation with AI</p>
      </div>
    </div>
  ),
  notes: `(0:00–0:20) Let the clip run for a beat before speaking. The title comes in on its own after ~3 s.

Say: "Same physics. One of these is ten times faster."

Left: the numerical solver, computing the tsunami step by step, timer running. Right: Simula's forecast, already done, the wave already hitting the island.
Then: "This is Simula. We speed up simulation with AI."

Timings come from src/data/results.json (timing.tsunami, measured 12×). The race is drawn to scale but slowed, with no seconds on screen; if asked, one forecast is 0.02 s vs 0.23 s on one GPU. If the panels say "placeholder timing" or "mock frames", you're rehearsing, not presenting real numbers.`,
} satisfies SlideDef

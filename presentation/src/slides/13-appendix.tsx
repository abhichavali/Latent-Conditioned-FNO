import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'
import { Img } from '../components/Media'
import fig from '../../../baseline_FNO/results/error_vs_data.png' // bundled at build time, stays in sync with the repo

export default {
  title: 'Appendix: baseline sweep',
  tone: 'sea',
  section: 'Appendix',
  Component: () => (
    <Frame title="Appendix: the baseline sweep">
      <Img src={fig} alt="Test normalized RMSE against number of training initial conditions for shallow water and Aliev–Panfilov, one-step and 19-step rollout"
        framed caption="baseline_FNO/results/error_vs_data.png: one-step and 19-step rollout error vs. training ICs, 3 seeds for N ≤ 64." />
    </Frame>
  ),
  notes: `Backup for Q&A: the original matplotlib figure behind the baseline slide.
Settings: FNO2d, 4 Fourier layers, width 64, 16 modes, residual one-step map, AdamW + cosine, batch 16. 128² periodic grid, 20 frames per rollout.
Long horizon (57 steps, 3× training): SWE error falls once the tsunami is absorbed at the coast; AP drifts up to ~0.39.`,
} satisfies SlideDef

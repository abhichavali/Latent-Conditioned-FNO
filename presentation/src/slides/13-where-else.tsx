import type { ReactNode } from 'react'
import type { SlideDef } from '../deck/SlideFrame'
import { build } from '../components/layouts'

// Line icons (120 × 120), drawn in with DrawSVG.
const ICONS: { label: string; tone: string; icon: ReactNode }[] = [
  { label: 'Floods & weather', tone: 'sea', icon: <>
    <path d="M30 62 a18 18 0 0 1 8 -34 a24 24 0 0 1 44 6 a16 16 0 0 1 8 28 Z" />
    <path d="M40 76 l-6 12 M60 76 l-6 12 M80 76 l-6 12" />
    <path d="M14 104 q11 -8 22 0 t22 0 t22 0 t22 0 t22 0" />
  </> },
  { label: 'Drug & tissue modeling', tone: 'ember', icon: <>
    <rect x="22" y="44" width="76" height="32" rx="16" transform="rotate(-35 60 60)" />
    <path d="M47 41 L73 79" />
    <circle cx="94" cy="96" r="9" /><circle cx="24" cy="96" r="6" /><circle cx="98" cy="24" r="5" />
  </> },
  { label: 'Materials', tone: 'latent', icon: <>
    <path d="M60 14 L100 37 L100 83 L60 106 L20 83 L20 37 Z" />
    <path d="M60 14 L60 60 L100 83 M60 60 L20 83" />
    <circle cx="60" cy="60" r="5" /><circle cx="60" cy="14" r="4" /><circle cx="100" cy="83" r="4" /><circle cx="20" cy="83" r="4" />
  </> },
  { label: 'Engineering design', tone: 'sea', icon: <>
    <path d="M14 66 C30 44 70 38 106 56 C76 64 40 70 14 66 Z" />
    <path d="M8 38 q30 -10 60 -2 t48 6 M8 92 q30 8 60 0 t48 -6" />
  </> },
]

export default {
  title: 'Where else this goes',
  section: 'Where else',
  tone: 'latent',
  steps: 1,
  // DrawSVG works on shapes, not groups: draw every stroke of every icon (selectors are scoped to this slide)
  build: (k, tl, at) => { if (k === 0) tl.fromTo('.line-icon > *', { drawSVG: '0%' }, { drawSVG: '100%', duration: 1.4, stagger: 0.06, ease: 'power3.inOut' }, `${at}+=0.5`) },
  Component: () => (
    <div className="frame where">
      <h2 className="slide-head" {...build(0, 'words')}>Where else this goes</h2>
      <div className="domains">
        {ICONS.map(({ label, tone, icon }, i) => (
          <div key={label} className={`domain glass t-${tone}`} {...build(0, 'fade', 0.3 + i * 0.15)}>
            <svg viewBox="0 0 120 120" className="line-icon">
              {icon}
            </svg>
            <span>{label}</span>
          </div>
        ))}
      </div>
      <p className="where-3d" {...build(0, 'fade', 1.1)}><b>Next: 3D.</b> A 3D fluid simulation (CFD) can take hours. AI surrogates already compute car drag <b>26,000×</b> faster than GPU CFD<sup>1</sup>: hours become seconds.</p>
      <p className="footnote" {...build(0, 'fade', 1.3)}><sup>1</sup> Li et al., Geometry-Informed Neural Operator for Large-Scale 3D PDEs, NeurIPS 2023.</p>
      <p className="punch wide" {...build(1, 'words')}>Anywhere people wait on physics simulations.</p>
    </div>
  ),
  notes: `(3:35–3:50)

Floods & weather · drug & tissue modeling · materials · engineering design.
"And the big one is 3D. A single 3D fluid simulation, like airflow over a car, can take hours. Neural surrogates have already shown 26,000× speed-ups on exactly that (Li et al., NeurIPS 2023). Hours become seconds." 
(click) Punchline: "Anywhere people wait on physics simulations."`,
} satisfies SlideDef

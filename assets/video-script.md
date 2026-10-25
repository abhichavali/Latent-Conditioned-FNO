# Simula: 2-minute video script

About 290 words at a relaxed pace. Brackets are on-screen actions.

| Time | Slide | Say |
|---|---|---|
| 0:00 | 1 · Hook | [Click **▶ Run both**] Same physics. Same tsunami. One of these finished twelve times faster. |
| 0:08 | 2 · Problem | When an earthquake hits, forecasters have minutes to decide who evacuates. A heart surgeon has dozens of possible ablation sites, and each one needs its own simulation. Simulation is too slow for decisions that can't wait. |
| 0:20 | 3 · Why AI hasn't fixed it | AI surrogates like Fourier Neural Operators are fast. But every new problem needs thousands of simulations to train one. The bottleneck just moves. |
| 0:30 | 4 · Simula | Simula is a neural operator pretrained on over five thousand kinds of physics. It reads your equation, then learns your problem from a handful of simulations instead of thousands. |
| 0:45 | 5 · Applications | Three problems it never saw in training. |
| 0:47 | 6 · Tsunami setup | First, a tsunami. An undersea quake sends a wave at the coast. [Click: equations] The physics: the shallow-water equations. |
| 0:54 | 7 · Tsunami demo | [Drag the epicentre] Move the quake, and Simula forecasts the wave hitting the coast while the solver is still working. [→ Flood-risk map, click **Run 1,000 earthquakes**] Forecasters think in thousands of scenarios. Here's a thousand earthquakes, flood risk for every stretch of coast, in five and a half seconds. |
| 1:10 | 8 · Heart setup | Next, a heart. Scar tissue conducts slowly, and that can trigger an arrhythmia. [Click: equations] |
| 1:15 | 9 · Heart demo | [Click a scar ring] Place a scar, and the spiral forms. Each rollout is about twelve times faster than the simulator. |
| 1:23 | 10 · Wildfire setup | Last, a wildfire, burning through fuel, pushed by the wind. [Click: equations] |
| 1:28 | 11 · Wildfire demo | [Click an ignition ring] Start a fire anywhere, and see where it's headed before it gets there. |
| 1:36 | 12 · Results | The numbers. On the heart, five simulations got Simula the accuracy a standard model needed sixty-four for. That's twelve times fewer. On wildfire, about two times. And a thousand scenarios, run one at a time, finish ten times faster than the simulator. |
| 1:50 | 13 · Where else | And this is 2D. In 3D, a single fluid simulation can take hours, and neural operators have already cut that to seconds. |
| 1:57 | 14 · Close | Simula turns hours of simulation into decisions in seconds. |

## Recording tips
- Run `npm run dev` in `presentation/`, full-screen the browser, and record with Cmd+Shift+5.
- Slide 1 waits for **▶ Run both**. Start talking as you click.
- The risk map takes about 5.5 s to fill. Keep talking over it.
- On slides 6, 8 and 10, one click reveals the equations card.

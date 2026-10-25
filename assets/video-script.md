# Simula: 2-minute video script

About 320 words, brisk pace at a relaxed pace. Brackets are on-screen actions.

| Time | Slide | Say |
|---|---|---|
| 0:00 | 1 · Hook | [Click **▶ Run both**] Same physics. Same tsunami. One of these finished twelve times faster. |
| 0:07 | 2 · Problem | When an earthquake hits, forecasters have minutes to decide who evacuates. A heart surgeon has dozens of possible ablation sites, each needing its own simulation. Simulation is too slow for decisions that can't wait. |
| 0:18 | 3 · Why AI hasn't fixed it | Neural operators like FNOs are fast, but every new problem needs thousands of simulations to train one. The bottleneck just moves. |
| 0:26 | 4 · Simula | Simula is pretrained on over five thousand kinds of physics. It reads your equation and learns your problem from a handful of simulations. |
| 0:35 | 5 · Equation → graph | [Click through] Every equation becomes a graph: variables and constants are nodes, operations connect them. Pair it with simulations and you get a training example. We built 5,120 of them, about 200,000 simulations, on four rented GPUs. |
| 0:47 | 6 · Equation → forecast | [Click through] Give it an equation it's never seen. The encoder writes a latent code, a few real simulations sharpen it, and the neural operator turns it into a forecast. |
| 0:57 | 7 · Applications | Three problems it never saw in training. |
| 0:59 | 8 · Tsunami setup | A tsunami: an undersea quake sends a wave at the coast. [Click: equations] The shallow-water equations. |
| 1:05 | 9 · Tsunami demo | [Drag the epicentre] Simula forecasts the wave while the solver is still working. [→ Flood-risk map, **Run 1,000 earthquakes**] A thousand earthquakes, flood risk for every stretch of coast, in five and a half seconds. |
| 1:18 | 10 · Heart setup | A heart: scar tissue conducts slowly and can trigger an arrhythmia. [Click: equations] |
| 1:22 | 11 · Heart demo | [Click a scar ring] Place a scar, and the spiral forms, about twelve times faster than the simulator. |
| 1:28 | 12 · Wildfire setup | A wildfire, burning through fuel, pushed by the wind. [Click: equations] |
| 1:32 | 13 · Wildfire demo | [Click an ignition ring] Start a fire anywhere and see where it's headed before it gets there. |
| 1:38 | 14 · Results | On the heart, five simulations got Simula the accuracy a standard model needed sixty-four for: twelve times fewer. On wildfire, about two times. A thousand scenarios, one at a time, run ten times faster than the simulator. |
| 1:51 | 15 · Where else | And this is 2D. In 3D, one fluid simulation can take hours, and neural operators have already cut that to seconds. |
| 1:57 | 16 · Close | Simula turns hours of simulation into decisions in seconds. |

## Recording tips
- Run `npm run dev` in `presentation/`, full-screen the browser, and record with Cmd+Shift+5.
- Slide 1 waits for **▶ Run both**. Start talking as you click.
- The risk map takes about 5.5 s to fill. Keep talking over it.
- On slides 8, 10 and 12, one click reveals the equations card. Slides 5 and 6 build in 4 and 3 clicks.

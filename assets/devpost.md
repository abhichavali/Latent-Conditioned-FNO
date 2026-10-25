## Inspiration
Fourier Neural Operators are absurdly fast at physics. An FNO learns a PDE's solution operator, then predicts in milliseconds what a numerical solver grinds through for minutes. The catch: every new problem needs its own FNO, trained on thousands of simulations from the slow solver you were trying to avoid. We wanted one model that has already seen thousands of kinds of physics, so a new problem only costs a handful of runs.

## What it does
Simula is a pretrained FNO that reads your equation. Give it the PDE and a few simulations, and it adapts and forecasts the rest.

We tested it on three problems it never saw in pretraining:
- **Tsunami:** drag an earthquake, watch the wave hit the coast, map flood risk for 1,000 quakes
- **Heart:** place scar tissue, watch an arrhythmia form
- **Wildfire:** start a fire, watch the wind push it toward a town

On the heart, 5 simulations got Simula the accuracy a from-scratch FNO needed 64 for: 12× fewer. On wildfire, about 2× fewer. Each forecast runs 12× faster than our GPU solver, and 1,000 tsunami scenarios run one at a time take 23 seconds instead of almost 4 minutes.

## How we built it
- A symbolic PDE generator (sympy) that turns any equation into a graph of its terms, plus a batched spectral solver on exponax/JAX that simulates thousands of different equations without recompiling.
- 5,120 equations and ~197k simulations at 128×128, generated on rented GPUs. Initial conditions are picked by maximum entropy so the data isn't a pile of near-duplicates. Tsunamis, hearts and wildfires were deliberately left out.
- A conditioned FNO in JAX/Equinox. Every equation gets a latent code, and a graph neural net learns to predict that code from the equation itself.
- Fine-tuning on a few simulations of each new problem, with an active-learning loop choosing which ones to run.
- A demo deck in React, GSAP and Three.js.

## Challenges we ran into
- One blown-up simulation poisoned our feature normalization, and 1,280 equations silently reused the same two initial conditions. We caught it, fixed it and reselected.
- Units bite. Running the tsunami in kilometres instead of metres broke our normalization.
- Our heart scar slows electrical waves instead of blocking them, so no ablation line ever stopped the arrhythmia, in the solver or the model. We cut ablation from the demo instead of faking it.
- Smart experiment selection lost to random picking. We report it anyway.
- Our 515M-parameter v3 went NaN ten minutes into training.
- Three rented GPUs, one tight budget.

## Accomplishments that we're proud of
- A full pipeline, from random equation to pretrained model to interactive demo, in one weekend.
- Real transfer: a model that never saw a heart or a wildfire learns them from far fewer simulations.
- Every number in the deck is measured. When something didn't work, we said so.

## What we learned
- Pretraining pays off when data is scarce. By ~100 simulations, a from-scratch FNO catches up.
- Most of the gain came from fine-tuning, not from reading the equation. A 32-number latent was too small to describe new physics on its own.
- Covering the right physics beats raw volume. 5× more equations barely moved our results.

## What's next for PDE Dynamics Encoder
- v3 is training now. A transformer reads the equation plus every simulation it has, and updates the latent in one forward pass: no fine-tuning. Because it's trained to estimate its own uncertainty, smart experiment selection should finally earn its keep.
- 3D, where one CFD run takes hours and learning from a handful of runs matters most.
- Patient-specific heart models and real-time hazard maps.

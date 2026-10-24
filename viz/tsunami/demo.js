// Synthetic stand-in for a solver: a leading-depression tsunami crossing a shelf and flooding a coast.
// Not physics: per-column travel times with c = sqrt(g h) and Green's-law shoaling, so fronts slow,
// bend and steepen plausibly. Produces exactly what a real solver would hand the engine: b and h frames.

const hash = (x, y) => { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); };
const lerp = (a, b, t) => a + (b - a) * t;
function vnoise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), u = x - xi, v = y - yi, su = u * u * (3 - 2 * u), sv = v * v * (3 - 2 * v);
  return lerp(lerp(hash(xi, yi), hash(xi + 1, yi), su), lerp(hash(xi, yi + 1), hash(xi + 1, yi + 1), su), sv);
}
const fbm = (x, y) => { let s = 0, a = 0.5; for (let o = 0; o < 5; o++) { s += a * vnoise(x, y); x *= 2.03; y *= 2.03; a *= 0.5; } return s; };
const sech2 = x => { const c = Math.cosh(x); return 1 / (c * c); };

export function tsunamiDemo() {
  const nx = 256, ny = 256, lx = 4000, ly = 4000, N = nx * ny;
  const b = new Float32Array(N), isl = new Float32Array(N);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = i / (nx - 1), y = j / (ny - 1);
    const coast = 0.6 + 0.045 * Math.sin(7 * y + 0.6) + 0.025 * Math.sin(17 * y + 2) - 0.11 * Math.exp(-(((y - 0.42) / 0.085) ** 2))
      + 0.07 * Math.exp(-(((y - 0.8) / 0.045) ** 2)) + 0.05 * (fbm(3 * y, 7.3) - 0.5);
    const d = x - coast;
    let z;
    if (d < 0) {
      const shelf = 1 - Math.exp(d * 7);
      z = -58 * shelf + 1.2 * Math.sin(d * 140) * Math.exp(d * 25) + 5 * (fbm(x * 6, y * 6) - 0.5) * shelf;
    } else {
      const hills = (0.5 + 0.9 * fbm(x * 5 + 3, y * 5)) * (1 - 0.5 * Math.exp(-(((y - 0.42) / 0.12) ** 2)));
      const valley = Math.exp(-(((y - 0.42 - 0.04 * Math.sin(x * 14)) / (0.03 + 0.05 * d)) ** 2));
      z = 85 * (1 - Math.exp(-d * 4.5)) * hills * (1 - 0.85 * valley);
    }
    isl[j * nx + i] = 74 * Math.exp(-((x - 0.3) ** 2 + (y - 0.7) ** 2) / 0.0016); // island
    b[j * nx + i] = z + isl[j * nx + i];
  }

  // arrival time T and amplitude A per cell, marching shoreward column by column
  const g = 9.81, dxm = lx / (nx - 1), c0 = Math.sqrt(g * 58), R = 5000, A0 = 3;
  const T = new Float32Array(N), A = new Float32Array(N);
  for (let j = 0; j < ny; j++) {
    let t = (Math.hypot((j / (ny - 1)) * ly - 0.25 * ly, R) - R) / c0, loss = 1, green = 1;
    const d0 = Math.max(-b[j * nx], 1);
    for (let i = 0; i < nx; i++) {
      // the island only shadows amplitude; timing ignores it so fronts stay continuous behind it
      const k = j * nx + i, depth = isl[k] - b[k];
      let c = 7;
      if (depth > 0.3) { c = Math.sqrt(g * depth); green = Math.min((d0 / depth) ** 0.25, 2.6); }
      else loss *= Math.exp(-dxm / 350);
      if (b[k] > 0 && depth > 0.3) loss *= Math.exp(-dxm / 600);
      T[k] = t; A[k] = A0 * green * loss;
      t += dxm / c;
    }
  }
  for (const [F, passes] of [[T, 4], [A, 10]]) for (let pass = 0; pass < passes; pass++) for (let i = 0; i < nx; i++) {
    let prev = F[i];
    for (let j = 1; j < ny - 1; j++) { const k = j * nx + i, cur = F[k]; F[k] = 0.25 * prev + 0.5 * cur + 0.25 * F[k + nx]; prev = cur; }
  }

  // drawdown first, then the main crest, then a smaller follower
  const profile = s => -0.55 * sech2(s / 22) + sech2((s - 60) / 16) + 0.4 * sech2((s - 140) / 20);
  const dt = 2, t0 = -30;
  return {
    nx, ny, b, count: 240,
    meta: { lx, ly, dt, time_unit: 's', title: 'Synthetic tsunami', vexag: 4, wexag: 6 },
    frame(k) {
      const t = t0 + k * dt, h = new Float32Array(N);
      for (let i = 0; i < N; i++) h[i] = Math.max(A[i] * profile(t - T[i]) - b[i], 0);
      return h;
    },
  };
}

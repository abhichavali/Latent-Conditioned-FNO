// Synthetic stand-in for solver output: a meandering spiral rotor (re-entry) on an x-periodic grid.
// Analytic, not Aliev–Panfilov — it only exists so the engine has something to show.
const ss = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
const ap = p => ss(0, 0.035, p) * (1 - ss(0.3, 0.55, p)) * (1 - 0.7 * p); // upstroke, plateau, repolarisation, rest

export function heartDemo() {
  const nx = 256, ny = 160, aspect = 0.55, count = 360, period = 44, lambda = 0.3, TAU = 2 * Math.PI;
  return {
    nx, ny, count,
    meta: { dt: 5, time_unit: 'ms', title: 'Spiral re-entry', geometry: 'ventricle', range: [0, 1] },
    frame(k) {
      const u = new Float32Array(nx * ny);
      const X0 = 0.25 + 0.035 * Math.cos(k * 0.021) + 0.015 * Math.cos(k * 0.067), Y0 = aspect * 0.48 + 0.03 * Math.sin(k * 0.029);
      const cx = Math.cos(TAU * X0), cy = Math.sin(TAU * X0), wt = (TAU * k) / period;
      for (let j = 0; j < ny; j++) {
        const dy = (j / (ny - 1)) * aspect - Y0, e = Math.exp(TAU * dy);
        for (let i = 0; i < nx; i++) {
          const X = i / nx;
          // cylinder -> plane (w = exp(2π(Y + iX))) makes arg(w - w0) continuous across the periodic seam
          const arg = Math.atan2(e * Math.sin(TAU * X) - cy, e * Math.cos(TAU * X) - cx);
          let dx = X - X0; dx -= Math.round(dx);
          const psi = arg + (TAU * Math.hypot(dx, dy)) / lambda - wt;
          const p = -psi / TAU;
          u[j * nx + i] = ap(p - Math.floor(p));
        }
      }
      return u;
    },
  };
}

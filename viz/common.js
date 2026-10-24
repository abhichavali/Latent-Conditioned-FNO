// Shared by both engines: .npy loading, colormap LUTs, and the playback timeline.
import * as THREE from 'three';

// ---------- .npy ----------
export function parseNpy(buf) {
  const u8 = new Uint8Array(buf);
  if (u8[0] !== 0x93 || new TextDecoder().decode(u8.subarray(1, 6)) !== 'NUMPY') throw new Error('not a .npy file');
  const dv = new DataView(buf);
  const v1 = u8[6] === 1;
  const hlen = v1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const off = (v1 ? 10 : 12) + hlen;
  const header = new TextDecoder().decode(u8.subarray(v1 ? 10 : 12, off));
  const descr = header.match(/'descr':\s*'([^']+)'/)[1];
  if (/'fortran_order':\s*True/.test(header)) throw new Error('fortran_order arrays not supported, use np.ascontiguousarray');
  let shape = header.match(/'shape':\s*\(([^)]*)\)/)[1].split(',').filter(s => s.trim()).map(Number);
  // torch-style (T,1,Ny,Nx) -> (T,Ny,Nx)
  shape = [shape[0], ...shape.slice(1).filter(d => d !== 1)];
  const Ctor = { '<f4': Float32Array, '<f8': Float64Array, '|f4': Float32Array, '=f4': Float32Array, '=f8': Float64Array }[descr];
  if (!Ctor) throw new Error(`unsupported dtype ${descr}; save as float32`);
  const n = shape.reduce((a, b) => a * b, 1);
  const raw = new Ctor(buf.slice(off, off + n * Ctor.BYTES_PER_ELEMENT));
  return { data: raw instanceof Float32Array ? raw : Float32Array.from(raw), shape };
}

// (T, Ny, Nx) array -> list of per-frame views (no copy)
export function splitFrames({ data, shape }) {
  if (shape.length !== 3) throw new Error(`expected (T, Ny, Nx), got (${shape})`);
  const [T, ny, nx] = shape, n = nx * ny;
  return { nx, ny, frames: Array.from({ length: T }, (_, t) => data.subarray(t * n, (t + 1) * n)) };
}

// ?data=dir -> fetch dir/<name>.npy for each name (missing files are skipped) and dir/meta.json
export async function loadFromUrl(names) {
  const dir = new URLSearchParams(location.search).get('data');
  if (!dir) return null;
  const out = { meta: {} };
  const get = p => fetch(`${dir.replace(/\/$/, '')}/${p}`).then(r => (r.ok ? r : null));
  await Promise.all(names.map(async k => {
    const r = await get(`${k}.npy`);
    if (r) out[k] = parseNpy(await r.arrayBuffer());
  }));
  const m = await get('meta.json');
  if (m) out.meta = await m.json();
  return out;
}

// Drag & drop: arrays are routed by dimensionality (3D = frames, 2D = static field), .json = meta.
export function onDrop(cb) {
  const veil = document.getElementById('drop');
  addEventListener('dragover', e => { e.preventDefault(); veil?.classList.add('on'); });
  addEventListener('dragleave', e => { if (!e.relatedTarget) veil?.classList.remove('on'); });
  addEventListener('drop', async e => {
    e.preventDefault();
    veil?.classList.remove('on');
    const out = { meta: {} };
    for (const f of e.dataTransfer.files) {
      if (f.name.endsWith('.json')) out.meta = JSON.parse(await f.text());
      else if (f.name.endsWith('.npy')) {
        const a = parseNpy(await f.arrayBuffer());
        out[a.shape.length === 3 ? 'frames' : 'static'] = a;
      }
    }
    cb(out);
  });
}

// ---------- colormaps ----------
export const CMAPS = {
  inferno: ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4'],
  turbo: ['#30123b', '#4145ab', '#4675ed', '#39a2fc', '#1bcfd4', '#24eca6', '#61fc6c', '#a4fc3b', '#d1e834', '#f3c63a', '#fe9b2d', '#f36315', '#d93806', '#b11901', '#7a0402'],
  berlin: ['#a8c8ff', '#4a90d9', '#1b4a7a', '#0d1620', '#5c1f12', '#b8502e', '#ffb59a'], // dark-centred diverging
};

// 256x1 sRGB texture; sample with texture(lut, vec2(t, .5)) to get linear color
export function makeLUT(stops) {
  const cs = stops.map(s => new THREE.Color(s)), d = new Uint8Array(256 * 4);
  const c = new THREE.Color();
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (cs.length - 1), k = Math.min(Math.floor(x), cs.length - 2);
    c.lerpColors(cs[k], cs[k + 1], x - k);
    c.convertLinearToSRGB();
    d.set([c.r * 255, c.g * 255, c.b * 255, 255], i * 4);
  }
  const t = new THREE.DataTexture(d, 256, 1);
  t.colorSpace = THREE.SRGBColorSpace;
  t.magFilter = t.minFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
}

// R32F grid texture; shaders read it with texelFetch (manual bilinear), so no float-filtering extension needed
export function gridTexture(nx, ny, data = new Float32Array(nx * ny)) {
  const t = new THREE.DataTexture(data, nx, ny, THREE.RedFormat, THREE.FloatType);
  t.needsUpdate = true;
  return t;
}

// GLSL: bilinear sample of a gridTexture at uv in [0,1]^2. wrapX makes x periodic (cell i at uv.x = i/nx).
export const GRID_GLSL = /* glsl */`
float gridAt(sampler2D t, vec2 uv, vec2 res, bool wrapX) {
  vec2 p = uv * vec2(wrapX ? res.x : res.x - 1.0, res.y - 1.0);
  ivec2 i = ivec2(floor(p)), m = ivec2(res) - 1;
  vec2 f = fract(p);
  int x0 = wrapX ? (i.x % int(res.x) + int(res.x)) % int(res.x) : clamp(i.x, 0, m.x);
  int x1 = wrapX ? (x0 + 1) % int(res.x) : min(x0 + 1, m.x);
  int y0 = clamp(i.y, 0, m.y), y1 = min(y0 + 1, m.y);
  return mix(mix(texelFetch(t, ivec2(x0, y0), 0).r, texelFetch(t, ivec2(x1, y0), 0).r, f.x),
             mix(texelFetch(t, ivec2(x0, y1), 0).r, texelFetch(t, ivec2(x1, y1), 0).r, f.x), f.y);
}`;

// ---------- timeline ----------
// Owns frames + playback time. Engines only ever see (frameA, frameB, alpha).
export class Player {
  constructor() {
    this.frames = [];
    this.series = []; // optional per-frame scalar drawn behind the scrubber
    this.t = 0;
    this.fps = 12;
    this.dt = 1;
    this.unit = 's';
    this.playing = true;
    const $ = id => document.getElementById(id);
    this.ui = { play: $('play'), scrub: $('scrub'), time: $('time'), speed: $('speed'), spark: $('spark') };
    this.ui.play.onclick = () => this.toggle();
    this.ui.scrub.oninput = () => { this.t = +this.ui.scrub.value; this.playing = false; this.sync(); };
    this.ui.speed.onchange = () => { this.fps = +this.ui.speed.value; };
    addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
      if (e.code === 'Space') { e.preventDefault(); this.toggle(); }
      if (e.code === 'ArrowRight') this.step(1);
      if (e.code === 'ArrowLeft') this.step(-1);
    });
    new ResizeObserver(() => this.drawSeries()).observe(this.ui.spark);
  }
  reset(dt = 1, unit = 's') { this.frames = []; this.series = []; this.t = 0; this.dt = dt; this.unit = unit; this.sync(); }
  push(frame, value) {
    this.frames.push(frame);
    if (value !== undefined) this.series.push(value);
    this.ui.scrub.max = Math.max(0, this.frames.length - 1);
    if (!this._drawQueued) { this._drawQueued = true; requestAnimationFrame(() => { this._drawQueued = false; this.drawSeries(); }); }
  }
  toggle() { this.playing = !this.playing; this.sync(); }
  step(d) { this.playing = false; this.t = Math.max(0, Math.min(this.frames.length - 1, Math.round(this.t) + d)); this.sync(); }
  advance(sec) {
    const last = this.frames.length - 1;
    if (this.playing && last > 0) { this.t += sec * this.fps; if (this.t > last) this.t = 0; }
    this.sync();
  }
  // -> [frameA, frameB, alpha] or null when empty
  sample() {
    const n = this.frames.length;
    if (!n) return null;
    const i = Math.min(Math.floor(this.t), n - 1), j = Math.min(i + 1, n - 1);
    return [this.frames[i], this.frames[j], this.t - i];
  }
  sync() {
    const { play, scrub, time } = this.ui, n = this.frames.length, t = Math.min(this.t, Math.max(n - 1, 0));
    play.textContent = this.playing ? '❚❚' : '▶';
    play.setAttribute('aria-label', this.playing ? 'Pause' : 'Play');
    scrub.value = t;
    const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
    time.textContent = n ? `${fmt(t * this.dt)} / ${fmt((n - 1) * this.dt)} ${this.unit}` : '—';
  }
  drawSeries() {
    const cv = this.ui.spark, s = this.series, dpr = devicePixelRatio;
    cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
    const g = cv.getContext('2d');
    if (s.length < 2) return;
    let lo = Infinity, hi = -Infinity;
    for (const v of s) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const span = hi - lo || 1, W = cv.width, H = cv.height, n = Math.max(this.frames.length - 1, 1);
    g.beginPath();
    s.forEach((v, i) => g[i ? 'lineTo' : 'moveTo']((i / n) * W, H - 3 * dpr - ((v - lo) / span) * (H - 6 * dpr)));
    g.strokeStyle = getComputedStyle(cv).color;
    g.lineWidth = 1.5 * dpr;
    g.shadowColor = g.strokeStyle; g.shadowBlur = 6 * dpr;
    g.stroke();
  }
}

// Wire a <input type=range data-k="name"> panel to a params object; calls onChange(name, value).
export function bindPanel(params, onChange) {
  for (const el of document.querySelectorAll('[data-k]')) {
    const k = el.dataset.k, out = el.parentElement.querySelector('output');
    const show = () => out && (out.textContent = el.type === 'checkbox' ? '' : (+el.value).toFixed(el.step < 1 ? 2 : 0));
    if (el.type === 'checkbox') el.checked = params[k]; else el.value = params[k];
    show();
    el.addEventListener('input', () => {
      params[k] = el.type === 'checkbox' ? el.checked : el.tagName === 'SELECT' ? el.value : +el.value;
      show();
      onChange(k, params[k]);
    });
  }
}

export function status(msg, isError = false) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.classList.toggle('err', isError);
}

// Training-set browser: pick an equation and an initial condition, step through its rollout.
// Data: <base>/index.jsonl, <base>/<id>/{spec.json, u.npy (ic, frame, field, y, x) f16, static.npy (ic, y, x)}.
// Only the chosen IC is fetched (HTTP Range), so it works over an SSH tunnel to the GPU box.
import katex from 'katex';
import { CMAPS, Player, npyHeader, halfToFloat, status } from '../common.js';

const $ = id => document.getElementById(id);
const q = new URLSearchParams(location.search);
const BASE = (q.get('data') ?? '/data').replace(/\/$/, '');
const player = new Player();
player.fps = +$('speed').value;
player.playing = false;

let index = [], cur = null, figs = [], lut = makeLUT(CMAPS.turbo), session = 0;
const specs = new Map(), heads = new Map();

function makeLUT(stops) {
  const rgb = stops.map(s => [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16)));
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (rgb.length - 1), k = Math.min(Math.floor(x), rgb.length - 2), f = x - k;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = rgb[k][c] + (rgb[k + 1][c] - rgb[k][c]) * f;
  }
  return out;
}

async function range(url, a, b) {
  const r = await fetch(url, { headers: { Range: `bytes=${a}-${b}` } });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const buf = await r.arrayBuffer();
  return r.status === 206 ? buf : buf.slice(a, b + 1); // server without Range support sent everything
}

// i-th slice along axis 0 of a .npy -> { data: Float32Array, shape }
async function slice(url, i) {
  if (!heads.has(url)) heads.set(url, npyHeader(await range(url, 0, 1023)));
  const h = heads.get(url), per = h.shape.slice(1).reduce((a, b) => a * b, 1), bytes = h.descr.endsWith('2') ? 2 : 4;
  const a = h.offset + i * per * bytes, buf = await range(url, a, a + per * bytes - 1);
  return { data: bytes === 2 ? halfToFloat(new Uint16Array(buf)) : new Float32Array(buf), shape: h.shape.slice(1) };
}

async function getSpec(id) {
  if (!specs.has(id)) specs.set(id, fetch(`${BASE}/${id}/spec.json`).then(r => r.json()));
  return specs.get(id);
}

async function load(id, ic) {
  const s = ++session;
  status(`loading ${id} · IC ${ic}…`);
  try {
    const spec = await getSpec(id);
    const { data, shape: [T, C, ny, nx] } = await slice(`${BASE}/${id}/u.npy`, ic);
    if (s !== session) return;
    const n = nx * ny, { shift, scale } = spec.norm;
    const fields = spec.fields.map((name, c) => ({ name, frames: [] }));
    for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) {
      const src = data.subarray((t * C + c) * n, (t * C + c + 1) * n), f = new Float32Array(n);
      for (let k = 0; k < n; k++) f[k] = src[k] * scale[c] + shift[c];
      fields[c].frames.push(f);
    }
    for (const f of fields) {
      let lo = Infinity, hi = -Infinity;
      for (const fr of f.frames) for (const v of fr) { if (v < lo) lo = v; if (v > hi) hi = v; }
      f.lo = lo; f.hi = hi;
    }
    let stat = null;
    if (spec.static.length) {
      const a = (await slice(`${BASE}/${id}/static.npy`, ic)).data;
      let lo = Infinity, hi = -Infinity;
      for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
      stat = { name: `${spec.static[0]} (static)`, frames: [a], lo, hi };
    }
    if (s !== session) return;
    cur = { id, ic, spec, fields, stat, nx, ny, T };
    show();
    status(`${T} frames · Δt = ${fmt(spec.frame_dt)}`);
  } catch (e) { if (s === session) status(e.message, true); }
}

const fmt = v => (Math.abs(v) >= 1e3 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(2) : v.toPrecision(3));

function show() {
  const { spec, fields, stat, nx, ny, T, ic } = cur, smp = spec.samples?.[ic] ?? {};
  $('title').textContent = spec.id;
  $('family').textContent = spec.family;
  $('dims').textContent = `${nx}×${ny} · L = ${fmt(spec.L)} · ${spec.fields.length} field${spec.fields.length > 1 ? 's' : ''}`;
  $('eqn').innerHTML = spec.fields.map(f => katex.renderToString(`\\partial_t ${f} = ${spec.latex[f]}`, { displayMode: true, throwOnError: false })).join('');
  $('meta').textContent = [
    `tags   ${spec.tags.join(', ') || '—'}`,
    `frame  Δt ${fmt(spec.frame_dt)} = ${spec.substeps} × dt ${fmt(spec.dt)}`,
    `IC     #${ic} · ${smp.ic_family ?? '?'} · window from frame ${smp.start_frame ?? 0}`,
    `range  ${fields.map(f => `${f.name} [${fmt(f.lo)}, ${fmt(f.hi)}]`).join('  ')}`,
  ].join('\n');
  const all = stat ? [...fields, stat] : fields;
  $('fields').style.setProperty('--cols', all.length);
  $('fields').replaceChildren(...all.map(f => {
    const fig = document.createElement('figure'), cv = document.createElement('canvas'), cap = document.createElement('figcaption');
    cv.width = nx; cv.height = ny;
    cv.className = $('interp').value === 'px' ? 'px' : '';
    cap.innerHTML = `<b>${f.name}</b><span></span>`;
    cv.onmousemove = e => {
      const r = cv.getBoundingClientRect(), i = Math.floor(((e.clientX - r.left) / r.width) * nx), j = Math.floor(((e.clientY - r.top) / r.height) * ny);
      const fr = f.frames.length > 1 ? f.frames[Math.round(player.t)] : f.frames[0];
      cap.lastChild.textContent = `(${i}, ${j}) = ${fmt(fr[j * nx + i])}`;
    };
    cv.onmouseleave = () => (cap.lastChild.textContent = `[${fmt(f.lo)}, ${fmt(f.hi)}]`);
    cv.onmouseleave();
    fig.append(cv, cap);
    return Object.assign(fig, { f, ctx: cv.getContext('2d'), img: new ImageData(nx, ny) });
  }));
  figs = [...$('fields').children];
  const keepT = Math.min(player.t, T - 1);
  player.reset(spec.frame_dt, '');
  fields[0].frames.forEach(fr => {
    let m = 0, m2 = 0;
    for (const v of fr) { m += v; m2 += v * v; }
    m /= fr.length;
    player.push(fr, Math.sqrt(Math.max(m2 / fr.length - m * m, 0)));
  });
  player.t = keepT;
  player.sync();
  history.replaceState(null, '', `?${new URLSearchParams({ ...(q.has('data') ? { data: q.get('data') } : {}), eq: cur.id, ic })}`);
  draw();
}

function draw() {
  if (!cur) return;
  const t = Math.min(player.t, cur.T - 1), i = Math.floor(t), j = Math.min(i + 1, cur.T - 1), al = t - i;
  const perFrame = $('range').value === 'frame', diverging = $('cmap').value === 'berlin';
  for (const { f, ctx, img } of figs) {
    const A = f.frames[Math.min(i, f.frames.length - 1)], B = f.frames[Math.min(j, f.frames.length - 1)], px = img.data;
    let lo = f.lo, hi = f.hi;
    if (perFrame && f.frames.length > 1) {
      lo = Infinity; hi = -Infinity;
      for (let k = 0; k < A.length; k++) { const v = A[k] + (B[k] - A[k]) * al; if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    if (diverging) { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
    const sc = 255 / (hi - lo || 1);
    for (let k = 0; k < A.length; k++) {
      const v = A[k] + (B[k] - A[k]) * al, c = Math.max(0, Math.min(255, ((v - lo) * sc) | 0)) * 3;
      px[k * 4] = lut[c]; px[k * 4 + 1] = lut[c + 1]; px[k * 4 + 2] = lut[c + 2]; px[k * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }
}

let last = performance.now();
(function loop(now) {
  player.advance((now - last) / 1000);
  last = now;
  draw();
  requestAnimationFrame(loop);
})(last);

function fillList() {
  const terms = $('filter').value.toLowerCase().split(/\s+/).filter(Boolean);
  const hit = index.filter(e => terms.every(w => `${e.id} ${e.family} ${e.fields.join(' ')} ${Object.values(e.rhs).join(' ')}`.toLowerCase().includes(w)));
  $('eqs').replaceChildren(...hit.map(e => new Option(`${e.id} · ${e.family.replace(/^[a-z]+\//, '')} · ${e.fields.join(',')}`, e.id)));
  if (cur) $('eqs').value = cur.id;
}

$('filter').oninput = fillList;
$('eqs').onchange = () => load($('eqs').value, +$('ic').value);
$('ic').oninput = () => { $('icv').textContent = $('ic').value; };
$('ic').onchange = () => cur && load(cur.id, +$('ic').value);
$('rand').onclick = () => {
  const e = index[Math.floor(Math.random() * index.length)], ic = Math.floor(Math.random() * (+$('ic').max + 1));
  $('ic').value = ic; $('icv').textContent = ic; load(e.id, ic);
};
$('cmap').onchange = () => { lut = makeLUT(CMAPS[$('cmap').value]); };
$('interp').onchange = () => figs.forEach(f => f.firstChild.classList.toggle('px', $('interp').value === 'px'));

(async () => {
  try {
    const txt = await (await fetch(`${BASE}/index.jsonl`)).text();
    index = txt.split('\n').filter(Boolean).map(l => JSON.parse(l)).sort((a, b) => a.id.localeCompare(b.id));
    if (!index.length) return status('no equations yet', true);
    fillList();
    const id = q.get('eq') ?? index[0].id, ic = +(q.get('ic') ?? 0);
    $('ic').value = ic; $('icv').textContent = ic;
    await load(id, ic);
    if (q.has('t')) { player.t = +q.get('t'); player.sync(); }
  } catch (e) { status(`${BASE}/index.jsonl: ${e.message}`, true); }
})();

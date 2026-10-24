// Cardiac excitation renderer. Knows nothing about Aliev–Panfilov: give it a grid size once,
// then any two transmembrane-potential frames u(x,y) and a blend factor every tick.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { gridTexture, makeLUT, CMAPS, GRID_GLSL } from '../common.js';

// grid surface: f(s, t) -> [x, y, z, gridU, gridV] for s, t in [0,1]
function surface(nu, nv, f, flip = false) {
  const pos = [], grid = [], idx = [];
  for (let j = 0; j <= nv; j++) for (let i = 0; i <= nu; i++) {
    const [x, y, z, gu, gv] = f(i / nu, j / nv);
    pos.push(x, y, z); grid.push(gu, gv);
  }
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const a = j * (nu + 1) + i, c = a + nu + 1;
    flip ? idx.push(a, a + 1, c, a + 1, c + 1, c) : idx.push(a, c, a + 1, a + 1, c, c + 1);
  }
  return { pos, grid, idx, nu, nv };
}

// merge surfaces into one geometry; `weld` parts get their periodic seam (and optional apex row) normals averaged
function build(parts) {
  const pos = [], grid = [], idx = [], offs = [];
  for (const p of parts) {
    const o = pos.length / 3;
    offs.push(o);
    for (const x of p.pos) pos.push(x);
    for (const x of p.grid) grid.push(x);
    for (const i of p.idx) idx.push(i + o);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('aGrid', new THREE.Float32BufferAttribute(grid, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  const n = g.attributes.normal.array, v = new THREE.Vector3();
  const avg = ks => { v.set(0, 0, 0); for (const k of ks) v.add({ x: n[k * 3], y: n[k * 3 + 1], z: n[k * 3 + 2] }); v.normalize(); for (const k of ks) n.set([v.x, v.y, v.z], k * 3); };
  parts.forEach((p, q) => {
    if (!p.weld) return;
    const w = p.nu + 1;
    for (let j = 0; j <= p.nv; j++) avg([offs[q] + j * w, offs[q] + j * w + p.nu]);
    if (p.apex) avg(Array.from({ length: w }, (_, i) => offs[q] + i));
  });
  return g;
}

const ss = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };

// LV-like bullet: closed, slightly pointed apex, widest ~2/3 up, gentle taper to the base, RV crescent on one side
function ventricle(nu = 256, nv = 128) {
  const L = 3, R = 0.78, yb = L / 2;
  const outer = (s, v) => {
    const phi = s * 2 * Math.PI, d = Math.atan2(Math.sin(phi - 0.2), Math.cos(phi - 0.2)); // angle from RV lobe centre
    const lv = Math.sin((Math.min(v / 0.66, 1) * Math.PI) / 2) ** 0.85 * (1 - 0.13 * Math.max(0, (v - 0.66) / 0.34) ** 2);
    const span = ss(0.18, 0.55, v);
    const rv = (0.4 * Math.exp(-((d / 0.95) ** 4)) - 0.05 * Math.exp(-(((Math.abs(d) - 1.15) / 0.16) ** 2))) * span; // lobe + sulci
    const wob = 0.02 * Math.sin(3 * phi + 5 * v) + 0.012 * Math.sin(5 * phi - 3 * v + 1);
    const r = R * lv * (1 + rv + wob);
    return [r * Math.cos(phi), v * L - L / 2, r * Math.sin(phi)];
  };
  const inner = (s, v) => { const [x, y, z] = outer(s, v); return [x * 0.78, yb + (y - yb) * 0.9, z * 0.78]; };
  const lerp = (p, q, t) => p.map((a, i) => a + (q[i] - a) * t);
  return build([
    // apex row samples one grid point, else per-vertex contraction would crack the apex fan open
    { ...surface(nu, nv, (s, v) => [...outer(s, v), v ? s : 0, v]), weld: true, apex: true },
    { ...surface(nu, nv, (s, v) => [...inner(s, v), v ? s : 0, v], true), weld: true, apex: true },
    { ...surface(nu, 2, (s, t) => [...lerp(outer(s, 1), inner(s, 1), t), s, 1]), weld: true },
  ]);
}

function sheet(lx, ly, nu = 256, nv = 256) {
  const k = Math.min(3.4 / lx, 3.4 / ly), W = lx * k, H = ly * k, th = 0.14;
  const P = (s, v, d) => [(s - 0.5) * W, -0.35 * (2 * s - 1) ** 2 - 0.15 * (2 * v - 1) ** 2 + d * th, (v - 0.5) * H];
  const side = (n, edge, flip) => surface(n, 1, (t, h) => { const [s, v] = edge(t); return [...P(s, v, h - 0.5), s, v]; }, flip);
  return build([
    surface(nu, nv, (s, v) => [...P(s, v, 0.5), s, v]),
    surface(nu, nv, (s, v) => [...P(s, v, -0.5), s, v], true),
    side(nu, t => [t, 0]), side(nu, t => [t, 1], true), side(nv, t => [0, t], true), side(nv, t => [1, t]),
  ]);
}

export class HeartEngine {
  params = { cmap: 'ember', glow: 1.0, front: 2.5, contract: 0.04, geometry: 'ventricle', autorotate: true };

  constructor(container) {
    const r = (this.renderer = new THREE.WebGLRenderer({ powerPreference: 'high-performance' }));
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.0;
    container.appendChild(r.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#050307');
    this.scene.environment = new THREE.PMREMGenerator(r).fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.1;

    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    this.camera.position.set(0, 0.4, 8.2);
    this.controls = new OrbitControls(this.camera, r.domElement);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: 0.06, autoRotateSpeed: 0.5, minDistance: 2.5, maxDistance: 16 });

    const key = new THREE.DirectionalLight('#ffd6b8', 2.6), rim = new THREE.DirectionalLight('#5f9dff', 1.2), fill = new THREE.DirectionalLight('#ff6a5a', 0.3);
    key.position.set(4, 5, 1.5); rim.position.set(-4, 2, -5); fill.position.set(-3, -2, 3);
    this.scene.add(key, rim, fill);

    this.heart = new THREE.Group();
    this.scene.add(this.heart);

    this.luts = { ember: makeLUT(CMAPS.inferno), clinical: makeLUT(CMAPS.turbo) };
    this.u = {
      uA: { value: gridTexture(1, 1) }, uB: { value: gridTexture(1, 1) }, uMix: { value: 0 },
      uRes: { value: new THREE.Vector2(1, 1) }, uRange: { value: new THREE.Vector2(0, 1) }, uWrap: { value: true },
      uContract: { value: 0 }, uGlow: { value: 1 }, uFront: { value: 1 }, uCmap: { value: 0 }, uLut: { value: this.luts.ember },
    };
    this.material = this.tissueMaterial();

    const rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(r, rt);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(1, 1), 0.45, 0.1, 1.6));
    this.composer.addPass(new OutputPass());

    this.resize = () => {
      const w = container.clientWidth, h = container.clientHeight;
      r.setSize(w, h);
      this.composer.setPixelRatio(r.getPixelRatio());
      this.composer.setSize(w, h);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    };
    new ResizeObserver(this.resize).observe(container);
    this.resize();
    this.applyParams();

    const clock = new THREE.Clock();
    r.setAnimationLoop(() => {
      this.onFrame?.(Math.min(clock.getDelta(), 0.1));
      this.controls.autoRotate = this.params.autorotate;
      this.controls.update();
      this.composer.render();
    });
  }

  // geometry 'ventricle' treats x as periodic (circumferential), 'sheet' as a flat patch of size lx × ly
  setGrid(nx, ny, { geometry = this.params.geometry, lx = nx, ly = ny } = {}) {
    this.params.geometry = geometry;
    for (const k of ['uA', 'uB']) { this.u[k].value.dispose(); this.u[k].value = gridTexture(nx, ny); }
    this.u.uRes.value.set(nx, ny);
    this.u.uWrap.value = geometry === 'ventricle';
    this.heart.clear();
    this.mesh?.geometry.dispose();
    this.mesh = new THREE.Mesh(geometry === 'ventricle' ? ventricle() : sheet(lx, ly), this.material);
    this.heart.add(this.mesh);
    this.heart.rotation.set(geometry === 'ventricle' ? -0.5 : 0.9, 0, geometry === 'ventricle' ? -0.6 : 0); // ventricle: base up-back, apex down-left-forward
  }

  setFrames(a, b, alpha) {
    const { uA, uB } = this.u;
    if (uA.value.image.data !== a) { uA.value.image.data = a; uA.value.needsUpdate = true; }
    if (uB.value.image.data !== b) { uB.value.image.data = b; uB.value.needsUpdate = true; }
    this.u.uMix.value = alpha;
  }

  set range([lo, hi]) { this.u.uRange.value.set(lo, hi > lo ? hi : lo + 1); }

  applyParams() {
    const p = this.params, u = this.u;
    u.uGlow.value = p.glow;
    u.uFront.value = p.front;
    u.uContract.value = p.contract;
    u.uCmap.value = p.cmap === 'clinical' ? 1 : 0;
    u.uLut.value = this.luts[p.cmap];
  }

  tissueMaterial() {
    const m = new THREE.MeshPhysicalMaterial({
      color: '#76101a', roughness: 0.38, metalness: 0, clearcoat: 0.8, clearcoatRoughness: 0.15,
      sheen: 0.1, sheenColor: new THREE.Color('#c84a4a'), sheenRoughness: 0.45, side: THREE.DoubleSide,
    });
    m.onBeforeCompile = sh => {
      Object.assign(sh.uniforms, this.u);
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', `#include <common>
          attribute vec2 aGrid;
          uniform sampler2D uA, uB; uniform float uMix, uContract; uniform vec2 uRes, uRange; uniform bool uWrap;
          varying float vU, vGrad, vRise, vApex; varying vec2 vGrid;
          ${GRID_GLSL}
          float field(vec2 uv) { return mix(gridAt(uA, uv, uRes, uWrap), gridAt(uB, uv, uRes, uWrap), uMix); }`)
        .replace('#include <begin_vertex>', /* glsl */`
          #include <begin_vertex>
          float span = uRange.y - uRange.x;
          vec2 e = 1.0 / uRes;
          float u = clamp((field(aGrid) - uRange.x) / span, 0.0, 1.0);
          float g = length(vec2(field(aGrid + vec2(e.x, 0.0)) - field(aGrid - vec2(e.x, 0.0)),
                                field(aGrid + vec2(0.0, e.y)) - field(aGrid - vec2(0.0, e.y)))) * 0.5 / span;
          float rise = (gridAt(uB, aGrid, uRes, uWrap) - gridAt(uA, aGrid, uRes, uWrap)) / span;
          vU = u; vGrad = g; vRise = rise; vGrid = aGrid;
          vApex = uWrap ? smoothstep(0.0, 0.06, aGrid.y) : 1.0; // the grid pinches to a point at the apex; don't glow there
          transformed *= 1.0 - uContract * u;`);

      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', /* glsl */`#include <common>
          uniform sampler2D uLut; uniform float uGlow, uFront; uniform int uCmap;
          varying float vU, vGrad, vRise, vApex; varying vec2 vGrid;
          vec3 bumpN(vec3 p, vec3 n, vec2 dH, float fd) { // perturbNormalArb, procedural height
            vec3 sx = normalize(dFdx(p)), sy = normalize(dFdy(p)), r1 = cross(sy, n), r2 = cross(n, sx);
            float det = dot(sx, r1) * fd;
            return normalize(abs(det) * n - sign(det) * (dH.x * r1 + dH.y * r2));
          }`)
        .replace('#include <clipping_planes_fragment>', /* glsl */`
          #include <clipping_planes_fragment>
          // helical fibre striations (integer x-frequencies keep the periodic seam invisible)
          float q = vGrid.x * 64.0 + vGrid.y * 30.0 + 0.25 * sin(6.2831 * (vGrid.x * 7.0 - vGrid.y * 3.0));
          float fib = (0.5 + 0.5 * sin(6.2831 * q)) * (1.0 - smoothstep(0.25, 0.7, fwidth(q)));
          float blob = 0.5 + 0.5 * sin(6.2831 * (vGrid.x * 5.0 + vGrid.y * 2.0)) * sin(6.2831 * vGrid.y * 4.0);`)
        .replace('#include <color_fragment>', /* glsl */`
          #include <color_fragment>
          vec3 lut = texture(uLut, vec2(vU, 0.5)).rgb;
          vec3 tissue = diffuseColor.rgb * (0.75 + 0.2 * fib + 0.15 * blob);
          diffuseColor.rgb = uCmap == 1 ? mix(tissue, lut, 0.85) : tissue * (1.0 - 0.4 * vU);`)
        .replace('#include <normal_fragment_begin>', /* glsl */`
          #include <normal_fragment_begin>
          normal = bumpN(-vViewPosition, normal, vec2(dFdx(fib), dFdy(fib)) * 0.012, faceDirection);`)
        .replace('#include <emissivemap_fragment>', /* glsl */`
          #include <emissivemap_fragment>
          // white-hot leading (depolarising) edge, cooling through orange to deep red as u decays along the plateau
          float heat = pow(vU, 5.0);
          float front = smoothstep(0.08, 0.3, vGrad) * smoothstep(0.0, 0.05, vRise) * exp(-pow((vU - 0.55) / 0.25, 2.0));
          totalEmissiveRadiance = (uCmap == 1 ? lut * 0.3 : texture(uLut, vec2(0.3 + 0.62 * heat, 0.5)).rgb * heat * 1.5) * uGlow
                                + vec3(1.0, 0.9, 0.8) * front * uFront;
          totalEmissiveRadiance *= vApex;`)
        .replace('#include <opaque_fragment>', 'outgoingLight = min(outgoingLight, vec3(6.0)); // no specular fireflies in the bloom\n#include <opaque_fragment>');
    };
    return m;
  }
}

// Shallow-water / tsunami renderer. Knows nothing about solvers: give it a bed b(x,y) once,
// then any two water-depth frames h(x,y) and a blend factor every tick.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Sky } from 'three/addons/objects/Sky.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { gridTexture, makeLUT, CMAPS } from '../common.js';

const SIZE = 100; // scene units across the longest side of the domain
const smooth = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };

// Water surface height. eta = b + h is interpolated over wet corners only (interpolating h itself
// smears fake water up steep shores), so the shoreline is where that surface meets the bed.
// uWaveExag scales elevation above sea level offshore and flow depth onshore; the two agree at the shoreline.
// returns (displayY, eta, h, bed) in meters
const SURF_GLSL = /* glsl */`
uniform sampler2D uHA, uHB, uBed;
uniform float uMix, uWaveExag, uHEps, uLandFade;
uniform vec2 uRes, uDomain;
vec4 surf(vec2 uv) {
  vec2 p = clamp(uv, 0.0, 1.0) * (uRes - 1.0), f = fract(p);
  ivec2 i0 = ivec2(floor(p)), m = ivec2(uRes) - 1;
  float b = 0.0, etaSum = 0.0, wSum = 0.0;
  for (int k = 0; k < 4; k++) {
    ivec2 o = ivec2(k & 1, k >> 1), c = min(i0 + o, m);
    float w = mix(1.0 - f.x, f.x, float(o.x)) * mix(1.0 - f.y, f.y, float(o.y));
    float hk = mix(texelFetch(uHA, c, 0).r, texelFetch(uHB, c, 0).r, uMix), bk = texelFetch(uBed, c, 0).r;
    b += w * bk;
    if (hk > uHEps) { etaSum += w * (bk + hk); wSum += w; }
  }
  float eta = wSum > 0.0 ? etaSum / wSum : b, h = eta - b;
  // onshore the exaggeration relaxes to true scale with height, so floods don't stand up as walls on steep ground
  float y = b < 0.0 ? eta * uWaveExag : b + h * mix(uWaveExag, 1.0, smoothstep(0.0, uLandFade, b));
  y = h > uHEps ? max(y, b + 0.25 * h) : b - uHEps;
  return vec4(y, eta, h, b);
}`;

const NOISE_GLSL = /* glsl */`
float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash2(i), hash2(i + vec2(1, 0)), u.x), mix(hash2(i + vec2(0, 1)), hash2(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + 17.0; a *= 0.5; } return s; }`;

export class TsunamiEngine {
  params = { vexag: 4, wexag: 1, sun: 0.3, clarity: 1, foam: 1, mode: 'real', autorotate: false };

  constructor(container) {
    const r = (this.renderer = new THREE.WebGLRenderer({ powerPreference: 'high-performance' }));
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 0.85;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(r.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.5, 4000);
    this.camera.position.set(-66, 34, 80);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.target.set(6, 0, -4);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: 0.06, maxPolarAngle: Math.PI * 0.48, minDistance: 8, maxDistance: 400 });

    this.world = new THREE.Group(); // geometry lives in meters; this group maps meters -> scene units
    this.scene.add(this.world);

    // sky -> environment map (reflections) + matching backdrop gradient
    this.sky = new Sky();
    this.sky.scale.setScalar(50);
    Object.assign(this.sky.material.uniforms, { turbidity: { value: 3 }, rayleigh: { value: 2.2 }, mieCoefficient: { value: 0.004 }, mieDirectionalG: { value: 0.86 } });
    this.skyScene = new THREE.Scene().add(this.sky);
    this.pmrem = new THREE.PMREMGenerator(r);
    this.bgCanvas = Object.assign(document.createElement('canvas'), { width: 2, height: 512 });
    this.scene.background = new THREE.CanvasTexture(this.bgCanvas);
    this.scene.background.colorSpace = THREE.SRGBColorSpace;

    this.sunDir = new THREE.Vector3();
    this.sun = new THREE.DirectionalLight(0xffffff, 3);
    this.sun.castShadow = true;
    Object.assign(this.sun.shadow.camera, { left: -80, right: 80, top: 80, bottom: -80, near: 1, far: 500 });
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.04;
    this.scene.add(this.sun, this.sun.target);

    // uniforms shared by every water/terrain shader
    this.u = {
      uHA: { value: gridTexture(1, 1) }, uHB: { value: gridTexture(1, 1) }, uBed: { value: gridTexture(1, 1) },
      uMix: { value: 0 }, uRes: { value: new THREE.Vector2(1, 1) }, uDomain: { value: new THREE.Vector2(1, 1) },
      uWaveExag: { value: 1 }, uHEps: { value: 1e-3 }, uLandFade: { value: 1 },
      uTime: { value: 0 }, uClarity: { value: 10 }, uFoamDepth: { value: 1 }, uFoamSlope: { value: 0.05 }, uFoamScale: { value: 10 },
      uRippleLen: { value: 20 }, uMode: { value: 0 }, uEtaMax: { value: 1 }, uLut: { value: makeLUT(CMAPS.berlin) },
      uInvScale: { value: new THREE.Vector3(1, 1, 1) }, uSunDir: { value: this.sunDir }, uSunColor: { value: new THREE.Color() },
      uSkyLight: { value: 1 }, uCausticLen: { value: 30 },
    };

    // bloom only catches sun glints on the water
    const rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(r, rt);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass((this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.25, 0.5, 1.6)));
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
    this.updateSun();

    const clock = new THREE.Clock();
    r.setAnimationLoop(() => {
      const dt = Math.min(clock.getDelta(), 0.1);
      this.u.uTime.value += dt;
      this.onFrame?.(dt);
      this.controls.autoRotate = this.params.autorotate;
      this.controls.update();
      this.composer.render();
    });
  }

  // b: Float32Array (ny*nx) bed elevation in meters, sea level = 0. opts: { lx, ly, dry }
  setBathymetry(b, nx, ny, { lx = nx, ly = ny, dry } = {}) {
    this.world.clear();
    this.u.uBed.value.dispose();
    this.u.uBed.value = gridTexture(nx, ny, b);
    this.u.uHA.value = gridTexture(nx, ny);
    this.u.uHB.value = gridTexture(nx, ny);
    this.u.uRes.value.set(nx, ny);
    this.u.uDomain.value.set(lx, ly);
    this.dims = { nx, ny, lx, ly };

    let lo = Infinity, hi = -Infinity;
    for (const v of b) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const relief = Math.max(hi - lo, 1e-6), maxDepth = Math.max(-lo, relief * 0.05), L = Math.max(lx, ly);
    this.stats = { lo, hi, relief, maxDepth };
    this.autoVexag = Math.max(1, (0.13 * L) / relief);
    this.u.uHEps.value = dry ?? 1e-4 * relief;
    this.u.uLandFade.value = 0.05 * relief;
    this.u.uFoamScale.value = L / 500;
    this.u.uRippleLen.value = L / 90;
    this.u.uCausticLen.value = L / 160;

    // mesh resolution: upsample small grids so the surface stays smooth
    const up = n => Math.min(1024, n < 400 ? n * 2 : n);
    const mx = up(nx), my = up(ny);
    const bed = (u, v) => {
      const x = u * (nx - 1), y = v * (ny - 1), i = Math.min(Math.floor(x), nx - 2), j = Math.min(Math.floor(y), ny - 2), fx = x - i, fy = y - j, k = j * nx + i;
      return (b[k] * (1 - fx) + b[k + 1] * fx) * (1 - fy) + (b[k + nx] * (1 - fx) + b[k + nx + 1] * fx) * fy;
    };

    // surface grid (shared by terrain and water)
    const n = mx * my, grid = new Float32Array(n * 2), tpos = new Float32Array(n * 3), wpos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    for (let j = 0, k = 0; j < my; j++) for (let i = 0; i < mx; i++, k++) {
      const u = i / (mx - 1), v = j / (my - 1), x = (u - 0.5) * lx, z = (v - 0.5) * ly;
      grid.set([u, v], k * 2);
      tpos.set([x, bed(u, v), z], k * 3);
      wpos.set([x, 0, z], k * 3);
    }
    const index = [];
    for (let j = 0; j < my - 1; j++) for (let i = 0; i < mx - 1; i++) {
      const a = j * mx + i, c = a + mx;
      index.push(a, c, a + 1, a + 1, c, c + 1);
    }

    const terrainGeo = new THREE.BufferGeometry();
    terrainGeo.setAttribute('position', new THREE.BufferAttribute(tpos, 3));
    terrainGeo.setAttribute('aGrid', new THREE.BufferAttribute(grid, 2));
    terrainGeo.setIndex(index);
    terrainGeo.computeVertexNormals();
    this.colorTerrain(tpos, terrainGeo.attributes.normal.array, col, lx / (mx - 1));
    terrainGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const terrain = new THREE.Mesh(terrainGeo, this.terrainMaterial());
    terrain.castShadow = terrain.receiveShadow = true;

    const waterGeo = new THREE.BufferGeometry();
    waterGeo.setAttribute('position', new THREE.BufferAttribute(wpos, 3));
    waterGeo.setAttribute('aGrid', new THREE.BufferAttribute(grid, 2));
    waterGeo.setIndex(index);
    waterGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), L * 4);
    this.water = new THREE.Mesh(waterGeo, this.waterMaterial());
    this.water.receiveShadow = true;

    this.world.add(terrain, this.water, ...this.walls(bed, mx, my, lo - 0.18 * relief));
    this.applyParams();
  }

  // h frames for the two bracketing timesteps + blend factor in [0,1]
  setFrames(a, b, alpha) {
    const { uHA, uHB } = this.u;
    if (uHA.value.image.data !== a) { uHA.value.image.data = a; uHA.value.needsUpdate = true; }
    if (uHB.value.image.data !== b) { uHB.value.image.data = b; uHB.value.needsUpdate = true; }
    this.u.uMix.value = alpha;
  }

  applyParams() {
    const p = this.params, d = this.dims;
    if (!d) return;
    const S = SIZE / Math.max(d.lx, d.ly);
    this.world.scale.set(S, S * p.vexag, S);
    this.u.uInvScale.value.set(1 / S, 1 / (S * p.vexag), 1 / S);
    this.u.uWaveExag.value = p.wexag;
    this.u.uMode.value = p.mode === 'eta' ? 1 : 0;
    const clarity = p.clarity * 0.15 * this.stats.maxDepth;
    this.u.uClarity.value = clarity;
    this.u.uFoamDepth.value = 0.005 * this.stats.relief * p.foam;
    this.u.uFoamSlope.value = 0.06 / Math.max(p.foam, 0.05);
    if (this.water) this.water.material.attenuationDistance = clarity * S * p.vexag;
  }

  set etaMax(v) { this.u.uEtaMax.value = Math.max(v * 0.5, 1e-6); } // saturate the run-up peaks, keep offshore crests readable

  updateSun() {
    const elev = THREE.MathUtils.lerp(1.5, 62, this.params.sun), e = THREE.MathUtils.degToRad(elev);
    this.sunDir.setFromSphericalCoords(1, Math.PI / 2 - e, -Math.PI / 2 - 0.55);
    this.sky.material.uniforms.sunPosition.value.copy(this.sunDir);
    this.envRT?.dispose();
    this.envRT = this.pmrem.fromScene(this.skyScene, 0.02);
    this.scene.environment = this.envRT.texture;
    this.scene.environmentIntensity = THREE.MathUtils.lerp(0.35, 1, smooth(0, 30, elev));

    const warm = smooth(2, 35, elev);
    this.sun.color.lerpColors(new THREE.Color('#ff9a52'), new THREE.Color('#fff6ea'), warm);
    this.sun.intensity = THREE.MathUtils.lerp(2.2, 3.4, warm);
    this.sun.position.copy(this.sunDir).multiplyScalar(200);
    this.u.uSunColor.value.copy(this.sun.color).multiplyScalar(this.sun.intensity / 3);
    this.u.uSkyLight.value = THREE.MathUtils.lerp(0.35, 1.1, warm);

    // backdrop: screen-space gradient tuned to the sun height
    const mix = (a, b) => '#' + new THREE.Color(a).lerp(new THREE.Color(b), warm).getHexString();
    const g = this.bgCanvas.getContext('2d'), grad = g.createLinearGradient(0, 0, 0, 512);
    grad.addColorStop(0, mix('#0d1424', '#2a5a8c'));
    grad.addColorStop(0.55, mix('#3b3a5a', '#7fa9cf'));
    grad.addColorStop(0.85, mix('#c9876a', '#cfe1ee'));
    grad.addColorStop(1, mix('#e9b08a', '#e6eff5'));
    g.fillStyle = grad;
    g.fillRect(0, 0, 2, 512);
    this.scene.background.needsUpdate = true;
  }

  colorTerrain(pos, nrm, col, cell) {
    const { hi, maxDepth } = this.stats, land = Math.max(hi, 1e-6);
    const C = s => new THREE.Color(s);
    const deep = C('#3a3428'), sand = C('#b9a37a'), beach = C('#e8d6ad'), grass = C('#6b7d3a'), forest = C('#2f4a22'), rock = C('#6e655b'), peak = C('#a49a8e');
    const c = new THREE.Color(), t = new THREE.Color();
    for (let k = 0; k < pos.length / 3; k++) {
      const x = pos[k * 3], z = pos[k * 3 + 1], y = pos[k * 3 + 2], slope = 1 - nrm[k * 3 + 1];
      const nz = Math.sin(x * 0.0123 + Math.sin(y * 0.0071) * 3) * 0.5 + Math.sin(y * 0.0193 + x * 0.004) * 0.5; // cheap wobble
      if (z < 0) {
        c.lerpColors(sand, deep, smooth(0, 0.6, -z / maxDepth));
        c.lerp(beach, 1 - smooth(0, 0.04, -z / maxDepth));
      } else {
        const hz = z / land + nz * 0.04;
        c.lerpColors(beach, grass, smooth(0.0, 0.06, hz));
        c.lerp(forest, smooth(0.15, 0.45, hz) * 0.85);
        c.lerp(rock, smooth(0.65, 0.9, hz));
        c.lerp(peak, smooth(0.9, 1.0, hz));
        c.lerp(rock, smooth(0.25, 0.5, slope * 6 * (cell / 20)));
      }
      t.copy(c).multiplyScalar(0.92 + 0.16 * nz);
      col.set([t.r, t.g, t.b], k * 3);
    }
  }

  // Diorama cut-away: earth strata on the sides, plus translucent water walls that follow the surface.
  walls(bed, mx, my, base) {
    const { lx, ly } = this.dims;
    const edges = [
      Array.from({ length: mx }, (_, i) => [i / (mx - 1), 0]),
      Array.from({ length: mx }, (_, i) => [i / (mx - 1), 1]),
      Array.from({ length: my }, (_, j) => [0, j / (my - 1)]),
      Array.from({ length: my }, (_, j) => [1, j / (my - 1)]),
    ];
    const R = 48, earth = [], water = [];
    const soil = [new THREE.Color('#2b211a'), new THREE.Color('#4a3a2b'), new THREE.Color('#5d4a36'), new THREE.Color('#3a2e24')];
    const c = new THREE.Color();
    for (const e of edges) {
      const pos = [], cols = [], idx = [], wpos = [], wgrid = [], wtop = [], widx = [];
      e.forEach(([u, v], s) => {
        const x = (u - 0.5) * lx, z = (v - 0.5) * ly, top = bed(u, v);
        for (let r = 0; r <= R; r++) {
          const y = base + ((top - base) * r) / R, band = (y - base) / (top - base + 1e-9);
          const layer = Math.abs(Math.floor((y / this.stats.relief) * 9 + Math.sin(s * 0.05) * 0.6)) % soil.length;
          c.copy(soil[layer]).multiplyScalar(0.55 + 0.45 * band);
          pos.push(x, y, z); cols.push(c.r, c.g, c.b);
        }
        wpos.push(x, 0, z, x, 0, z); wgrid.push(u, v, u, v); wtop.push(0, 1);
        if (s) {
          for (let r = 0; r < R; r++) { const a = (s - 1) * (R + 1) + r, b = a + R + 1; idx.push(a, b, a + 1, a + 1, b, b + 1); }
          const a = (s - 1) * 2; widx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
        }
      });
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
      g.setIndex(idx);
      g.computeVertexNormals();
      earth.push(g);
      const w = new THREE.BufferGeometry();
      w.setAttribute('position', new THREE.Float32BufferAttribute(wpos, 3));
      w.setAttribute('aGrid', new THREE.Float32BufferAttribute(wgrid, 2));
      w.setAttribute('aTop', new THREE.Float32BufferAttribute(wtop, 1));
      w.setIndex(widx);
      w.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Math.max(lx, ly) * 4);
      water.push(w);
    }
    const earthMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, side: THREE.DoubleSide });
    const waterMat = new THREE.ShaderMaterial({
      uniforms: this.u, transparent: true, side: THREE.DoubleSide, depthWrite: false,
      vertexShader: SURF_GLSL + /* glsl */`
        attribute vec2 aGrid; attribute float aTop;
        varying float vDepth, vH;
        void main() {
          vec4 s = surf(aGrid);
          float y = aTop > 0.5 ? s.x : s.w;
          vDepth = s.x - y; vH = s.z;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position.x, y, position.z, 1.0);
        }`,
      fragmentShader: /* glsl */`
        uniform float uClarity, uHEps, uSkyLight;
        varying float vDepth, vH;
        void main() {
          if (vH < uHEps) discard;
          float T = exp(-vDepth / (uClarity * 2.0));
          vec3 col = mix(vec3(0.004, 0.03, 0.06), vec3(0.05, 0.32, 0.38), T) * uSkyLight;
          col += vec3(0.5, 0.8, 0.9) * exp(-vDepth / (uClarity * 0.04)) * 0.6 * uSkyLight;
          gl_FragColor = vec4(col, 0.86);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
    return [...earth.map(g => Object.assign(new THREE.Mesh(g, earthMat), { receiveShadow: true })), ...water.map(g => new THREE.Mesh(g, waterMat))];
  }

  terrainMaterial() {
    const m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.93, metalness: 0, envMapIntensity: 0.7 });
    m.onBeforeCompile = sh => {
      Object.assign(sh.uniforms, this.u);
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nattribute vec2 aGrid; varying vec2 vGrid; varying vec3 vObj;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvGrid = aGrid; vObj = position;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec2 vGrid; varying vec3 vObj;\nuniform float uTime, uCausticLen, uClarity; uniform vec3 uSunColor;\n' + SURF_GLSL + /* glsl */`
          // tileable water caustic (after Dave Hoskins)
          float caustic(vec2 p, float t) {
            vec2 i = p; float c = 1.0;
            for (int n = 0; n < 4; n++) {
              float tt = t * (1.0 - 3.5 / float(n + 1));
              i = p + vec2(cos(tt - i.x) + sin(tt + i.y), sin(tt - i.y) + cos(tt + i.x));
              c += 1.0 / length(vec2(p.x / (sin(i.x + tt) / 0.005), p.y / (cos(i.y + tt) / 0.005)));
            }
            c = 1.17 - pow(c / 4.0, 1.4);
            return min(pow(abs(c), 8.0), 3.0);
          }`)
        .replace('#include <color_fragment>', '#include <color_fragment>\nvec4 sw = surf(vGrid);\nbool wet = sw.z > uHEps;\nif (wet) diffuseColor.rgb *= 0.72;')
        .replace('#include <opaque_fragment>', /* glsl */`
          if (wet && uMode == 0) {
            float fade = exp(-sw.z / (uClarity * 0.6)) * smoothstep(0.0, uClarity * 0.05, sw.z);
            outgoingLight += diffuseColor.rgb * uSunColor * caustic(mod(vObj.xz / uCausticLen, 6.2831853) - 250.0, uTime * 0.6) * fade * 1.6;
          }
          #include <opaque_fragment>`);
      sh.fragmentShader = 'uniform int uMode;\n' + sh.fragmentShader;
    };
    return m;
  }

  waterMaterial() {
    const m = new THREE.MeshPhysicalMaterial({
      color: 0xffffff, roughness: 0.05, metalness: 0, transmission: 1, ior: 1.333, thickness: 1,
      attenuationColor: new THREE.Color('#5fd0c8'), attenuationDistance: 1, specularIntensity: 1, envMapIntensity: 1.1,
    });
    m.onBeforeCompile = sh => {
      Object.assign(sh.uniforms, this.u);
      const varyings = 'varying vec3 vObj, vN; varying float vH, vEta, vBed, vSlope;';
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', `#include <common>\nattribute vec2 aGrid;\n${varyings}\n${SURF_GLSL}`)
        .replace('#include <beginnormal_vertex>', /* glsl */`
          vec4 sC = surf(aGrid);
          vec2 e = 1.0 / (uRes - 1.0);
          vec4 sL = surf(aGrid - vec2(e.x, 0.0)), sR = surf(aGrid + vec2(e.x, 0.0));
          vec4 sD = surf(aGrid - vec2(0.0, e.y)), sU = surf(aGrid + vec2(0.0, e.y));
          // dry neighbours would fake a cliff at the shoreline; fall back to the centre value
          #define WET(s, c) (s.z > uHEps ? s : c)
          sL = WET(sL, sC); sR = WET(sR, sC); sD = WET(sD, sC); sU = WET(sU, sC);
          vec2 cell = uDomain * e * 2.0;
          vec3 objectNormal = normalize(vec3(-(sR.x - sL.x) / cell.x, 1.0, -(sU.x - sD.x) / cell.y));
          vN = objectNormal;
          vSlope = length(vec2((sR.y - sL.y) / cell.x, (sU.y - sD.y) / cell.y));
          vH = sC.z; vEta = sC.y; vBed = sC.w;`)
        .replace('#include <begin_vertex>', 'vec3 transformed = vec3(position.x, sC.x, position.z);\nvObj = transformed;');

      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>\n${varyings}\n${NOISE_GLSL}
          uniform float uHEps, uTime, uClarity, uFoamDepth, uFoamSlope, uFoamScale, uRippleLen, uEtaMax, uSkyLight;
          uniform int uMode; uniform sampler2D uLut; uniform vec3 uInvScale, uSunDir, uSunColor;`)
        .replace('#include <clipping_planes_fragment>', /* glsl */`
          #include <clipping_planes_fragment>
          if (vH < uHEps) discard;
          float mud = smoothstep(-0.02, 0.02, vBed / max(uClarity, 1e-6) * 0.1); // flood water over land is silty
          // foam: thin sheet flow, steep fronts, and shoaling crests (breaking starts around eta/h ~ 0.4)
          float foamAmt = max((1.0 - smoothstep(0.0, uFoamDepth, vH)) * 0.85, smoothstep(uFoamSlope * 0.35, uFoamSlope, vSlope));
          foamAmt = max(foamAmt, smoothstep(0.12, 0.45, vEta / max(vH, 1e-3) * uFoamSlope / 0.06) * step(vBed, 0.0));
          vec2 fp = vObj.xz / uFoamScale;
          float fn = fbm(fp + vec2(uTime * 0.15, 0.0)) * 0.6 + fbm(fp * 3.1 - uTime * 0.1) * 0.4;
          float foam = uMode == 0 ? smoothstep(1.0 - foamAmt, 1.25 - foamAmt, fn) * foamAmt : 0.0;`)
        .replace('#include <color_fragment>', /* glsl */`
          #include <color_fragment>
          if (uMode == 1) {
            float val = vBed < 0.0 ? vEta : vH; // offshore: surface elevation, onshore: flow depth
            diffuseColor.rgb = texture(uLut, vec2(0.5 + 0.5 * clamp(val / uEtaMax, -1.0, 1.0), 0.5)).rgb;
            totalEmissiveRadiance = diffuseColor.rgb * 0.9; // readable at any sun angle
            diffuseColor.rgb *= 0.6;
          }
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.92, 0.95, 0.96), foam);`)
        .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = uMode == 1 ? 0.45 : mix(roughnessFactor, 0.8, foam);')
        .replace('#include <normal_fragment_begin>', /* glsl */`
          #include <normal_fragment_begin>
          // small wind ripples in gusty patches, faded out before they alias
          vec2 rip = vec2(0.0);
          float px = length(fwidth(vObj.xz));
          for (int k = 0; k < 8; k++) {
            float fk = 6.2831 / uRippleLen * pow(1.47, float(k));
            float ang = float(k) * 2.39996 + 0.3;
            vec2 dir = vec2(cos(ang), sin(ang));
            float aa = 1.0 - smoothstep(0.2, 0.7, px * fk / 6.2831);
            rip += dir * 0.05 / (1.0 + float(k) * 0.35) * cos(dot(dir, vObj.xz) * fk + uTime * sqrt(9.81 * fk) * 1.4 + float(k) * 1.7) * aa;
          }
          rip *= 0.35 + 1.3 * fbm(vObj.xz / (uRippleLen * 6.0) + uTime * 0.03);
          rip *= (1.0 - foam) * (uMode == 1 ? 0.2 : 1.0) * smoothstep(0.0, uFoamDepth * 2.0, vH);
          // ripples perturb the world-space normal so vertical exaggeration doesn't amplify them
          vec3 nW = normalize(normalize(vN * uInvScale) + vec3(-rip.x, 0.0, -rip.y));
          normal = normalize((viewMatrix * vec4(nW, 0.0)).xyz);`)
        .replace('material.transmission = transmission;', 'material.transmission = uMode == 1 ? 0.0 : transmission * (1.0 - foam);')
        .replace('material.thickness = thickness;', 'material.thickness = vH;')
        .replace('material.attenuationColor = attenuationColor;', 'material.attenuationColor = mix(attenuationColor, vec3(0.55, 0.42, 0.25), mud);')
        .replace('#include <opaque_fragment>', /* glsl */`
          if (uMode == 0) {
            // light scattered back out of the water column, plus a glow through backlit crests
            vec3 V = normalize(vViewPosition);
            float F = 0.02 + 0.98 * pow(1.0 - saturate(dot(normal, V)), 5.0);
            float T = exp(-vH / uClarity);
            vec3 scat = mix(vec3(0.012, 0.10, 0.17), vec3(0.09, 0.07, 0.04), mud);
            vec3 viewRay = normalize(vWorldPosition - cameraPosition);
            float back = pow(saturate(dot(viewRay, uSunDir) * 0.5 + 0.5), 6.0);
            float crest = smoothstep(0.0, uEtaMax, vEta) * (1.0 - T);
            outgoingLight += (scat * (1.0 - T) + vec3(0.05, 0.4, 0.35) * crest * back * 0.6 * uSunColor) * (1.0 - F) * (1.0 - foam) * uSkyLight;
          }
          #include <opaque_fragment>`);
    };
    return m;
  }
}

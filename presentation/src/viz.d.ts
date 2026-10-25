// Types for the repo's viz engines (../viz, plain JS), imported through the '@viz' alias in vite.config.ts.
// Only what the deck uses.
declare module '@viz/common.js' {
  export type Npy = { data: Float32Array; shape: number[] }
  export function parseNpy(buf: ArrayBuffer): Npy
}

declare module '@viz/tsunami/engine.js' {
  import type * as THREE from 'three'
  export class TsunamiEngine {
    constructor(container: HTMLElement)
    params: { vexag: number; wexag: number; sun: number; clarity: number; foam: number; mode: 'real' | 'eta'; autorotate: boolean }
    renderer: THREE.WebGLRenderer
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    controls: { enabled: boolean; target: THREE.Vector3; update(): void; dispose(): void }
    world: THREE.Group // metres -> scene units (vertical exaggeration included)
    autoVexag: number
    onFrame?: (dt: number) => void
    resize(): void
    setBathymetry(b: Float32Array, nx: number, ny: number, opts?: { lx?: number; ly?: number; dry?: number }): void
    setFrames(a: Float32Array, b: Float32Array, alpha: number): void
    applyParams(): void
    updateSun(): void
    set etaMax(v: number)
  }
}

declare module '@viz/heart/engine.js' {
  import type * as THREE from 'three'
  export class HeartEngine {
    constructor(container: HTMLElement)
    params: { cmap: string; glow: number; front: number; contract: number; geometry: 'ventricle' | 'sheet'; autorotate: boolean }
    renderer: THREE.WebGLRenderer
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    controls: { enabled: boolean; target: THREE.Vector3; update(): void; dispose(): void }
    heart: THREE.Group
    mesh: THREE.Mesh
    onFrame?: (dt: number) => void
    resize(): void
    setGrid(nx: number, ny: number, opts?: { geometry?: 'ventricle' | 'sheet'; lx?: number; ly?: number }): void
    setFrames(a: Float32Array, b: Float32Array, alpha: number): void
    applyParams(): void
    set range(r: [number, number])
  }
}

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const here = (p: string) => decodeURIComponent(new URL(p, import.meta.url).pathname)

export default defineConfig({
  base: './', // relative asset paths: dist/ works from any URL, e.g. /presentation/dist/ next to /viz/
  plugins: [react()],
  resolve: {
    // The demo imports the repo's viz engines (../viz). They import 'three' by bare name, which would otherwise
    // resolve from viz/ (no node_modules there), so point it at the vendored copy: no CDN at runtime.
    alias: [
      { find: '@viz', replacement: here('../viz') },
      { find: /^three$/, replacement: here('./node_modules/three/build/three.module.js') },
      { find: /^three\/addons\//, replacement: here('./node_modules/three/examples/jsm/') },
    ],
  },
  build: { chunkSizeWarningLimit: 2000 }, // one local bundle (three.js + deck), no network: splitting buys nothing
  server: { fs: { allow: ['..'] } }, // the appendix imports a figure from ../baseline_FNO, the demo imports ../viz
})

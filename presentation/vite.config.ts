import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: './', // relative asset paths: dist/ works from any URL, e.g. /presentation/dist/ next to /viz/
  plugins: [react()],
  server: { fs: { allow: ['..'] } }, // the appendix imports a figure from ../baseline_FNO
})

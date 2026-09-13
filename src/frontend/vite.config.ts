import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Deployed behind a prefix when BASE_PATH is set (e.g. "/benchmark/"), at the
// origin root otherwise. Vite propagates this to import.meta.env.BASE_URL, so
// the router, the API client and the event stream all read one value.
const base = process.env.BASE_PATH || '/';

export default defineConfig({
  base,
  plugins: [react()],
  build: { outDir: 'dist', sourcemap: false },
  server: {
    port: 5173,
    // In development the SPA and the control plane are separate processes; in
    // the image they are one origin, so nothing here changes the deployed setup.
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/healthz': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
});

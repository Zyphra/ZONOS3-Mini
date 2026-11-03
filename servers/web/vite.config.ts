// Engine library build: dist/engine.js exports `engine` and `localFetch` from src/client.ts and spawns the worker
// from a sibling dist/worker.js. The studio loads it from /web-engine/engine.js (?api=web).
import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    copyPublicDir: false,
    modulePreload: false,
    target: 'es2022',
    rollupOptions: {
      input: { engine: 'src/client.ts' },
      // Vite drops entry exports in app builds (preserveEntrySignatures: false) unless told otherwise.
      preserveEntrySignatures: 'strict',
      output: { entryFileNames: '[name].js', chunkFileNames: '[name].js', assetFileNames: '[name][extname]' },
    },
  },
  worker: {
    format: 'es',
    rollupOptions: { output: { entryFileNames: '[name].js', chunkFileNames: '[name].js' } },
  },
  test: { environment: 'node', include: ['tests/**/*.test.ts'] },
});

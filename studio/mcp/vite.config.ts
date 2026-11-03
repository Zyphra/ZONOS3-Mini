import preact from '@preact/preset-vite';
import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

// The player widget: one self-contained HTML file (dist/player.html) served as the ui:// resource.
export default defineConfig({
  root: 'widget',
  plugins: [preact(), viteSingleFile()],
  build: { outDir: '../dist', emptyOutDir: false, rollupOptions: { input: 'player.html' } },
});

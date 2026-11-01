import { defineConfig, loadEnv } from 'vite';
import preact from '@preact/preset-vite';
import { fileURLToPath } from 'node:url';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = process.env.ZONOS_SERVER || env.ZONOS_SERVER;
  const proxy = target
    ? Object.fromEntries(['/tts', '/v1', '/dub', '/projects', '/media'].map((p) => [p, { target, changeOrigin: true, secure: true }]))
    : undefined;
  return {
    base: '/studio/',
    plugins: [preact()],
    build: {
      outDir: fileURLToPath(new URL('../python/zonos3/server/studio', import.meta.url)),
      emptyOutDir: true,
      target: 'es2022',
    },
    server: { proxy },
    preview: { proxy },
  };
});

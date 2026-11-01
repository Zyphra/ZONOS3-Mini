import { defineConfig, loadEnv } from 'vite';
import preact from '@preact/preset-vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = process.env.ZONOS_SERVER || env.ZONOS_SERVER;
  const proxy = target
    ? Object.fromEntries(['/v1', '/v2'].map((p) => [p, { target, changeOrigin: true, secure: true }]))
    : undefined;
  return {
    base: '/',
    plugins: [preact()],
    build: {
      outDir: 'dist',
      emptyOutDir: true,
      target: 'es2022',
    },
    server: { proxy },
    preview: { proxy },
  };
});

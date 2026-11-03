import { defineConfig, loadEnv, type Connect, type Plugin } from 'vite';
import preact from '@preact/preset-vite';
import sirv from 'sirv';

/**
 * The in-browser server (`?api=web`) on the dev and preview servers: its engine build at /web-engine
 * (`pnpm --dir ../servers/web build`) and its exported models at /models (`pnpm --dir ../servers/web export`).
 */
function webEngine(): Plugin {
  const mount = (use: Connect.Server['use']) => {
    use('/web-engine', sirv('../servers/web/dist', { dev: true }));
    use('/models', sirv('../servers/web/public/models', { dev: true }));
  };
  return {
    name: 'zonos3-web-engine',
    configureServer(server) {
      mount(server.middlewares.use.bind(server.middlewares));
    },
    configurePreviewServer(server) {
      mount(server.middlewares.use.bind(server.middlewares));
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = process.env.ZONOS_SERVER || env.ZONOS_SERVER;
  const proxy = target
    ? Object.fromEntries(['/v1', '/v2'].map((p) => [p, { target, changeOrigin: true, secure: true }]))
    : undefined;
  return {
    base: '/',
    plugins: [preact(), webEngine()],
    build: {
      outDir: 'dist',
      emptyOutDir: true,
      target: 'es2022',
    },
    server: { proxy },
    preview: { proxy },
  };
});

import { readdirSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { defineConfig, loadEnv, searchForWorkspaceRoot, type Connect, type Plugin } from 'vite';
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

/**
 * `emptyOutDir`, minus the committed `dist/web-engine/`: that engine build comes from `pnpm build:web`, not from
 * this build, so a plain `pnpm build` keeps it.
 */
function cleanOutDir(keep: string[]): Plugin {
  let outDir = '';
  return {
    name: 'zonos3-clean-out-dir',
    apply: 'build',
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
    },
    buildStart() {
      let entries: string[] = [];
      try {
        entries = readdirSync(outDir);
      } catch {
        return; // no dist yet
      }
      for (const entry of entries) if (!keep.includes(entry)) rmSync(join(outDir, entry), { recursive: true, force: true });
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
    plugins: [preact(), webEngine(), cleanOutDir(['web-engine'])],
    build: {
      outDir: 'dist',
      emptyOutDir: false,
      target: 'es2022',
      // Three pages: the chooser at /, the full studio at /studio and the lite voice-cloning page at /lite (the
      // servers' studio serving rule maps /<page> to <page>.html; so do the dev and preview servers).
      rolldownOptions: {
        input: Object.fromEntries(['index', 'studio', 'lite'].map((page) => [page, resolve(import.meta.dirname, `${page}.html`)])),
      },
    },
    // src/lib/languages.ts imports the language registry from the model package.
    server: { proxy, fs: { allow: [searchForWorkspaceRoot(import.meta.dirname), resolve(import.meta.dirname, '../model/src/zonos3/registries')] } },
    preview: { proxy },
  };
});

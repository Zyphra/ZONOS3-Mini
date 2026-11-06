import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
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

/**
 * /sw.js, at build time only (the dev server has no service worker): src/sw.js with its precache list (studio.html and
 * every assets/ file it reaches: its chunks, their imports, CSS and fonts, plus /theme-boot.js) and a version hashing
 * studio.html, theme-boot.js and that list (asset names carry content hashes).
 */
function serviceWorker(page: string): Plugin {
  let root = '';
  return {
    name: 'zonos3-service-worker',
    apply: 'build',
    configResolved(config) {
      root = config.root;
    },
    // writeBundle, not generateBundle: the HTML pages are only in the bundle once vite:build-html has emitted them.
    writeBundle(options, bundle) {
      const text = (fileName: string) => {
        const file = bundle[fileName];
        if (!file) return '';
        if (file.type === 'chunk') return file.code;
        return typeof file.source === 'string' ? file.source : new TextDecoder().decode(file.source);
      };
      const assets = Object.keys(bundle).filter((fileName) => fileName.startsWith('assets/'));
      // Breadth-first over files that mention an asset's (content-hashed, so unique) name.
      const reached = new Set([`${page}.html`]);
      for (const fileName of reached) {
        const source = text(fileName);
        for (const asset of assets) if (!reached.has(asset) && source.includes(basename(asset))) reached.add(asset);
      }
      // The unhashed theme-boot.js every page loads: precached, its content in the version.
      const precache = [...[...reached].map((fileName) => `/${fileName}`), '/theme-boot.js'].sort();
      const version = createHash('sha256')
        .update(text(`${page}.html`))
        .update(readFileSync(resolve(root, 'public/theme-boot.js')))
        .update(precache.join('\n'))
        .digest('hex')
        .slice(0, 12);
      const sw = readFileSync(resolve(root, 'src/sw.js'), 'utf8')
        .replace('__ZVS_VERSION__', JSON.stringify(version))
        .replace('__ZVS_PRECACHE__', JSON.stringify(precache, null, 2));
      writeFileSync(join(options.dir ?? resolve(root, 'dist'), 'sw.js'), sw);
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
    plugins: [preact(), webEngine(), cleanOutDir(['web-engine']), serviceWorker('studio')],
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

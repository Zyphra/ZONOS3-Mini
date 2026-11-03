// Runs harness/parity.html in headless Chrome (WebGPU) against public/goldens; one line per check, exit 1 on failure.
//   node harness/parity.mjs [--only a,b] [--port 5199]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { launchChrome } from './chrome.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const only = arg('--only', '');
const port = Number(arg('--port', '5199'));

// /models/* and /goldens/* straight from public/ (Vite's public dir would also serve them, but with SPA fallbacks).
const statics = { '/models/': path.join(root, 'public/models'), '/goldens/': path.join(root, 'public/goldens') };
const staticPlugin = {
  name: 'parity-statics',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const url = decodeURIComponent((req.url ?? '').split('?')[0]);
      for (const [prefix, dir] of Object.entries(statics)) {
        if (!url.startsWith(prefix)) continue;
        const file = path.join(dir, url.slice(prefix.length));
        if (!file.startsWith(dir) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
          res.statusCode = 404;
          res.end('not found');
          return;
        }
        res.setHeader('Content-Length', fs.statSync(file).size);
        res.setHeader('Content-Type', 'application/octet-stream');
        fs.createReadStream(file).pipe(res);
        return;
      }
      next();
    });
  },
};

const server = await createServer({
  root, configFile: path.join(root, 'vite.config.ts'), publicDir: false, logLevel: 'warn',
  optimizeDeps: { entries: ['harness/parity.html'] },
  server: { port, strictPort: true, host: '127.0.0.1' }, plugins: [staticPlugin],
});
await server.listen();
const browser = await launchChrome();
let failed = false;
try {
  const page = await browser.newPage();
  page.on('console', m => { if (m.type() === 'error') console.error('[page]', m.text()); });
  page.on('pageerror', e => console.error('[page error]', e.message));
  await page.goto(`http://127.0.0.1:${port}/harness/parity.html${only ? `?only=${only}` : ''}`);
  await page.waitForSelector('#result[data-done]', { timeout: 0 });
  const results = JSON.parse(await page.$eval('#result', el => el.textContent ?? '{}'));
  for (const [name, r] of Object.entries(results)) {
    console.log(`${r.ok ? 'PASS' : 'FAIL'} ${name} ${JSON.stringify(r.metric)}${r.error ? '\n  ' + r.error : ''}`);
    if (!r.ok) failed = true;
  }
  if (!Object.keys(results).length) { console.log('no checks ran'); failed = true; }
} finally {
  await browser.close();
  await server.close();
}
process.exit(failed ? 1 : 0);

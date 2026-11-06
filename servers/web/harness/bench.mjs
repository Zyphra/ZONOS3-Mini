// Engine benchmark: runs harness/bench.html in local headless Chrome (WebGPU) or in Chrome on an Android phone over
// `adb reverse`, and prints one line per measurement.
//   node harness/bench.mjs [--mode e2e|micro] [--runs 3] [--port 5199] [--android SERIAL [--browser PACKAGE]]
// e2e: engine init, then streamed /v2/generate runs (time to first audio, real-time factor) of a plain and a CFG request.
// micro: backbone decode step (single slot / CFG pair), codec chunk and host iSTFT+resampler medians.
// `http://127.0.0.1` is a secure context on the phone too (WebGPU, Cache Storage); models cross the tunnel once.
// --browser: the Android browser package that opens the page (default com.android.chrome).
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { launchChrome } from './chrome.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const mode = arg('--mode', 'e2e');
const runs = Number(arg('--runs', '3'));
const port = Number(arg('--port', '5199'));
const android = arg('--android', '');
const browserPkg = arg('--browser', 'com.android.chrome');
const ADB = process.env.ADB ?? 'adb';
if (mode !== 'e2e' && mode !== 'micro') { console.error(`bench: unknown --mode ${mode}`); process.exit(2); }

// /models/* and /goldens/* straight from public/ (as harness/parity.mjs).
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

const { promise: done, resolve: finish } = Promise.withResolvers();
const median = a => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const fmt = v => (typeof v === 'number' && Number.isFinite(v) ? String(Math.round(v * 100) / 100) : v ?? '-');

/** Prints a result posted by bench.ts; returns whether it is a success. */
function report(r) {
  if (r.error) { console.log(`error ${r.error}`); return false; }
  if (r.mode === 'micro') {
    console.log(`adapter=${r.adapter ?? '-'}`);
    console.log(`micro bb_step_ms=${fmt(r.bb_step_ms)} bb_pair_ms=${fmt(r.bb_pair_ms)} zac_chunk_ms=${fmt(r.zac_chunk_ms)} ` +
      `istft_ms=${fmt(r.istft_ms)} gemm_plan=${r.gemm_plan ?? '-'}`);
    return true;
  }
  console.log(`init_ms=${fmt(r.initMs)} adapter=${r.adapter ?? '-'}`);
  const cases = [...new Set(r.runs.map(x => x.case))];
  for (const x of r.runs)
    console.log(`run ${x.case} #${x.i} ttfa_ms=${fmt(x.ttfaMs)} total_ms=${fmt(x.totalMs)} audio_s=${fmt(x.audioS)} rtf=${fmt(x.rtf)}`);
  for (const c of cases) {
    const rs = r.runs.filter(x => x.case === c);
    console.log(`median ${c} ttfa_ms=${fmt(median(rs.map(x => x.ttfaMs)))} rtf=${fmt(median(rs.map(x => x.rtf)))}`);
  }
  return true;
}

const benchPlugin = {
  name: 'bench-endpoints',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const url = (req.url ?? '').split('?')[0];
      if (req.method !== 'POST' || (url !== '/bench/log' && url !== '/bench/result')) return next();
      const chunks = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => {
        const body = Buffer.concat(chunks).toString('utf8');
        res.statusCode = 204;
        res.end();
        if (url === '/bench/log') { console.log(body); return; }
        let r;
        try { r = JSON.parse(body); } catch { r = { error: 'unparseable result: ' + body }; }
        finish(report(r));
      });
    });
  },
};

// No HMR: a source edit during a (long, phone) run must not reload the page mid-measurement.
const server = await createServer({
  root, configFile: path.join(root, 'vite.config.ts'), publicDir: false, logLevel: 'warn',
  optimizeDeps: { entries: ['harness/bench.html'] },
  server: { port, strictPort: true, host: '127.0.0.1', hmr: false, watch: null }, plugins: [staticPlugin, benchPlugin],
});
await server.listen();
const url = `http://127.0.0.1:${port}/harness/bench.html?mode=${mode}&runs=${runs}`;
let ok = false;
if (!android) {
  const browser = await launchChrome();
  try {
    const page = await browser.newPage();
    page.on('console', m => { if (m.type() === 'error') console.error('[page]', m.text()); });
    page.on('pageerror', e => console.error('[page error]', e.message));
    await page.goto(url);
    ok = await done;
  } finally {
    await browser.close();
    await server.close();
  }
} else {
  const adb = (...a) => execFileSync(ADB, ['-s', android, ...a], { stdio: ['ignore', 'pipe', 'inherit'] }).toString();
  adb('reverse', `tcp:${port}`, `tcp:${port}`);
  try {
    adb('shell', 'input', 'keyevent', 'KEYCODE_WAKEUP');
    // adb shell joins its arguments into one remote command line: quote the URL (it holds '&').
    adb('shell', 'am', 'start', '-a', 'android.intent.action.VIEW', '-d', `'${url}'`, browserPkg);
    ok = await done;
  } finally {
    try { adb('reverse', '--remove', `tcp:${port}`); } catch { /* the device went away */ }
    await server.close();
  }
}
process.exit(ok ? 0 : 1);

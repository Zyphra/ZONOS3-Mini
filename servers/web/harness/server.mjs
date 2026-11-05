// HTTP front of the in-browser server, so HTTP clients (api/tests/conformance, curl) can reach it:
//   node harness/server.mjs [--port 1920]
// Headless Chrome (WebGPU) opens harness/page.html, which loads dist/engine.js; every /v1 and /v2 request is replayed
// through the page's localFetch and streamed back. Static files: studio/dist at / (the ggml studio serving rule),
// dist at /web-engine/, public/models at /models/, public/goldens at /goldens/. CORS and preflights as
// servers/ggml/src/server/http.cpp.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome } from './chrome.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const repo = path.resolve(root, '../..');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const port = Number(arg('--port', '1920'));

const EXPOSED = 'X-Audio-Sample-Rate, X-Audio-Channels, X-Audio-Format, X-Control-Span-Follow, X-Seed';
const MOUNTS = [
  ['/web-engine/', path.join(root, 'dist')],
  ['/models/', path.join(root, 'public/models')],
  ['/goldens/', path.join(root, 'public/goldens')],
  ['/harness/', here],
];
const STUDIO = path.join(repo, 'studio/dist');
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.woff2': 'font/woff2', '.wasm': 'application/wasm',
  '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.ico': 'image/x-icon',
};

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Expose-Headers', EXPOSED);
}

function sendError(res, status, detail) {
  const body = JSON.stringify({ detail });
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

/** A file inside `dir` (no traversal), or null. */
function fileIn(dir, rel) {
  const target = path.resolve(dir, '.' + path.sep + rel);
  if (target !== dir && !target.startsWith(dir + path.sep)) return null;
  try { return fs.statSync(target).isFile() ? target : null; } catch { return null; }
}

function sendFile(req, res, file, cacheControl) {
  const size = fs.statSync(file).size;
  res.writeHead(200, {
    'content-type': TYPES[path.extname(file).toLowerCase()] ?? 'application/octet-stream',
    'content-length': size, 'cache-control': cacheControl,
  });
  if (req.method === 'HEAD') return res.end();
  fs.createReadStream(file).pipe(res);
}

/** Static routes; false when nothing matched (the caller answers 404 JSON). */
function serveStatic(req, res, pathname) {
  if (req.method !== 'GET' && req.method !== 'HEAD') return false;
  for (const [prefix, dir] of MOUNTS) {
    if (!pathname.startsWith(prefix)) continue;
    const f = fileIn(dir, pathname.slice(prefix.length));
    if (!f) return false;
    sendFile(req, res, f, 'no-store');
    return true;
  }
  // The studio serving rule (install_studio): GET / is index.html (the chooser); /<path> is the file, else the page
  // <path>.html (/studio, /lite); pages (*.html) are no-store, any other file is immutable.
  if (pathname === '/') {
    const f = fileIn(STUDIO, 'index.html');
    if (!f) { sendError(res, 404, 'The studio is not built (pnpm --dir studio build).'); return true; }
    sendFile(req, res, f, 'no-store, max-age=0');
    return true;
  }
  const f = fileIn(STUDIO, pathname.slice(1)) ?? fileIn(STUDIO, pathname.slice(1) + '.html');
  if (!f) return false;
  sendFile(req, res, f, f.endsWith('.html') ? 'no-store, max-age=0' : 'public, max-age=31536000, immutable');
  return true;
}

// ---------------------------------------------------------------------------------------------- the browser bridge

const browser = await launchChrome();
const page = await browser.newPage();
page.on('console', m => { if (m.type() === 'error' || m.type() === 'warn') console.error('[page]', m.text()); });
page.on('pageerror', e => console.error('[page error]', e.message));

let nextId = 1;
const streams = new Map();   // id -> { onStart, onChunk, onEnd }
await page.exposeFunction('__zstart', (id, status, headers) => streams.get(id)?.onStart(status, headers));
await page.exposeFunction('__zchunk', (id, b64) => streams.get(id)?.onChunk(Buffer.from(b64, 'base64')));
await page.exposeFunction('__zend', (id, error) => streams.get(id)?.onEnd(error));

/** Replays one request through window.zfetch, streaming the response back through the exposed callbacks. */
function bridge(method, pathname, headers, body) {
  const id = nextId++;
  const b64 = body.length ? body.toString('base64') : null;
  page.evaluate(async (id, method, path, headers, b64) => {
    const toB64 = (u8) => {
      let s = '';
      for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000));
      return btoa(s);
    };
    try {
      const ctl = new AbortController();
      (window.__zaborts ??= new Map()).set(id, ctl);
      const init = { method, headers, signal: ctl.signal };
      if (b64 !== null) init.body = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
      const r = await window.zfetch(path, init);
      const h = {};
      r.headers.forEach((v, k) => { h[k] = v; });
      await window.__zstart(id, r.status, h);
      if (r.body) {
        const reader = r.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value.length) await window.__zchunk(id, toB64(value));
        }
      }
      await window.__zend(id, null);
    } catch (e) {
      await window.__zend(id, String(e && e.message || e));
    } finally {
      window.__zaborts.delete(id);
    }
  }, id, method, pathname, headers, b64).catch(e => streams.get(id)?.onEnd(String(e)));
  return id;
}

async function waitReady() {
  for (;;) {
    const s = await page.evaluate(() => ({ ready: !!window.zready, fatal: window.zfatal ?? null, status: document.getElementById('status')?.textContent }));
    if (s.fatal) throw new Error('engine failed: ' + s.fatal);
    if (s.ready) return;
    await new Promise(r => setTimeout(r, 500));
  }
}

// ---------------------------------------------------------------------------------------------- HTTP server

const server = http.createServer((req, res) => {
  const url = new URL(req.url ?? '/', 'http://localhost');
  const pathname = decodeURIComponent(url.pathname);
  cors(res);
  if (req.method === 'OPTIONS') {
    if (!req.headers['access-control-request-method']) {
      res.setHeader('Allow', 'GET, HEAD, POST, PUT, DELETE, PATCH');
      return sendError(res, 405, 'Method Not Allowed');
    }
    res.setHeader('Access-Control-Allow-Methods', 'DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT');
    if (req.headers['access-control-request-headers']) res.setHeader('Access-Control-Allow-Headers', req.headers['access-control-request-headers']);
    res.setHeader('Access-Control-Max-Age', '600');
    res.writeHead(200, { 'content-type': 'text/plain; charset=utf-8' });
    return res.end('OK');
  }
  const top = pathname.split('/')[1];
  if (top !== 'v1' && top !== 'v2') {
    if (!serveStatic(req, res, pathname)) sendError(res, 404, 'Not Found');
    return;
  }
  const chunks = [];
  req.on('data', c => chunks.push(c));
  req.on('end', async () => {
    try {
      await (engineReady ??= waitReady());   // a page that (re)loads answers after its engine is up
    } catch (e) {
      engineReady = null;
      return sendError(res, 503, String(e.message ?? e));
    }
    const headers = {};
    for (const [k, v] of Object.entries(req.headers)) {
      if (['host', 'connection', 'content-length', 'accept-encoding', 'transfer-encoding'].includes(k)) continue;
      headers[k] = Array.isArray(v) ? v.join(', ') : v;
    }
    let started = false;
    const id = bridge(req.method, url.pathname + url.search, headers, Buffer.concat(chunks));
    streams.set(id, {
      onStart(status, h) {
        started = true;
        const out = { ...h };
        delete out['content-length'];
        res.writeHead(status, out);
      },
      onChunk(buf) { res.write(buf); },
      onEnd(error) {
        streams.delete(id);
        if (error) console.error(`harness: ${req.method} ${pathname}: ${error}`);
        if (!started) return sendError(res, 500, error ?? 'no response');
        res.end();
      },
    });
    // A client that hangs up mid-stream aborts the generation (localFetch posts the worker's abort).
    res.on('close', () => {
      if (res.writableEnded) return;
      page.evaluate(id => window.__zaborts?.get(id)?.abort(), id).catch(() => {});
    });
  });
});

let engineReady = null;
// A reload of the page (e.g. the engine rebuilt under it) restarts the engine: requests wait for it again.
page.on('framenavigated', f => { if (f === page.mainFrame()) engineReady = null; });
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', resolve); });
await page.goto(`http://127.0.0.1:${port}/harness/page.html`);
console.log(`harness: loading the engine (models from ${path.join(root, 'public/models')}) …`);
await (engineReady ??= waitReady());
console.log(`harness: ready on http://127.0.0.1:${port}`);

const shutdown = async () => { server.close(); await browser.close(); process.exit(0); };
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

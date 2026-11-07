// The engine API's request dispatch: the v2 routes this server serves, 404 {"detail":"Not Found"} for every other path,
// and HttpError / error rendering (servers/ggml/src/server/http.cpp guarded + the 404 filler).
import { NO_STEERING } from '../request/steer';
import { capabilities } from './capabilities';
import { alignRoute, editRoute, transcribeRoute } from './edit';
import type { Engine } from './engine';
import { generateRoute, speechRoute } from './generate';
import { fail, HttpError, parseBody, sendJson, type EngineRequest, type Sink } from './http';
import { Speakers } from './speakers';

export class Api {
  readonly speakers: Speakers;

  constructor(private e: Engine, now = Date.now() / 1000) {
    this.speakers = new Speakers(e, now);
  }

  /** Serves one request into `sink`; never throws (errors are answered as JSON when nothing was sent yet).
   *  POST /v2/edit[/...] with `X-Keepalive: 1` takes the keepalive wire mode (model/src/zonos3/server/keepalive.py). */
  async handle(req: EngineRequest, sink: Sink, aborted: () => boolean = () => false): Promise<void> {
    if (keepaliveApplies(req)) return this.keepalive(req, sink, aborted);
    return this.serve(req, sink, aborted);
  }

  /** 200 + a space at once and every KEEPALIVE_MS, then the route's JSON; an error status travels in the body as
   *  {"detail", "status_code"}. */
  private async keepalive(req: EngineRequest, sink: Sink, aborted: () => boolean): Promise<void> {
    let status = 500;
    const parts: Uint8Array[] = [];
    const capture: Sink = { start: s => { status = s; }, chunk: d => { parts.push(d.slice()); }, end: () => {} };
    sink.start(200, { 'content-type': 'application/json', 'cache-control': 'no-store', 'x-keepalive': '1' });
    const space = new Uint8Array([0x20]);
    sink.chunk(space);
    const timer = setInterval(() => sink.chunk(space), KEEPALIVE_MS);
    try {
      await this.serve(req, capture, aborted);
    } finally {
      clearInterval(timer);
    }
    let payload = new Uint8Array(parts.reduce((a, p) => a + p.length, 0));
    let o = 0;
    for (const p of parts) { payload.set(p, o); o += p.length; }
    if (status >= 400) {
      const text = new TextDecoder().decode(payload);
      let detail: unknown;
      try {
        const j = JSON.parse(text);
        detail = j && typeof j === 'object' && !Array.isArray(j) ? j.detail ?? null : text;
      } catch {
        detail = text;
      }
      payload = new TextEncoder().encode(JSON.stringify({ detail, status_code: status }));
    }
    sink.chunk(payload);
    sink.end();
  }

  private async serve(req: EngineRequest, sink: Sink, aborted: () => boolean): Promise<void> {
    let started = false;
    const guarded: Sink = {
      start: (status, headers) => { started = true; sink.start(status, headers); },
      chunk: d => sink.chunk(d),
      end: () => sink.end(),
    };
    try {
      await this.route(req, guarded, aborted);
    } catch (err) {
      if (started) {
        console.error('engine api:', err);
        sink.end();
        return;
      }
      if (err instanceof HttpError) sendJson(sink, err.body, err.status);
      else sendJson(sink, { detail: err instanceof Error ? err.message : String(err) }, 500);
    }
  }

  private async route(req: EngineRequest, sink: Sink, aborted: () => boolean): Promise<void> {
    const e = this.e, spk = this.speakers;
    const path = req.path.split('?')[0];
    const m = req.method.toUpperCase();
    const get = m === 'GET' || m === 'HEAD';
    if (get && path === '/v2/health') return sendJson(sink, { status: 'ok' });
    if (get && path === '/v1/models')
      return sendJson(sink, { object: 'list', data: [{ id: 'zonos3', object: 'model', created: 0, owned_by: 'zyphra', root: e.modelFile }] });
    if (get && path === '/v2/capabilities') return sendJson(sink, capabilities(e, spk.checkpointModes()));
    if (get && path === '/v2/controls') {
      if (!e.steer) fail(404, `${NO_STEERING}.`);
      return sendJson(sink, e.controlsDoc);
    }
    if (path === '/v2/speakers') {
      if (get) return spk.list(req, sink);
      if (m === 'POST') return spk.create(req, sink, Date.now() / 1000);
    }
    const sp = /^\/v2\/speakers\/([^/]+)\/(preview|embedding)$/.exec(path);
    if (sp && get) {
      const id = decodeURIComponent(sp[1]);
      return sp[2] === 'preview' ? spk.preview(req, sink, id) : spk.embedding(req, sink, id);
    }
    if (m === 'POST' && path === '/v2/generate') return generateRoute(e, spk, req, sink, aborted);
    if (m === 'POST' && path === '/v1/audio/speech') return speechRoute(e, spk, req, sink, aborted);
    if (m === 'POST' && path === '/v2/edit/align') return alignRoute(e, req, sink);
    if (m === 'POST' && path === '/v2/edit') return editRoute(e, req, sink, aborted);
    if (m === 'POST' && path === '/v2/edit/transcribe') transcribeRoute(e, req);
    // phonemize.cpp without a lexicon: the body is validated (422), then 404.
    if (m === 'POST' && path === '/v2/phonemize') {
      parseBody(req, 'PhonemizeRequest');
      fail(404, 'This server has no pronunciation lexicon.');
    }
    fail(404, 'Not Found');
  }
}

const KEEPALIVE_MS = 10_000;
const KEEPALIVE_PATHS = ['/v2/edit'];

function keepaliveApplies(req: EngineRequest): boolean {
  if (req.method.toUpperCase() !== 'POST' || req.headers['x-keepalive'] !== '1') return false;
  const path = req.path.split('?')[0];
  return KEEPALIVE_PATHS.some(p => path === p || path.startsWith(p + '/'));
}

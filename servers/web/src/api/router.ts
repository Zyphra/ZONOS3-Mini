// The engine API's request dispatch: the v2 routes this server serves, 404 {"detail":"Not Found"} for every other path,
// and HttpError / error rendering (servers/ggml/src/server/http.cpp guarded + the 404 filler).
import { capabilities, dubCapabilities } from './capabilities';
import type { Engine } from './engine';
import { generateRoute, speechRoute } from './generate';
import { fail, HttpError, parseBody, sendJson, type EngineRequest, type Sink } from './http';
import { Speakers } from './speakers';

export class Api {
  readonly speakers: Speakers;

  constructor(private e: Engine, now = Date.now() / 1000) {
    this.speakers = new Speakers(e, now);
  }

  /** Serves one request into `sink`; never throws (errors are answered as JSON when nothing was sent yet). */
  async handle(req: EngineRequest, sink: Sink, aborted: () => boolean = () => false): Promise<void> {
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
      if (!e.steer) fail(404, 'This server has no steering bundle (place steer.gguf next to the backbone or pass --steer).');
      return sendJson(sink, e.controlsDoc);
    }
    if (get && path === '/v2/dub/capabilities') return sendJson(sink, dubCapabilities(e));
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
    // phonemize.cpp without a lexicon: the body is validated (422), then 404.
    if (m === 'POST' && path === '/v2/phonemize') {
      parseBody(req, 'PhonemizeRequest');
      fail(404, 'This server has no pronunciation lexicon.');
    }
    fail(404, 'Not Found');
  }
}

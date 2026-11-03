/**
 * The lite page's server access: the v2 API through an `openapi-fetch` client typed by the generated spec
 * (`../api/schema.ts`), plus the streamed `/v2/generate` and the preview on raw requests. Every request, raw ones
 * included, goes through the one transport of `../lib/web-engine.ts`: the network, or the in-browser server.
 *
 * Same origin by default; `?server=https://host` points the page at another server (CORS is open). `?api=web`
 * forces the in-browser server, `?api=live` the HTTP server; `auto` (the default) takes the in-browser server when
 * no ZONOS server answers and this browser and site can run it. Every request carries `X-TTS-Session-ID`, random
 * per tab and kept in sessionStorage, so the voices this tab caches stay its own.
 */
import createClient from 'openapi-fetch';
import type { Readable } from 'openapi-typescript-helpers';
import type { components, paths } from '../api/schema';
import { selectWebEngine, startWebEngine, transport, webEngineAvailable, type Download } from '../lib/web-engine';

type Schemas = components['schemas'];
export type Capabilities = Schemas['Capabilities'];
export type Speaker = Schemas['Speaker'];
export type GenerateRequest = Schemas['GenerateRequest'];
export type SpeakerCreateRequest = Schemas['SpeakerCreateRequest'];

export const SAMPLE_RATE = 44100;

const params = new URLSearchParams(location.search);
const mode = params.get('api') || import.meta.env.VITE_ZONOS_API || 'auto';
export const server = (params.get('server') || '').replace(/\/+$/, '');
if (mode === 'web') selectWebEngine();

const SESSION_KEY = 'zonos3.lite.session';

function newSessionId(): string {
  // crypto.randomUUID exists only in secure contexts; a plain-http cluster host has none.
  return typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('');
}

const sessionId: string = (() => {
  try {
    const saved = sessionStorage.getItem(SESSION_KEY);
    if (saved) return saved;
    const id = newSessionId();
    sessionStorage.setItem(SESSION_KEY, id);
    return id;
  } catch {
    return newSessionId(); // storage blocked: one session for this page load
  }
})();

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

/** The server's `detail`: a string (400/404/500) or the 422 list of `{loc, msg}`. */
function detailText(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: { loc?: unknown; msg?: unknown }) => {
        const loc = Array.isArray(d?.loc) ? d.loc.filter((p) => p !== 'body').join('.') : '';
        const msg = typeof d?.msg === 'string' ? d.msg : JSON.stringify(d);
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join('; ');
  }
  if (typeof body === 'string' && body.trim() && !/^\s*</.test(body)) return body.slice(0, 300);
  return `The server answered ${status}`;
}

async function responseError(res: Response): Promise<ApiError> {
  const text = await res.text().catch(() => '');
  let body: unknown = text;
  try {
    body = JSON.parse(text);
  } catch {
    /* not JSON: the text itself */
  }
  return new ApiError(res.status, detailText(res.status, body));
}

const client = createClient<paths>({ baseUrl: server, fetch: transport });
client.use({
  onRequest({ request }) {
    request.headers.set('X-TTS-Session-ID', sessionId);
    return request;
  },
});

/** A client call's data typed as the spec schema `T`; an HTTP error becomes an ApiError with its `detail`. */
async function ok<T>(call: Promise<{ data?: Readable<T>; error?: unknown; response: Response }>): Promise<T> {
  const { data, error, response } = await call;
  if (!response.ok) throw new ApiError(response.status, detailText(response.status, error));
  return data as T;
}

const capabilities = () => ok<Capabilities>(client.GET('/v2/capabilities'));

/** Where this page's requests go and what that server can do. */
export interface Backend {
  kind: 'live' | 'web';
  caps: Capabilities;
}

/** The in-browser server: loads its models (`download` gets the progress as they come), then serves this tab. */
async function startWeb(download: (d: Download) => void): Promise<Backend> {
  await startWebEngine(download);
  return { kind: 'web', caps: await capabilities() };
}

/**
 * Picks the backend: the HTTP server when its `/v2/capabilities` answers (or `?api=live`), else the in-browser
 * server in `auto` when this browser has WebGPU and the site serves the engine. A 403 is a ZONOS server's access gate,
 * not a missing server: it stays the error.
 */
export async function connect(download: (d: Download) => void): Promise<Backend> {
  if (mode === 'web') return startWeb(download);
  try {
    return { kind: 'live', caps: await capabilities() };
  } catch (err) {
    if (mode === 'live' || (err instanceof ApiError && err.status === 403) || !(await webEngineAvailable())) throw err;
    return startWeb(download);
  }
}

export const listSpeakers = async (): Promise<Speaker[]> =>
  (await ok<Schemas['SpeakerList']>(client.GET('/v2/speakers'))).speakers;

export const createSpeaker = (body: SpeakerCreateRequest) => ok<Speaker>(client.POST('/v2/speakers', { body }));

export async function speakerEmbedding(id: string): Promise<Float32Array> {
  const data = await ok<Schemas['SpeakerEmbedding']>(
    client.GET('/v2/speakers/{speaker_id}/embedding', { params: { path: { speaker_id: id } } }),
  );
  return Float32Array.from(data.embedding);
}

export async function speakerPreview(id: string): Promise<Blob> {
  const res = await transport(`${server}/v2/speakers/${encodeURIComponent(id)}/preview`, {
    headers: { 'X-TTS-Session-ID': sessionId },
  });
  if (!res.ok) throw await responseError(res);
  return res.blob();
}

/**
 * `POST /v2/generate` as streamed float32 PCM: `onPcm` gets each chunk as it arrives (split on whole samples);
 * resolves with the seed the server used (`X-Seed`).
 */
export async function generateStream(
  body: GenerateRequest,
  onPcm: (pcm: Float32Array) => void,
  signal: AbortSignal,
): Promise<{ seed: string | null }> {
  const res = await transport(`${server}/v2/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-TTS-Session-ID': sessionId },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw await responseError(res);
  if (!res.body) throw new ApiError(res.status, 'The response has no body');
  const seed = res.headers.get('X-Seed');
  const reader = res.body.getReader();
  let carry = new Uint8Array(0);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    let bytes = value;
    if (carry.length) {
      bytes = new Uint8Array(carry.length + value.length);
      bytes.set(carry);
      bytes.set(value, carry.length);
    }
    const whole = bytes.length - (bytes.length % 4);
    carry = bytes.slice(whole);
    // slice: an own, 4-byte aligned buffer for the Float32Array view (float32 little-endian on the wire).
    if (whole) onPcm(new Float32Array(bytes.slice(0, whole).buffer));
  }
  return { seed };
}

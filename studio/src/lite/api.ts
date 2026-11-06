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
import type { components, paths } from '../api/schema';
import { ApiError, createHttp, newSessionId, readError, readFloat32Stream } from '../lib/http';
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

const { client, headers, ok } = createHttp<paths>({ baseUrl: server, session: () => sessionId, fetch: transport });

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
    headers: headers(),
  });
  if (!res.ok) throw await readError(res);
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
    headers: headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw await readError(res);
  if (!res.body) throw new ApiError(res.status, 'The response has no body');
  const seed = res.headers.get('X-Seed');
  await readFloat32Stream(res.body, onPcm);
  return { seed };
}

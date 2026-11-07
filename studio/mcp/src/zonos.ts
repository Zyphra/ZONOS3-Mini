/**
 * The only module that talks to the ZONOS3 server: JSON routes through an openapi-fetch client typed by the
 * studio's generated `paths`, binary routes (`/v2/generate`, `/v2/media`) on fetch. Mirrors the studio's
 * `api.ts` (session header, keep-alive edits, error bodies).
 */
import type { Alignment, ControlsSpec, EditRequest, EditResponse, GenerateRequest, ProjectPut, ProjectSaved, ServerProject, ServerProjectEntry, Speaker, SpanFollow } from '../../src/api';
import type { components, paths } from '../../src/api/schema';
import { ApiError, bodyOf, createHttp, errorOf, MissingMedia, readError, RevisionConflict } from '../../src/lib/http';
import { config } from './config';
import { state } from './state';

type Schemas = components['schemas'];
export type Capabilities = Schemas['Capabilities'];
export type SpeakerCreate = Schemas['SpeakerCreateRequest'];

const http = createHttp<paths>({ baseUrl: config.server, session: () => state.sessionId, token: () => config.token || undefined });
const client = http.client;

/** The server unreachable (a network error) as a 503 that says how to point at it. */
function unreachable(e: unknown): ApiError {
  return new ApiError(503, `The Zonos server at ${config.server} does not answer (${e instanceof Error ? (e.cause instanceof Error ? e.cause.message : e.message) : String(e)}). Start it, or set ZONOS_SERVER.`);
}

const reached = <C>(call: Promise<C>): Promise<C> =>
  call.catch((e: unknown) => {
    throw unreachable(e);
  });

/** The data of a client call (keep-alive errors and proxy pages as ApiError, the server unreachable as a 503). */
const ok = <T>(call: Parameters<typeof http.ok<T>>[0]): Promise<T> => http.ok<T>(reached(call));

async function raw(path: string, init: RequestInit = {}): Promise<Response> {
  return reached(fetch(`${config.server}${path}`, { ...init, headers: http.headers(init.headers) }));
}

// ── Routes ────────────────────────────────────────────────────────────────────────────────────────

export const capabilities = () => ok<Capabilities>(client.GET('/v2/capabilities'));

/** The steering bundle, or null when the server has none (404). */
export async function controls(): Promise<ControlsSpec | null> {
  try {
    return (await ok<Schemas['ControlsDocument']>(client.GET('/v2/controls'))) as ControlsSpec;
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return null;
    throw e;
  }
}

export const speakers = async () => (await ok<Schemas['SpeakerList']>(client.GET('/v2/speakers'))).speakers;

export async function speakerEmbedding(id: string): Promise<number[]> {
  return (await ok<Schemas['SpeakerEmbedding']>(client.GET('/v2/speakers/{speaker_id}/embedding', { params: { path: { speaker_id: id } } }))).embedding;
}

export async function speakerPreview(id: string): Promise<Uint8Array> {
  const res = await raw(`/v2/speakers/${encodeURIComponent(id)}/preview`);
  if (!res.ok) throw await readError(res);
  return new Uint8Array(await res.arrayBuffer());
}

export const createSpeaker = (body: SpeakerCreate) => ok<Speaker>(client.POST('/v2/speakers', { body }));

export interface Generated {
  wav: Uint8Array;
  seed: number;
  spanFollow?: SpanFollow;
  ttfbMs: number;
  totalMs: number;
}

/** `POST /v2/generate` as one buffered 16-bit WAV. */
export async function generate(req: GenerateRequest): Promise<Generated> {
  const t0 = performance.now();
  const body: GenerateRequest = { ...req, stream: false, format: 'wav' };
  const res = await raw('/v2/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const ttfbMs = performance.now() - t0;
  if (!res.ok) throw await readError(res);
  const wav = new Uint8Array(await res.arrayBuffer());
  const spanFollow: SpanFollow | undefined = body.control_spans?.length ? (res.headers.get('X-Control-Span-Follow') === 'reading' ? 'reading' : 'rate') : undefined;
  const seed = Number(res.headers.get('X-Seed') ?? req.seed ?? 0);
  return { wav, seed, ttfbMs: Math.round(ttfbMs), totalMs: Math.round(performance.now() - t0), ...(spanFollow ? { spanFollow } : {}) };
}

export const transcribe = (audio_base64: string, language: string) => ok<Alignment>(client.POST('/v2/edit/transcribe', { body: { audio_base64, language } }));
export const align = (audio_base64: string, text: string, language: string) => ok<Alignment>(client.POST('/v2/edit/align', { body: { audio_base64, text, language } }));
export const edit = (req: EditRequest) => ok<EditResponse>(client.POST('/v2/edit', { body: req }));

export const listProjects = () => ok<ServerProjectEntry[]>(client.GET('/v2/projects'));
export const getProject = (id: string) => ok<ServerProject>(client.GET('/v2/projects/{project_id}', { params: { path: { project_id: id } } }));

/** Save a project; 409 → RevisionConflict, 400 with `missing` → MissingMedia. */
export async function putProject(id: string, body: ProjectPut): Promise<ProjectSaved> {
  const { response, text } = await http.send(reached(client.PUT('/v2/projects/{project_id}', { params: { path: { project_id: id } }, body })));
  if (response.ok) return bodyOf<ProjectSaved>(response.status, text);
  let err: Partial<Schemas['Error']> = {};
  try {
    err = JSON.parse(text);
  } catch {
    /* not JSON: errorOf below */
  }
  if (response.status === 409) throw new RevisionConflict(err.revision ?? null);
  if (response.status === 400 && err.missing) throw new MissingMedia(err.missing);
  throw errorOf(response.status, response.statusText, text);
}

const mediaPath = (sha: string) => `/v2/media/${encodeURIComponent(sha)}`;

export async function hasMedia(sha: string): Promise<boolean> {
  const res = await raw(mediaPath(sha), { method: 'HEAD' });
  if (res.status === 404) return false;
  if (!res.ok) throw await readError(res);
  return true;
}

export async function putMedia(sha: string, bytes: Uint8Array, type: string): Promise<void> {
  const res = await raw(mediaPath(sha), { method: 'PUT', headers: { 'Content-Type': type || 'application/octet-stream' }, body: bytes as BodyInit });
  if (!res.ok) throw await readError(res);
}

export async function getMedia(sha: string): Promise<Uint8Array> {
  const res = await raw(mediaPath(sha));
  if (!res.ok) throw await readError(res);
  return new Uint8Array(await res.arrayBuffer());
}

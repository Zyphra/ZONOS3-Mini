/**
 * The only module that talks to the ZONOS3 server: JSON routes through an openapi-fetch client typed by the
 * studio's generated `paths`, binary routes (`/v2/generate`, `/v2/media`) on fetch. Mirrors the studio's
 * `api.ts` (session header, keep-alive edits, error bodies).
 */
import createClient from 'openapi-fetch';
import type { Alignment, ControlsSpec, EditRequest, EditResponse, GenerateRequest, ProjectPut, ProjectSaved, Revision, ServerProject, ServerProjectEntry, Speaker, SpanFollow } from '../../src/api';
import type { components, paths } from '../../src/api/schema';
import { config } from './config';
import { state } from './state';

type Schemas = components['schemas'];
export type Capabilities = Schemas['Capabilities'];
export type SpeakerCreate = Schemas['SpeakerCreateRequest'];

export class ZonosError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
  }
}

/** 409 from `PUT /v2/projects/{id}`: saved elsewhere since `base_revision`. */
export class RevisionConflict extends ZonosError {
  constructor(readonly revision: Revision | null) {
    super(409, 'The project changed elsewhere');
  }
}

/** 400 from `PUT /v2/projects/{id}`: media the document names that the server lacks. */
export class MissingMedia extends ZonosError {
  constructor(readonly missing: string[]) {
    super(400, `${missing.length} media file(s) missing on the server`);
  }
}

const PROXY_PAGE = 'The server sent a web page instead of an answer (a proxy error?)';

/** An error response as a short message: the server's `detail`, never a proxy's HTML page. */
export function errorOf(status: number, text: string): ZonosError {
  try {
    const body = JSON.parse(text);
    const detail = body?.detail ?? body;
    return new ZonosError(typeof body?.status_code === 'number' ? body.status_code : status, typeof detail === 'string' ? detail : JSON.stringify(detail));
  } catch {
    if (status === 524 || status === 504 || status === 408) return new ZonosError(status, `The server didn't answer in time (proxy ${status})`);
    if (/^\s*</.test(text)) return new ZonosError(status, PROXY_PAGE);
    if (!text.trim()) return new ZonosError(status, `Server error (${status})`);
    return new ZonosError(status, text.slice(0, 300));
  }
}

/** A successful response's JSON; a keep-alive body (spaces, then JSON) carrying `status_code` ≥ 400 is its error. */
export function parseBody<T>(status: number, text: string): T {
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    throw new ZonosError(status, /^\s*</.test(text) ? PROXY_PAGE : 'unreadable response');
  }
  const code = (body as { status_code?: unknown } | null)?.status_code;
  if (typeof code === 'number' && code >= 400) throw errorOf(code, text);
  return body as T;
}

const KEEPALIVE = /^\/v2\/edit(\/|$)/;

function headers(extra: Record<string, string> = {}): Record<string, string> {
  return { 'X-TTS-Session-ID': state.sessionId, ...(config.token ? { Authorization: `Bearer ${config.token}` } : {}), ...extra };
}

/** Body text of each client response, kept by the middleware (openapi-fetch consumes the body itself). */
const bodies = new WeakMap<Response, string>();
const client = createClient<paths>({ baseUrl: config.server });
client.use({
  onRequest({ request, schemaPath }) {
    for (const [k, v] of Object.entries(headers())) request.headers.set(k, v);
    if (KEEPALIVE.test(schemaPath)) request.headers.set('X-Keepalive', '1');
    return request;
  },
  async onResponse({ response }) {
    bodies.set(response, await response.clone().text());
    return undefined;
  },
});

/** A client call's response and body text (the server unreachable as a ZonosError). */
async function send(call: Promise<{ response: Response }>): Promise<{ response: Response; text: string }> {
  try {
    const { response } = await call;
    return { response, text: bodies.get(response) ?? '' };
  } catch (e) {
    throw unreachable(e);
  }
}

/** The data of a client call, read from text so keep-alive errors and proxy pages are caught. */
async function ok<T>(call: Promise<{ response: Response }>): Promise<T> {
  const { response, text } = await send(call);
  if (!response.ok) throw errorOf(response.status, text);
  return parseBody<T>(response.status, text);
}

function unreachable(e: unknown): ZonosError {
  return new ZonosError(503, `The Zonos server at ${config.server} does not answer (${e instanceof Error ? (e.cause instanceof Error ? e.cause.message : e.message) : String(e)}). Start it, or set ZONOS_SERVER.`);
}

async function raw(path: string, init: RequestInit = {}): Promise<Response> {
  try {
    return await fetch(`${config.server}${path}`, { ...init, headers: headers((init.headers ?? {}) as Record<string, string>) });
  } catch (e) {
    throw unreachable(e);
  }
}

async function fail(res: Response): Promise<never> {
  throw errorOf(res.status, await res.text().catch(() => ''));
}

// ── Routes ────────────────────────────────────────────────────────────────────────────────────────

export const capabilities = () => ok<Capabilities>(client.GET('/v2/capabilities'));

/** The steering bundle, or null when the server has none (404). */
export async function controls(): Promise<ControlsSpec | null> {
  try {
    return await ok<ControlsSpec>(client.GET('/v2/controls'));
  } catch (e) {
    if (e instanceof ZonosError && e.status === 404) return null;
    throw e;
  }
}

export const speakers = async () => (await ok<Schemas['SpeakerList']>(client.GET('/v2/speakers'))).speakers;

export async function speakerEmbedding(id: string): Promise<number[]> {
  return (await ok<Schemas['SpeakerEmbedding']>(client.GET('/v2/speakers/{speaker_id}/embedding', { params: { path: { speaker_id: id } } }))).embedding;
}

export async function speakerPreview(id: string): Promise<Uint8Array> {
  const res = await raw(`/v2/speakers/${encodeURIComponent(id)}/preview`);
  if (!res.ok) await fail(res);
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
  if (!res.ok) await fail(res);
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
  const { response, text } = await send(client.PUT('/v2/projects/{project_id}', { params: { path: { project_id: id } }, body }));
  if (response.ok) return parseBody<ProjectSaved>(response.status, text);
  let err: Partial<Schemas['Error']> = {};
  try {
    err = JSON.parse(text);
  } catch {
    /* not JSON: errorOf below */
  }
  if (response.status === 409) throw new RevisionConflict(err.revision ?? null);
  if (response.status === 400 && err.missing) throw new MissingMedia(err.missing);
  throw errorOf(response.status, text);
}

const mediaPath = (sha: string) => `/v2/media/${encodeURIComponent(sha)}`;

export async function hasMedia(sha: string): Promise<boolean> {
  const res = await raw(mediaPath(sha), { method: 'HEAD' });
  if (res.status === 404) return false;
  if (!res.ok) await fail(res);
  return true;
}

export async function putMedia(sha: string, bytes: Uint8Array, type: string): Promise<void> {
  const res = await raw(mediaPath(sha), { method: 'PUT', headers: { 'Content-Type': type || 'application/octet-stream' }, body: bytes as BodyInit });
  if (!res.ok) await fail(res);
}

export async function getMedia(sha: string): Promise<Uint8Array> {
  const res = await raw(mediaPath(sha));
  if (!res.ok) await fail(res);
  return new Uint8Array(await res.arrayBuffer());
}

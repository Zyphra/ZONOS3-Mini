/**
 * The HTTP core every ZONOS3 client shares (the studio, the lite page, the MCP server): error bodies as ApiError,
 * keep-alive answers, the session / token headers, an `openapi-fetch` client typed by the spec, and the streamed
 * float32 PCM reader. Pure: nothing here touches `location`, storage or `import.meta.env` (it runs on Node too).
 */
import createClient, { type Client } from 'openapi-fetch';
import type { Readable } from 'openapi-typescript-helpers';
import type { components } from '../api/schema';

type Revision = components['schemas']['Project']['revision'];

/**
 * Long POSTs the server answers in keep-alive form (`X-Keepalive: 1`): 200 at once, whitespace heartbeats, then the
 * JSON — an error as `{detail, status_code}`. A proxy in front (a Cloudflare quick tunnel drops requests silent for
 * 100 s) then never times them out.
 */
export const KEEPALIVE_PATH = /^\/v2\/edit(\/|$)/;

export const PROXY_PAGE = 'The server sent a web page instead of an answer (a proxy error?)';

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
  }
}

/** 409 from `PUT /v2/projects/{id}`: someone saved since `base_revision`; `revision` is the stored one (null: deleted). */
export class RevisionConflict extends ApiError {
  constructor(readonly revision: Revision | null) {
    super(409, 'The project changed elsewhere since it was opened');
  }
}

/** 400 from `PUT /v2/projects/{id}`: media the document names that the server lacks (upload them, retry). */
export class MissingMedia extends ApiError {
  constructor(readonly missing: string[]) {
    super(400, `${missing.length} media file${missing.length === 1 ? ' is' : 's are'} missing on the server`);
  }
}

/** A random session id (crypto.randomUUID exists only in secure contexts; a plain-http cluster host has none). */
export function newSessionId(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const hex = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/** A `detail` as text: a string as is, the 422 list of `{loc, msg}` as `loc: msg; …`, anything else as JSON. */
function detailText(detail: unknown): string {
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
  return JSON.stringify(detail);
}

/** An error response as a short message: the server's `detail`, never a proxy's HTML page. */
export function errorOf(status: number, statusText: string, text: string): ApiError {
  try {
    const body = JSON.parse(text);
    return new ApiError(typeof body?.status_code === 'number' ? body.status_code : status, detailText(body?.detail ?? body));
  } catch {
    if (status === 524 || status === 504 || status === 408) return new ApiError(status, `The server didn’t answer in time (proxy ${status}) — try again`);
    if (/^\s*</.test(text) || !text.trim()) return new ApiError(status, status >= 500 ? `Server error (${status})` : `${status} ${statusText}`.trim());
    return new ApiError(status, text.slice(0, 300));
  }
}

/** A successful response's JSON; a keep-alive body carrying `status_code` ≥ 400 is the error it reports. */
export function bodyOf<T>(status: number, text: string): T {
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    throw new ApiError(status, /^\s*</.test(text) ? `${PROXY_PAGE} — try again` : 'unreadable response');
  }
  const code = (body as { status_code?: unknown } | null)?.status_code;
  if (typeof code === 'number' && code >= 400) throw errorOf(code, '', text);
  return body as T;
}

/** A failed raw response as ApiError. */
export async function readError(res: Response): Promise<ApiError> {
  const text = await res.text().catch(() => '');
  return errorOf(res.status, res.statusText, text);
}

/** A client call as openapi-fetch resolves it; `data` typed so `ok<T>` keeps the route ↔ schema check. */
type Call<T> = Promise<{ data?: Readable<T>; error?: unknown; response: Response }>;

export interface HttpOptions {
  baseUrl?: string;
  /** The `X-TTS-Session-ID` of every request. */
  session: () => string;
  /** A bearer token (the server's access gate), when there is one. */
  token?: () => string | undefined;
  fetch?: typeof fetch;
}

export interface Http<paths extends object> {
  /** Every JSON route: typed by the spec's `paths`, with the session (and keep-alive) headers. */
  client: Client<paths>;
  /** The session / token headers (plus `extra`) for raw requests. */
  headers(extra?: HeadersInit): Headers;
  /**
   * The data of a client call, typed as the spec schema `T` (an HTTP error, a keep-alive error body or a proxy page
   * as ApiError). openapi-fetch types data as `Readable<…>`, which turns tuples into arrays and drops null-typed
   * keys: matching it against `Readable<T>` keeps the route ↔ schema check while callers get the schema's own type.
   */
  ok<T>(call: Call<T>): Promise<T>;
  /** A client call's response and body text (network errors rethrown unchanged). */
  send(call: Promise<{ response: Response }>): Promise<{ response: Response; text: string }>;
}

export function createHttp<paths extends object>(opts: HttpOptions): Http<paths> {
  const headers = (extra?: HeadersInit): Headers => {
    const h = new Headers(extra);
    h.set('X-TTS-Session-ID', opts.session());
    const token = opts.token?.();
    if (token) h.set('Authorization', `Bearer ${token}`);
    return h;
  };
  /** Body text of each client response, kept by the middleware (openapi-fetch consumes the body itself). */
  const bodies = new WeakMap<Response, string>();
  const client = createClient<paths>({ baseUrl: opts.baseUrl ?? '', ...(opts.fetch ? { fetch: opts.fetch } : {}) });
  client.use({
    onRequest({ request, schemaPath }) {
      headers().forEach((v, k) => request.headers.set(k, v));
      if (KEEPALIVE_PATH.test(schemaPath)) request.headers.set('X-Keepalive', '1');
      return request;
    },
    async onResponse({ response }) {
      // Never a streamed (audio) body: cloning it would buffer the whole stream.
      if (!/^(audio\/|application\/octet-stream)/.test(response.headers.get('Content-Type') ?? '')) bodies.set(response, await response.clone().text());
      return undefined;
    },
  });
  const send = async (call: Promise<{ response: Response }>) => {
    const { response } = await call;
    return { response, text: bodies.get(response) ?? '' };
  };
  const ok = async <T>(call: Call<T>): Promise<T> => {
    const { response, text } = await send(call);
    if (!response.ok) throw errorOf(response.status, response.statusText, text);
    if (response.status === 204 || !text) return undefined as T;
    return bodyOf<T>(response.status, text);
  };
  return { client, headers, ok, send };
}

/** A streamed float32 little-endian body: `onPcm` gets each chunk split on whole samples, `onChunk` each network read. */
export async function readFloat32Stream(body: ReadableStream<Uint8Array>, onPcm: (pcm: Float32Array) => void, onChunk?: () => void): Promise<void> {
  const reader = body.getReader();
  let carry = new Uint8Array(0);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    onChunk?.();
    let bytes = value;
    if (carry.length) {
      bytes = new Uint8Array(carry.length + value.length);
      bytes.set(carry);
      bytes.set(value, carry.length);
    }
    const whole = bytes.length - (bytes.length % 4);
    carry = bytes.slice(whole);
    // slice: an own, 4-byte aligned buffer for the Float32Array view.
    if (whole) onPcm(new Float32Array(bytes.slice(0, whole).buffer));
  }
}

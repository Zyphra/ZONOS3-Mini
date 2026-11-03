// In-process HTTP plumbing of the engine API: requests, the response sink, {"detail"} errors and body parsing
// (servers/ggml/src/server/http.{h,cpp}, schema.cpp).
import { validate } from './validate';

export interface EngineRequest {
  method: string;
  path: string;                       // path and query, e.g. /v2/speakers/x/preview?y=1
  headers: Record<string, string>;    // lower-case names
  body: ArrayBuffer | null;
  pcm?: Float32Array;                 // speaker_audio_base64 decoded to 44.1 kHz mono by the client (src/client.ts)
}

/** Response channel: start() once with status and headers, then chunk() any number of times, then end(). */
export interface Sink {
  start(status: number, headers: Record<string, string>): void;
  chunk(data: Uint8Array): void;
  end(): void;
}

/** HTTPException(status_code, detail) or a full error body (422 ValidationError). */
export class HttpError extends Error {
  constructor(readonly status: number, readonly body: Record<string, unknown>) {
    super(typeof body.detail === 'string' ? body.detail : `HTTP ${status}`);
  }
}

export function fail(status: number, detail: string): never {
  throw new HttpError(status, { detail });
}

const utf8 = new TextEncoder();

export function sendJson(sink: Sink, body: unknown, status = 200): void {
  sink.start(status, { 'content-type': 'application/json' });
  sink.chunk(utf8.encode(JSON.stringify(body)));
  sink.end();
}

export function sendBytes(sink: Sink, data: Uint8Array, contentType: string, headers: Record<string, string> = {}): void {
  sink.start(200, { ...headers, 'content-type': contentType });
  sink.chunk(data);
  sink.end();
}

/** X-TTS-Session-ID, stripped; null when absent or blank (sgl _normalize_session_id). */
export function sessionId(req: EngineRequest): string | null {
  const v = req.headers['x-tts-session-id']?.trim();
  return v ? v : null;
}

/** Parse the body as JSON and validate it against #/components/schemas/<name>; 422 when either fails. */
export function parseBody(req: EngineRequest, name: string): Record<string, unknown> {
  const text = req.body ? new TextDecoder().decode(req.body) : '';
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch (e) {
    const m = e instanceof SyntaxError ? /position (\d+)/.exec(e.message) : null;
    throw new HttpError(422, { detail: [{ loc: ['body', m ? Number(m[1]) : 0], msg: 'JSON decode error', type: 'json_invalid' }] });
  }
  const errors = validate(name, body);
  if (errors.length) throw new HttpError(422, { detail: errors });
  return body as Record<string, unknown>;  // validated: every schema the routes parse is an object schema
}

// --- typed reads of a validated request body (sgl's `set` / `num` helpers)

/** The key is present and not null. */
export function isSet(j: Record<string, unknown>, key: string): boolean {
  return j[key] !== undefined && j[key] !== null;
}

export function num(j: Record<string, unknown>, key: string, d: number): number {
  const v = j[key];
  return typeof v === 'number' ? v : d;
}

export function str(j: Record<string, unknown>, key: string): string | null {
  const v = j[key];
  return typeof v === 'string' ? v : null;
}

export function bool(j: Record<string, unknown>, key: string): boolean | null {
  const v = j[key];
  return typeof v === 'boolean' ? v : null;
}

/** Python truthiness of an optional container (None, {} and [] are false). */
export function truthy(j: Record<string, unknown>, key: string): boolean {
  const v = j[key];
  if (v === undefined || v === null) return false;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === 'object') return Object.keys(v).length > 0;
  return Boolean(v);
}

/** Python repr() of a str for error messages. */
export function pyRepr(s: string): string {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = q;
  for (const ch of s) {
    if (ch === '\\') out += '\\\\';
    else if (ch === q) out += '\\' + q;
    else if (ch === '\n') out += '\\n';
    else if (ch === '\r') out += '\\r';
    else if (ch === '\t') out += '\\t';
    else out += ch;
  }
  return out + q;
}

export function pyList(items: string[]): string {
  return '[' + items.map(pyRepr).join(', ') + ']';
}

/** Python str() of a float for error messages (4 -> '4.0'). */
export function pyFloat(v: number): string {
  return Number.isInteger(v) ? v.toFixed(1) : String(v);
}

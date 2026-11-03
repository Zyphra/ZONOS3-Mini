// Main-thread facade of the in-browser server: spawns the engine worker and exposes a fetch-compatible localFetch()
// that serves the v2 API in-process. Audio the worker cannot decode (it has no AudioContext) is decoded here: default
// voices at init, speaker_audio_base64 per request (sent next to the JSON as 44.1 kHz mono PCM).
import { b64DecodeStrict } from './b64';
import { fetchJson, modelsBaseUrl } from './gpu/weights';
import type { InMsg, OutMsg } from './worker';

export type Progress = { model: string; loaded: number; total: number };

interface Pending {
  resolve: (r: Response) => void;
  reject: (e: unknown) => void;
  controller?: ReadableStreamDefaultController<Uint8Array>;
}

const MAX_REFERENCE_SECONDS = 60;
const AUDIO_ROUTES: Record<string, true> = { '/v2/speakers': true, '/v2/generate': true, '/v1/audio/speech': true };
const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
const pending = new Map<number, Pending>();
let nextId = 1;
let ready = false;
let initPromise: Promise<void> | null = null;
let initResolve: (() => void) | null = null;
let initReject: ((e: Error) => void) | null = null;
let onProgress: ((p: Progress) => void) | null = null;
const fatalCbs: ((m: string) => void)[] = [];
const encoderCbs: ((p: Progress) => void)[] = [];

const send = (m: InMsg, transfer: Transferable[] = []) => worker.postMessage(m, transfer);

worker.onmessage = (ev: MessageEvent<OutMsg>) => {
  const m = ev.data;
  switch (m.type) {
    case 'progress':
      if (!ready) onProgress?.(m);
      else for (const cb of encoderCbs) cb(m);
      break;
    case 'ready':
      ready = true;
      initResolve?.();
      break;
    case 'fatal':
      if (!ready) initReject?.(new Error(m.message));
      else for (const cb of fatalCbs) cb(m.message);
      break;
    case 'res': {
      const p = pending.get(m.id);
      if (!p) break;
      const body = new ReadableStream<Uint8Array>({ start: c => { p.controller = c; } });
      p.resolve(new Response(body, { status: m.status, headers: m.headers }));
      break;
    }
    case 'chunk':
      pending.get(m.id)?.controller?.enqueue(new Uint8Array(m.data));
      break;
    case 'end':
      pending.get(m.id)?.controller?.close();
      pending.delete(m.id);
      break;
  }
};
worker.onerror = ev => {
  const msg = ev.message || 'engine worker failed to start';
  if (!ready) initReject?.(new Error(msg));
  else for (const cb of fatalCbs) cb(msg);
};

export interface InitOptions {
  /** Models directory URL (default: `?models=` or `/models/`). */
  base?: string;
  /** Keep downloaded shards in Cache Storage (default true). */
  cache?: boolean;
}

/** Decode any browser-supported audio file to 44.1 kHz mono (downmix + resample), cropped to 60 s. */
export async function decodeTo44k(bytes: ArrayBuffer): Promise<Float32Array> {
  const probe = new OfflineAudioContext(1, 1, 44100);
  const decoded = await probe.decodeAudioData(bytes);
  const len = Math.min(Math.ceil(decoded.duration * 44100), MAX_REFERENCE_SECONDS * 44100);
  if (len <= 0) return new Float32Array(0);
  const ctx = new OfflineAudioContext(1, len, 44100);
  const src = ctx.createBufferSource();
  src.buffer = decoded;
  src.connect(ctx.destination);
  src.start();
  const out = await ctx.startRendering();
  return out.getChannelData(0).slice(0, len);
}

/** Each default voice's clip as 44.1 kHz mono (null-free: a clip that fails to decode is left out). */
async function defaultVoicesPcm(base: string): Promise<Record<string, Float32Array>> {
  const voicesBase = new URL('voices/', base).href;
  const voices = await fetchJson<{ id: string; file: string }[]>(voicesBase + 'voices.json');
  const out: Record<string, Float32Array> = {};
  await Promise.all(voices.map(async v => {
    try {
      const r = await fetch(new URL(v.file, voicesBase).href);
      if (r.ok) out[v.id] = await decodeTo44k(await r.arrayBuffer());
    } catch (e) {
      console.warn(`zonos3-web: default voice ${v.file} did not decode`, e);
    }
  }));
  return out;
}

export const engine = {
  init(progress: (p: Progress) => void, opts: InitOptions = {}): Promise<void> {
    onProgress = progress;
    initPromise ??= (async () => {
      if (!('gpu' in navigator) || !navigator.gpu) throw new Error('WebGPU is not available in this browser');
      const base = opts.base ?? modelsBaseUrl();
      const voicesPcm = await defaultVoicesPcm(base);
      const started = Promise.withResolvers<void>();
      initResolve = started.resolve;
      initReject = started.reject;
      send({ type: 'init', base, cache: opts.cache ?? true, voicesPcm }, Object.values(voicesPcm).map(p => p.buffer));
      await started.promise;
    })().catch(e => {
      initPromise = null;   // allow a retry
      throw e;
    });
    return initPromise;
  },
  onFatal(cb: (m: string) => void): void { fatalCbs.push(cb); },
  onEncoderProgress(cb: (p: Progress) => void): void { encoderCbs.push(cb); },
};

/** speaker_audio_base64 of a JSON body decoded to 44.1 kHz mono; undefined when absent or undecodable (the worker
 *  answers the API's 400 for those after validating the body). */
async function speakerPcm(body: ArrayBuffer | null): Promise<Float32Array | undefined> {
  if (!body) return undefined;
  let json: unknown;
  try { json = JSON.parse(new TextDecoder().decode(body)); } catch { return undefined; }
  if (!json || typeof json !== 'object' || !('speaker_audio_base64' in json)) return undefined;
  const v = json.speaker_audio_base64;
  if (typeof v !== 'string' || !v.trim()) return undefined;
  const bytes = b64DecodeStrict(v);
  if (!bytes) return undefined;
  try {
    const pcm = await decodeTo44k(bytes.slice().buffer);
    return pcm.length ? pcm : undefined;
  } catch {
    return undefined;
  }
}

/** fetch() for the v2 API, served by the engine worker. Accepts a Request (openapi-fetch) or a URL plus init. */
export async function localFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const req = input instanceof Request ? input : null;
  const url = new URL(req ? req.url : String(input), location.href);
  const path = url.pathname + url.search;
  const method = (init?.method ?? req?.method ?? 'GET').toUpperCase();
  const headers: Record<string, string> = {};
  new Headers(init?.headers ?? req?.headers).forEach((v, k) => { headers[k.toLowerCase()] = v; });
  let body: ArrayBuffer | null = null;
  if (init?.body !== undefined && init.body !== null) body = await new Response(init.body).arrayBuffer();
  else if (req && method !== 'GET' && method !== 'HEAD') body = await req.arrayBuffer();
  if (body && body.byteLength === 0) body = null;
  const pcm = method === 'POST' && AUDIO_ROUTES[url.pathname] ? await speakerPcm(body) : undefined;
  const signal = init?.signal ?? req?.signal ?? null;
  const id = nextId++;
  const { promise, resolve, reject } = Promise.withResolvers<Response>();
  const p: Pending = { resolve, reject };
  pending.set(id, p);
  if (signal) {
    const onAbort = () => {
      send({ type: 'abort', id });
      const err = new DOMException('The operation was aborted.', 'AbortError');
      if (p.controller) { try { p.controller.error(err); } catch { /* already closed */ } }
      else reject(err);
      pending.delete(id);
    };
    if (signal.aborted) { onAbort(); return promise; }
    signal.addEventListener('abort', onAbort, { once: true });
  }
  const transfer: Transferable[] = [];
  if (body) transfer.push(body);
  if (pcm) transfer.push(pcm.buffer);
  send({ type: 'req', id, method, path, headers, body, pcm }, transfer);
  return promise;
}

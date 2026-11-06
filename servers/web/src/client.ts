// Main-thread facade of the in-browser server: spawns the engine worker (and the codec worker holding the host half
// of the codec, connected to the engine worker over a MessageChannel) and exposes a fetch-compatible localFetch()
// that serves the v2 API in-process. Audio the worker cannot decode (it has no AudioContext) is decoded here: default
// voices at init, speaker_audio_base64 (and the edit routes' audio_base64) per request, sent next to the JSON as
// 44.1 kHz mono PCM with the upload's whole length.
import { b64DecodeStrict } from './b64';
import { EDIT_MAX_SECONDS, REFERENCE_AUDIO_MAX_SECONDS, SAMPLE_RATE } from './generation';
import { fetchJson, modelsBaseUrl, type LoadProgress } from './gpu/weights';
import type { InMsg, OutMsg } from './worker';

/** A model download readout: the models loading together, their bytes in (`loaded` of `total`) and how many of those
 *  came over the network (`fetched`; the rest from Cache Storage), for a bar and a download-rate ETA. */
export type Progress = LoadProgress;

interface Pending {
  resolve: (r: Response) => void;
  reject: (e: unknown) => void;
  controller?: ReadableStreamDefaultController<Uint8Array>;
}

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
/** The routes whose body carries audio the worker reads: its field and the longest stretch decoded (seconds). */
const AUDIO_ROUTES: Record<string, { field: string; maxSeconds: number }> = {
  '/v2/speakers': { field: 'speaker_audio_base64', maxSeconds: REFERENCE_AUDIO_MAX_SECONDS },
  '/v2/generate': { field: 'speaker_audio_base64', maxSeconds: REFERENCE_AUDIO_MAX_SECONDS },
  '/v1/audio/speech': { field: 'speaker_audio_base64', maxSeconds: REFERENCE_AUDIO_MAX_SECONDS },
  '/v2/edit': { field: 'audio_base64', maxSeconds: EDIT_MAX_SECONDS },
  '/v2/edit/align': { field: 'audio_base64', maxSeconds: EDIT_MAX_SECONDS },
};
const codecWorker = new Worker(new URL('./codec-worker.ts', import.meta.url), { type: 'module' });
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
    case 'progress': {
      const { type: _, ...p } = m;
      if (!ready) onProgress?.(p);
      else for (const cb of encoderCbs) cb(p);
      break;
    }
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
codecWorker.onerror = ev => {
  const msg = 'codec worker failed: ' + (ev.message || 'did not start');
  if (!ready) initReject?.(new Error(msg));
  else for (const cb of fatalCbs) cb(msg);
};

export interface InitOptions {
  /** Models directory URL (default: `?models=` or `/models/`). */
  base?: string;
  /** Keep downloaded shards in Cache Storage (default true). */
  cache?: boolean;
}

/** Decode any browser-supported audio file to API-rate mono (downmix + resample), cropped to `maxSeconds`
 *  (REFERENCE_AUDIO_MAX_SECONDS by default); `seconds` is the whole file's length. */
export async function decodeTo44k(bytes: ArrayBuffer, maxSeconds = REFERENCE_AUDIO_MAX_SECONDS):
    Promise<{ pcm: Float32Array; seconds: number }> {
  const probe = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const decoded = await probe.decodeAudioData(bytes);
  const len = Math.min(Math.ceil(decoded.duration * SAMPLE_RATE), maxSeconds * SAMPLE_RATE);
  if (len <= 0) return { pcm: new Float32Array(0), seconds: decoded.duration };
  const ctx = new OfflineAudioContext(1, len, SAMPLE_RATE);
  const src = ctx.createBufferSource();
  src.buffer = decoded;
  src.connect(ctx.destination);
  src.start();
  const out = await ctx.startRendering();
  return { pcm: out.getChannelData(0).slice(0, len), seconds: decoded.duration };
}

/** Each default voice's clip as 44.1 kHz mono (null-free: a clip that fails to decode is left out). */
async function defaultVoicesPcm(base: string): Promise<Record<string, Float32Array>> {
  const voicesBase = new URL('voices/', base).href;
  const voices = await fetchJson<{ id: string; file: string }[]>(voicesBase + 'voices.json');
  const out: Record<string, Float32Array> = {};
  await Promise.all(voices.map(async v => {
    try {
      const r = await fetch(new URL(v.file, voicesBase).href);
      if (r.ok) out[v.id] = (await decodeTo44k(await r.arrayBuffer())).pcm;
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
      const ch = new MessageChannel();
      codecWorker.postMessage({ type: 'port' }, [ch.port2]);
      send({ type: 'init', base, cache: opts.cache ?? true, voicesPcm, codecPort: ch.port1 },
        [...Object.values(voicesPcm).map(p => p.buffer), ch.port1]);
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

/** The route's audio field of a JSON body decoded to 44.1 kHz mono (at most the route's maxSeconds) and the upload's
 *  whole length; undefined when absent or undecodable (the worker answers the API's 400 for those after validating the
 *  body). */
async function requestPcm(body: ArrayBuffer | null, route: { field: string; maxSeconds: number }):
    Promise<{ pcm: Float32Array; seconds: number } | undefined> {
  if (!body) return undefined;
  let json: unknown;
  try { json = JSON.parse(new TextDecoder().decode(body)); } catch { return undefined; }
  if (!json || typeof json !== 'object' || !(route.field in json)) return undefined;
  const v: unknown = Reflect.get(json, route.field);
  if (typeof v !== 'string' || !v.trim()) return undefined;
  const bytes = b64DecodeStrict(v);
  if (!bytes) return undefined;
  try {
    const decoded = await decodeTo44k(bytes.slice().buffer, route.maxSeconds);
    return decoded.pcm.length ? decoded : undefined;
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
  const route = method === 'POST' ? AUDIO_ROUTES[url.pathname] : undefined;
  const audio = route ? await requestPcm(body, route) : undefined;
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
  if (audio) transfer.push(audio.pcm.buffer);
  send({ type: 'req', id, method, path, headers, body, pcm: audio?.pcm, pcmSeconds: audio?.seconds }, transfer);
  return promise;
}

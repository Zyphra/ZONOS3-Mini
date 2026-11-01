/**
 * Every render goes through here: a cache keyed by the full request, de-duplication of identical in-flight
 * requests, and a small concurrency limit (latency stays near 0.7 s up to 4 parallel requests on one GPU).
 */
import { generate, type GenerateRequest, type Rendered } from '../api';

const MAX_PARALLEL = 4;
const CACHE_SIZE = 120;

const cache = new Map<string, Rendered>();
const inflight = new Map<string, Promise<Rendered>>();
let running = 0;
const queue: (() => void)[] = [];

/** Stable key of a request (object keys sorted at every level). */
export function requestKey(req: GenerateRequest): string {
  const norm = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(norm);
    if (v && typeof v === 'object') {
      return Object.fromEntries(
        Object.keys(v as object)
          .sort()
          .map((k) => [k, norm((v as Record<string, unknown>)[k])]),
      );
    }
    return v;
  };
  return JSON.stringify(norm(req));
}

export function cached(req: GenerateRequest): Rendered | undefined {
  return cache.get(requestKey(req));
}

function remember(key: string, r: Rendered) {
  cache.delete(key);
  cache.set(key, r);
  while (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value!);
}

function slot(signal?: AbortSignal): Promise<void> {
  const { promise, resolve, reject } = Promise.withResolvers<void>();
  const go = () => {
    signal?.removeEventListener('abort', drop);
    running++;
    resolve();
  };
  const drop = () => {
    const i = queue.indexOf(go);
    if (i >= 0) queue.splice(i, 1);
    reject(new DOMException('Aborted', 'AbortError'));
  };
  if (signal?.aborted) drop();
  else if (running < MAX_PARALLEL) go();
  else {
    queue.push(go);
    signal?.addEventListener('abort', drop, { once: true });
  }
  return promise;
}

function release() {
  running--;
  queue.shift()?.();
}

export interface RenderOptions {
  signal?: AbortSignal;
  onChunk?: (chunk: Float32Array, sr: number) => void;
}

/** Render (or reuse) audio for a request. Abort rejects with an AbortError. */
export async function render(req: GenerateRequest, opts: RenderOptions = {}): Promise<Rendered> {
  const key = requestKey(req);
  const hit = cache.get(key);
  if (hit) {
    remember(key, hit);
    opts.onChunk?.(hit.pcm, hit.sr);
    return hit;
  }
  // Join an identical request already running only when nobody needs its stream.
  const pending = inflight.get(key);
  if (pending && !opts.onChunk) return pending;
  const job = (async () => {
    await slot(opts.signal);
    try {
      const r = await generate(req, opts);
      remember(key, r);
      return r;
    } finally {
      release();
    }
  })();
  inflight.set(key, job);
  try {
    return await job;
  } finally {
    if (inflight.get(key) === job) inflight.delete(key);
  }
}

export const isAbort = (e: unknown) => e instanceof DOMException && e.name === 'AbortError';

export function errorText(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

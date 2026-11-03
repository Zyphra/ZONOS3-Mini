// Sharded model download (Cache Storage backed) and upload into GPU buffers.
import type { Gpu } from './device';

export interface TensorInfo {
  dtype: 'f32' | 'f16' | 'bf16' | 'q8' | 'q4';
  shape: number[];
  shard: number;
  offset: number;
  nbytes: number;
  scales_offset?: number;
  scales_nbytes?: number;
  host?: boolean;
}

export interface Manifest {
  format: string;
  model: string;
  meta: Record<string, unknown>;
  shards: string[];
  tensors: Record<string, TensorInfo>;
}

export interface GpuTensor {
  buf: GPUBuffer;
  scales?: GPUBuffer;
  dtype: TensorInfo['dtype'];
  shape: number[];
}

export interface Model {
  meta: Record<string, unknown>;
  t: Map<string, GpuTensor>;
  host: Map<string, Float32Array>;
  get(name: string): GpuTensor;
  has(name: string): boolean;
}

export type Progress = (p: { model: string; loaded: number; total: number }) => void;

export const CACHE_NAME = 'zonos3-web-models-v2';

/** Deletes stale model caches (older exports, and web-zonos3's `zonos3-models-*`, reuse the same shard URLs) once
 *  per context. */
let pruned: Promise<void> | null = null;

/** `?models=` when given, else `/models/` on this origin (a worker's `location` shares the page origin). */
export function modelsBaseUrl(): string {
  const loc = typeof location === 'undefined' ? undefined : location;
  const q = loc ? new URLSearchParams(loc.search).get('models') : null;
  if (q) return q.endsWith('/') ? q : q + '/';
  return new URL('/models/', loc?.origin ?? 'http://localhost/').href;
}

/** `cache: false` reads straight from the network (e.g. weights bundled in a browser extension, where a Cache Storage
 *  copy would only duplicate files already on disk). */
async function cachedFetch(url: string, onBytes: (n: number) => void, useCache = true): Promise<ArrayBuffer> {
  if (useCache && typeof caches !== 'undefined') {
    await (pruned ??= caches.keys()
      .then(ks => Promise.all(ks.filter(k => (k.startsWith('zonos3-web-models-') || k.startsWith('zonos3-models-')) && k !== CACHE_NAME).map(k => caches.delete(k))))
      .then(() => {}));
  }
  const cache = useCache && typeof caches !== 'undefined' ? await caches.open(CACHE_NAME) : null;
  const hit = cache ? await cache.match(url) : undefined;
  if (hit) {
    const buf = await hit.arrayBuffer();
    onBytes(buf.byteLength);
    return buf;
  }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`failed to fetch ${url}: ${res.status}`);
  const len = Number(res.headers.get('content-length') ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.byteLength;
    onBytes(value.byteLength);
  }
  if (len && got !== len) throw new Error(`failed to fetch ${url}: truncated (${got}/${len})`);
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.byteLength; }
  // Cache Storage is a best-effort copy: a full quota (or a storage error) must not fail the model load.
  if (cache) {
    try {
      await cache.put(url, new Response(out.slice().buffer, { headers: { 'content-type': 'application/octet-stream' } }));
    } catch (e) {
      console.warn(`zonos3-web: not caching ${url}:`, e);
    }
  }
  return out.buffer;
}

export async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
  return res.json() as Promise<T>;
}

export async function loadModel(gpu: Gpu, baseUrl: string, name: string, onProgress?: Progress, useCache = true): Promise<Model> {
  const dir = new URL(name + '/', baseUrl).href;
  const man = await fetchJson<Manifest>(dir + 'manifest.json');
  if (man.format !== 'zonos3-web-v1') throw new Error(`bad manifest format for ${name}`);
  const byShard: [string, TensorInfo][][] = man.shards.map(() => []);
  let total = 0;
  for (const [tn, ti] of Object.entries(man.tensors)) {
    byShard[ti.shard].push([tn, ti]);
    total = Math.max(total, 0) + ti.nbytes + (ti.scales_nbytes ?? 0);
  }
  let loaded = 0;
  const report = () => onProgress?.({ model: name, loaded: Math.min(loaded, total), total });
  report();
  const t = new Map<string, GpuTensor>();
  const host = new Map<string, Float32Array>();
  const dev = gpu.device;
  const upload = (data: ArrayBuffer, off: number, n: number, label: string) => {
    const b = dev.createBuffer({ size: Math.max(16, Math.ceil(n / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label });
    dev.queue.writeBuffer(b, 0, data, off, Math.ceil(n / 4) * 4);
    return b;
  };
  for (let s = 0; s < man.shards.length; s++) {
    const shardBytes = byShard[s].reduce((a, [, ti]) => a + ti.nbytes + (ti.scales_nbytes ?? 0), 0);
    let shardGot = 0;
    const data = await cachedFetch(dir + man.shards[s], n => {
      // scale raw shard bytes (incl. padding) onto tensor bytes
      shardGot += n;
      report();
    }, useCache);
    loaded += shardBytes;
    void shardGot;
    for (const [tn, ti] of byShard[s]) {
      if (ti.host) {
        host.set(tn, new Float32Array(data.slice(ti.offset, ti.offset + ti.nbytes)));
        continue;
      }
      const g: GpuTensor = { buf: upload(data, ti.offset, ti.nbytes, tn), dtype: ti.dtype, shape: ti.shape };
      if (ti.dtype === 'q8' || ti.dtype === 'q4') g.scales = upload(data, ti.scales_offset!, ti.scales_nbytes!, tn + '.scales');
      t.set(tn, g);
    }
    report();
  }
  await dev.queue.onSubmittedWorkDone();
  return {
    meta: man.meta,
    t,
    host,
    get(n: string) {
      const v = t.get(n);
      if (!v) throw new Error(`${name}: missing tensor '${n}'`);
      return v;
    },
    has(n: string) { return t.has(n); },
  };
}

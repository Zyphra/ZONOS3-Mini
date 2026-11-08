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
  /** 'h8': a codec matrix repacked to k-major f16 for the gemm_h8 kernels (src/gpu/h8.ts). */
  dtype: TensorInfo['dtype'] | 'h8';
  shape: number[];
}

export interface Model {
  meta: Record<string, unknown>;
  t: Map<string, GpuTensor>;
  host: Map<string, Float32Array>;
  get(name: string): GpuTensor;
  has(name: string): boolean;
}

/** One model's progress: tensor bytes in so far (cached or not) of `total`, and how many came over the network. */
export type Progress = (p: { model: string; loaded: number; total: number; fetched: number }) => void;

/** Models that load together (init: backbone + zac; a clip voice: encoder + teacher), as one readout. */
export interface LoadProgress {
  models: string[];
  loaded: number;
  total: number;
  /** Bytes that came over the network (the rest came from Cache Storage): a page's download rate. */
  fetched: number;
}

export const CACHE_NAME = 'zonos3-web-models-v5';

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
async function cachedFetch(url: string, onBytes: (n: number, network: boolean) => void, useCache = true): Promise<ArrayBuffer> {
  if (useCache && typeof caches !== 'undefined') {
    await (pruned ??= caches.keys()
      .then(ks => Promise.all(ks.filter(k => (k.startsWith('zonos3-web-models-') || k.startsWith('zonos3-models-')) && k !== CACHE_NAME).map(k => caches.delete(k))))
      .then(() => {}));
  }
  const cache = useCache && typeof caches !== 'undefined' ? await caches.open(CACHE_NAME) : null;
  const hit = cache ? await cache.match(url) : undefined;
  if (hit) {
    const buf = await hit.arrayBuffer();
    onBytes(buf.byteLength, false);
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
    onBytes(value.byteLength, true);
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

/** Bytes a model's tensors take (its shards hold them plus alignment padding): the unit of every progress readout. */
function modelBytes(man: Manifest): number {
  let n = 0;
  for (const ti of Object.values(man.tensors)) n += ti.nbytes + (ti.scales_nbytes ?? 0);
  return n;
}

/** Manifests read once per worker: a readout reads them before its models load, so its total is set from the start. */
const manifests = new Map<string, Promise<Manifest>>();

function fetchManifest(dir: string, name: string): Promise<Manifest> {
  let m = manifests.get(dir);
  if (!m) {
    m = fetchJson<Manifest>(dir + 'manifest.json').then(man => {
      if (man.format !== 'zonos3-web-v1') throw new Error(`bad manifest format for ${name}`);
      return man;
    });
    m.catch(() => manifests.delete(dir));   // a failed read is retried by the next load
    manifests.set(dir, m);
  }
  return m;
}

/**
 * One readout for models that load together: pass the returned callback to each of their loadModel calls. Their
 * manifests are read first, so `total` is the whole set's from the first readout; readouts come at most every 100 ms,
 * plus the last one.
 */
export async function batchProgress(baseUrl: string, names: string[], report?: (p: LoadProgress) => void): Promise<Progress | undefined> {
  if (!report) return undefined;
  const totals = await Promise.all(names.map(n => fetchManifest(new URL(n + '/', baseUrl).href, n).then(modelBytes)));
  const total = totals.reduce((a, b) => a + b, 0);
  const loaded = new Map<string, number>();
  const fetched = new Map<string, number>();
  let last = -Infinity;
  return p => {
    loaded.set(p.model, p.loaded);
    fetched.set(p.model, p.fetched);
    let l = 0;
    let f = 0;
    for (const v of loaded.values()) l += v;
    for (const v of fetched.values()) f += v;
    const now = performance.now();
    if (l < total && now - last < 100) return;
    last = now;
    report({ models: names, loaded: l, total, fetched: f });
  };
}

export async function loadModel(gpu: Gpu, baseUrl: string, name: string, onProgress?: Progress, useCache = true): Promise<Model> {
  const dir = new URL(name + '/', baseUrl).href;
  const man = await fetchManifest(dir, name);
  const byShard: [string, TensorInfo][][] = man.shards.map(() => []);
  for (const [tn, ti] of Object.entries(man.tensors)) byShard[ti.shard].push([tn, ti]);
  const total = modelBytes(man);
  // Done shards plus the bytes of the one in flight (raw shard bytes include padding: capped at its tensor bytes).
  let loaded = 0;
  let shardBytes = 0;
  let shardGot = 0;
  let fetched = 0;
  const report = () => onProgress?.({ model: name, loaded: Math.min(loaded + Math.min(shardGot, shardBytes), total), total, fetched });
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
    shardBytes = byShard[s].reduce((a, [, ti]) => a + ti.nbytes + (ti.scales_nbytes ?? 0), 0);
    shardGot = 0;
    const data = await cachedFetch(dir + man.shards[s], (n, network) => {
      shardGot += n;
      if (network) fetched += n;
      report();
    }, useCache);
    loaded += shardBytes;
    shardGot = 0;
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

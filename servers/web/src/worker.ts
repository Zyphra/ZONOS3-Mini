// Engine worker: owns the WebGPU device and the models, serves the v2 API (src/api/router.ts) one request at a time.
// The codec's host half (iSTFT, mixer, resampler) runs in the codec worker behind `codecPort`.
/// <reference lib="webworker" />
import type { Engine, DefaultVoice } from './api/engine';
import { Api } from './api/router';
import { Backbone } from './backbone';
import { BLOCK_FRAMES, ZacStream } from './zac';
import { RefEncoder } from './encoder';
import { parseGguf } from './gguf';
import { Gpu, initGpu } from './gpu/device';
import { batchProgress, fetchJson, loadModel, type LoadProgress } from './gpu/weights';
import { loadReferenceCodes } from './npy';
import { buildPrompt, modelHp, NCB, W, type ModelHp } from './prompt';
import { LaciHeads } from './request/laci';
import { controlsDocument, loadSteer, type SteerBundle } from './request/steer';
import { SpeakerEncoder } from './speaker';
import { RemoteSpecSink } from './spec-sink';
import { createTokenizer } from './tokenizer';

declare const self: DedicatedWorkerGlobalScope;

export type InMsg =
  | { type: 'init'; base: string; cache: boolean; voicesPcm: Record<string, Float32Array>; codecPort: MessagePort }
  | { type: 'req'; id: number; method: string; path: string; headers: Record<string, string>; body: ArrayBuffer | null; pcm?: Float32Array }
  | { type: 'abort'; id: number };

export type OutMsg =
  | ({ type: 'progress' } & LoadProgress)
  | { type: 'ready' }
  | { type: 'fatal'; message: string }
  | { type: 'res'; id: number; status: number; headers: Record<string, string> }
  | { type: 'chunk'; id: number; data: ArrayBuffer }
  | { type: 'end'; id: number };

const post = (m: OutMsg, transfer: Transferable[] = []) => self.postMessage(m, transfer);

let api: Api | null = null;
const aborted = new Set<number>();
let queue: Promise<void> = Promise.resolve();

/** A companion GGUF, or null when the export has none (404). */
async function companion(base: string, name: string): Promise<ArrayBuffer | null> {
  const r = await fetch(new URL('companions/' + name, base).href);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`failed to fetch companions/${name}: ${r.status}`);
  return r.arrayBuffer();
}

/** laci/index.json and every heads file it names; empty without LACI heads (404). */
async function loadLaci(base: string): Promise<Map<string, LaciHeads>> {
  const out = new Map<string, LaciHeads>();
  const r = await fetch(new URL('laci/index.json', base).href);
  if (r.status === 404) return out;
  if (!r.ok) throw new Error(`failed to fetch laci/index.json: ${r.status}`);
  const index = await r.json() as Record<string, string>;
  const files = new Map<string, LaciHeads>();
  for (const f of new Set(Object.values(index))) {
    const text = await (await fetch(new URL('laci/' + f, base).href)).text();
    files.set(f, LaciHeads.parse(text, f));
  }
  for (const [lang, f] of Object.entries(index)) out.set(lang, files.get(f)!);
  return out;
}

async function init(base: string, cache: boolean, voicesPcm: Record<string, Float32Array>, codecPort: MessagePort): Promise<void> {
  const device = await initGpu(message => post({ type: 'fatal', message }));
  const gpu = new Gpu(device);
  const progress = (p: LoadProgress) => post({ type: 'progress', ...p });
  const voicesBase = new URL('voices/', base).href;
  const voicesRaw = await fetchJson<{ id: string; label: string; file: string; codes: string }[]>(voicesBase + 'voices.json');
  const voices: DefaultVoice[] = await Promise.all(voicesRaw.map(async v => {
    const url = new URL(v.codes, voicesBase).href;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`failed to fetch ${url}: ${r.status}`);
    return { id: v.id, label: v.label, file: v.file, pcm: voicesPcm[v.id] ?? null, codes: loadReferenceCodes(await r.arrayBuffer()) };
  }));
  // backbone + zac as one readout (one bar, one ETA), in the order they load.
  const boot = await batchProgress(base, ['backbone', 'zac'], progress);
  const bbw = await loadModel(gpu, base, 'backbone', boot, cache);
  const zacw = await loadModel(gpu, base, 'zac', boot, cache);
  const hp = modelHp(bbw.meta);

  const steerBuf = await companion(base, 'steer.gguf');
  const steer: SteerBundle | null = steerBuf
    ? loadSteer(parseGguf(steerBuf), { nEmbd: hp.nEmbd, nLayer: hp.nLayer, speakerDim: hp.speakerDim }) : null;
  const laci = await loadLaci(base);

  const backbone = new Backbone(gpu, bbw, steer ? steer.sources : []);
  const zac = await ZacStream.create(gpu, zacw, p => new RemoteSpecSink(codecPort, p));
  const tokenizer = createTokenizer();
  await warmup(backbone, zac, hp);

  let spk: Promise<SpeakerEncoder> | null = null;
  let ref: Promise<RefEncoder> | null = null;
  const engine: Engine = {
    hp, modelFile: 'zonos3-ptq.gguf', backbone, codec: zac, tokenizer,
    steer, controlsDoc: steer ? controlsDocument(steer, hp.speakerDim) : null, laci,
    voices, headroomDb: Number(zacw.meta.headroom_db ?? 3),
    async embedSpeaker(pcm) {
      const enc = await (spk ??= batchProgress(base, ['spk'], progress).then(p => SpeakerEncoder.load(gpu, base, p, cache)).catch((e: unknown) => { spk = null; throw e; }));
      return enc.embed(pcm);
    },
    async encodeReference(pcm) {
      const enc = await (ref ??= batchProgress(base, ['encoder', 'teacher'], progress).then(p => RefEncoder.load(gpu, base, p, cache)).catch((e: unknown) => { ref = null; throw e; }));
      return enc.encode(pcm);
    },
  };
  api = new Api(engine);
}

/** Runs every program the first request needs once (both slots' prefill tiles, the cached 256-row chunk program plus a
 *  remainder via a 309-row reference prompt, one decode step per slot, a full codec chunk) so pipeline compilation and
 *  first-use costs land before 'ready' instead of in the first request. */
async function warmup(backbone: Backbone, zac: ZacStream, hp: ModelHp): Promise<void> {
  const tok = createTokenizer();
  const short = buildPrompt(tok, 'Hi.', { buckets: [], silencePrefill: 3 }, hp);
  const ref = buildPrompt(tok, 'Hi.', { referenceCodes: new Int32Array(300 * NCB), buckets: [], silencePrefill: 3 }, hp);
  for (const slot of [0, 1] as const) {
    await backbone.prefill(slot, short.ids, short.cond);
    await backbone.prefill(slot, ref.ids, ref.cond);
    await backbone.step(slot, new Int32Array(W), null, null);
  }
  await zac.chunk(new Int32Array(NCB * BLOCK_FRAMES), 0, BLOCK_FRAMES);
  zac.reset();
}

async function serve(m: Extract<InMsg, { type: 'req' }>): Promise<void> {
  const id = m.id;
  const sink = {
    start: (status: number, headers: Record<string, string>) => post({ type: 'res', id, status, headers }),
    chunk: (data: Uint8Array) => {
      const copy = data.slice().buffer;
      post({ type: 'chunk', id, data: copy }, [copy]);
    },
    end: () => post({ type: 'end', id }),
  };
  try {
    if (!api) {
      sink.start(503, { 'content-type': 'application/json' });
      sink.chunk(new TextEncoder().encode(JSON.stringify({ detail: 'engine not initialized' })));
      sink.end();
      return;
    }
    await api.handle({ method: m.method, path: m.path, headers: m.headers, body: m.body, pcm: m.pcm }, sink, () => aborted.has(id));
  } finally {
    aborted.delete(id);
  }
}

self.onmessage = (ev: MessageEvent<InMsg>) => {
  const m = ev.data;
  if (m.type === 'init') {
    init(m.base, m.cache, m.voicesPcm, m.codecPort).then(() => post({ type: 'ready' }),
      e => post({ type: 'fatal', message: e instanceof Error ? e.message : String(e) }));
  } else if (m.type === 'req') {
    queue = queue.then(() => serve(m));
  } else if (m.type === 'abort') {
    aborted.add(m.id);
  }
};

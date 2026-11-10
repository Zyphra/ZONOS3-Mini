// Engine worker: owns the WebGPU device and the models, serves the v2 API (src/api/router.ts) one request at a time.
// The codec's host half (iSTFT, mixer, resampler) runs in the codec worker behind `codecPort`.
/// <reference lib="webworker" />
import type { Engine, Decoder, DefaultVoice, EditModels } from './api/engine';
import { loadEmbeddingVector } from './api/speakers';
import { Api } from './api/router';
import { Aligner } from './aligner';
import { Backbone } from './backbone';
import { BLOCK_FRAMES, ZacStream } from './zac';
import { RefEncoder } from './encoder';
import { parseGguf } from './gguf';
import { Gpu, initGpu } from './gpu/device';
import { batchProgress, fetchJson, loadModel, type LoadProgress, type Manifest } from './gpu/weights';
import { loadReferenceCodes } from './npy';
import { buildPrompt, modelHp, NCB, type ModelHp } from './prompt';
import { LaciHeads } from './request/laci';
import { controlsDocument, loadSteer, type SteerBundle } from './request/steer';
import { defaultSampling } from './sampler';
import { SpeakerEncoder } from './speaker';
import { codecChannel, RemoteSpecSink } from './spec-sink';
import { createTokenizer } from './tokenizer';

declare const self: DedicatedWorkerGlobalScope;

export type InMsg =
  | { type: 'init'; base: string; cache: boolean; voicesPcm: Record<string, Float32Array>; codecPort: MessagePort }
  | {
    type: 'req'; id: number; method: string; path: string; headers: Record<string, string>; body: ArrayBuffer | null;
    pcm?: Float32Array; pcmSeconds?: number;
  }
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

/** A model directory's manifest meta, or null when the export has no such model (404). */
async function manifestMeta(base: string, name: string): Promise<Record<string, unknown> | null> {
  const r = await fetch(new URL(name + '/manifest.json', base).href);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`failed to fetch ${name}/manifest.json: ${r.status}`);
  const man: Manifest = await r.json();
  return man.meta;
}

/** Speech editing when the export has the speech-edit adapter (`edit`) and FuseAlign (`aligner`): each loads on the
 *  first request that needs it (its download reported like the encoders'). The speech-edit LM runs the adapter on the
 *  release backbone's GPU weights (Backbone.withEdit). */
async function editModels(gpu: Gpu, base: string, cache: boolean, release: Backbone,
                          progress: (p: LoadProgress) => void): Promise<EditModels | null> {
  const [editMeta, alignerMeta] = await Promise.all([manifestMeta(base, 'edit'), manifestMeta(base, 'aligner')]);
  if (!editMeta || !alignerMeta) return null;
  const hp = modelHp(editMeta);
  if (hp.spanEndId === null || hp.textExtraTokens < 1) return null;
  let bb: Promise<Backbone> | null = null;
  let aligner: Promise<Aligner> | null = null;
  return {
    hp,
    backbone() {
      return bb ??= batchProgress(base, ['edit'], progress)
        .then(p => loadModel(gpu, base, 'edit', p, cache))
        .then(edit => release.withEdit(edit))
        .catch((e: unknown) => { bb = null; throw e; });
    },
    async align(pcm16k, tokens) {
      const a = await (aligner ??= batchProgress(base, ['aligner'], progress).then(p => Aligner.load(gpu, base, p, cache))
        .catch((e: unknown) => { aligner = null; throw e; }));
      return a.align(pcm16k, tokens);
    },
  };
}

/** The default decoder's model directory `zac` comes from this GGUF. */
const DEFAULT_DECODER_CHECKPOINT = 'zac-keep2.gguf';
/** Further decoders (model directory, the GGUF it was exported from): each is listed when the export has it (its
 *  manifest read at startup, no weights) and loads on the first request that picks it. */
const EXTRA_DECODERS = [{ dir: 'zac-full_mr', checkpoint: 'zac.gguf' }];

/** A decoder model directory's id: its head variant. */
function decoderId(meta: Record<string, unknown>, dir: string): string {
  const id = meta['head.variant'];
  if (typeof id !== 'string' || !id) throw new Error(`${dir}: no head.variant in its manifest; re-export with pnpm export`);
  return id;
}

/** Every decoder a request may pick: the default (`zac`, loaded at startup) first, then each further decoder the export
 *  has. A further decoder gets its own GPU weights and programs (the default's gemm plan; the matrices are the same
 *  shapes) and its own codec port, so its host half (its mixer weights) lives beside the default's in the codec
 *  worker. */
async function decoderList(gpu: Gpu, base: string, cache: boolean, zac: ZacStream, zacMeta: Record<string, unknown>,
                           codecPort: MessagePort, progress: (p: LoadProgress) => void): Promise<Decoder[]> {
  const out: Decoder[] = [{
    id: decoderId(zacMeta, 'zac'), checkpoint: DEFAULT_DECODER_CHECKPOINT, headroomDb: Number(zacMeta.headroom_db ?? 3),
    codec: async () => zac,
  }];
  for (const { dir, checkpoint } of EXTRA_DECODERS) {
    const meta = await manifestMeta(base, dir);
    if (!meta) continue;
    const id = decoderId(meta, dir);
    if (out.some(d => d.id === id)) throw new Error(`${dir}: decoder ${id} is exported twice`);
    let stream: Promise<ZacStream> | null = null;
    out.push({
      id, checkpoint, headroomDb: Number(meta.headroom_db ?? 3),
      codec() {
        return stream ??= batchProgress(base, [dir], progress)
          .then(p => loadModel(gpu, base, dir, p, cache))
          .then(w => ZacStream.create(gpu, w, p => new RemoteSpecSink(codecChannel(codecPort), p), zac.gemmPlan))
          .then(async z => { await warmCodec(z); return z; })
          .catch((e: unknown) => { stream = null; throw e; });
      },
    });
  }
  return out;
}

async function init(base: string, cache: boolean, voicesPcm: Record<string, Float32Array>, codecPort: MessagePort): Promise<void> {
  const device = await initGpu(message => post({ type: 'fatal', message }));
  const gpu = new Gpu(device);
  const progress = (p: LoadProgress) => post({ type: 'progress', ...p });
  const voicesBase = new URL('voices/', base).href;
  const voicesRaw = await fetchJson<{ id: string; label: string; file: string; codes: string; embedding: string }[]>(
    voicesBase + 'voices.json');
  const fetchBytes = async (rel: string) => {
    const url = new URL(rel, voicesBase).href;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`failed to fetch ${url}: ${r.status}`);
    return r.arrayBuffer();
  };
  const voiceFiles = Promise.all(voicesRaw.map(v => Promise.all([fetchBytes(v.codes), fetchBytes(v.embedding)])));
  // backbone + zac as one readout (one bar, one ETA), in the order they load.
  const boot = await batchProgress(base, ['backbone', 'zac'], progress);
  const bbw = await loadModel(gpu, base, 'backbone', boot, cache);
  const zacw = await loadModel(gpu, base, 'zac', boot, cache);
  const hp = modelHp(bbw.meta);
  const voices: DefaultVoice[] = (await voiceFiles).map(([codes, emb], i) => {
    const v = voicesRaw[i];
    return {
      id: v.id, label: v.label, file: v.file, pcm: voicesPcm[v.id] ?? null, codes: loadReferenceCodes(codes),
      embedding: hp.speakerDim > 0 ? loadEmbeddingVector(new Uint8Array(emb), hp.speakerDim, v.embedding) : null,
    };
  });

  const steerBuf = await companion(base, 'steer.gguf');
  const steer: SteerBundle | null = steerBuf
    ? loadSteer(parseGguf(steerBuf), { nEmbd: hp.nEmbd, nLayer: hp.nLayer, speakerDim: hp.speakerDim }) : null;
  const laci = await loadLaci(base);

  const steerSources = steer ? steer.sources : [];
  const backbone = new Backbone(gpu, bbw, steerSources);
  const zac = await ZacStream.create(gpu, zacw, p => new RemoteSpecSink(codecPort, p));
  const tokenizer = createTokenizer();
  await warmup(backbone, zac, hp);
  const edit = await editModels(gpu, base, cache, backbone, progress);
  const decoders = await decoderList(gpu, base, cache, zac, zacw.meta, codecPort, progress);

  let spk: Promise<SpeakerEncoder> | null = null;
  let ref: Promise<RefEncoder> | null = null;
  const engine: Engine = {
    hp, modelFile: 'zonos3-ptq.gguf', backbone, decoders, tokenizer,
    steer, controlsDoc: steer ? controlsDocument(steer, hp.speakerDim) : null, laci, voices,
    async embedSpeaker(pcm) {
      const enc = await (spk ??= batchProgress(base, ['spk'], progress).then(p => SpeakerEncoder.load(gpu, base, p, cache)).catch((e: unknown) => { spk = null; throw e; }));
      return enc.embed(pcm);
    },
    async encodeReference(pcm) {
      const enc = await (ref ??= batchProgress(base, ['encoder', 'teacher'], progress).then(p => RefEncoder.load(gpu, base, p, cache)).catch((e: unknown) => { ref = null; throw e; }));
      return enc.encode(pcm);
    },
    edit,
  };
  api = new Api(engine);
}

/** Runs every program the first request needs once (both slots' prefill tiles, the cached 256-row chunk program plus a
 *  remainder via a 309-row reference prompt, a sampled decode step alone and as a CFG pair, a full codec chunk) so
 *  pipeline compilation and first-use costs land before 'ready' instead of in the first request. */
async function warmup(backbone: Backbone, zac: ZacStream, hp: ModelHp): Promise<void> {
  const tok = createTokenizer();
  const short = buildPrompt(tok, 'Hi.', { buckets: [], silencePrefill: 3 }, hp);
  const ref = buildPrompt(tok, 'Hi.', { referenceCodes: new Int32Array(300 * NCB), buckets: [], silencePrefill: 3 }, hp);
  for (const slot of [0, 1] as const) {
    await backbone.prefill(slot, short.ids, short.cond);
    await backbone.prefill(slot, ref.ids, ref.cond);
  }
  for (const cfgScale of [null, 1.5]) {
    backbone.startSampling({ sp: defaultSampling(), minFrames: 0, cfgScale });
    await backbone.sampleFirst(0);
    await backbone.stepSample(1, null, null, null);
  }
  await warmCodec(zac);
}

/** A full codec chunk, then a reset: the decoder's pipelines compile before its first request. */
async function warmCodec(zac: ZacStream): Promise<void> {
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
    await api.handle({ method: m.method, path: m.path, headers: m.headers, body: m.body, pcm: m.pcm, pcmSeconds: m.pcmSeconds },
      sink, () => aborted.has(id));
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

// Deterministic stub engines for the API tests: the real tokenizer and prompt layout with the backbone, codec and
// encoders replaced by stand-ins (EOA after a few frames, a sine per decoded frame). stubApi adds the exported models'
// hyper-parameters, steering bundle and LACI heads (public/models) and, with the exported speech-edit backbone and
// aligner, a stub speech-edit backbone (span-end after a few frames) and a stub aligner (words spread evenly over the
// clip); bareApi needs no exported models.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { ProbeInput } from '../../src/backbone';
import type { AlignedWord } from '../../src/aligner/host';
import type { EditModels, Engine } from '../../src/api/engine';
import { Api } from '../../src/api/router';
import type { EngineRequest, Sink } from '../../src/api/http';
import { parseGguf } from '../../src/gguf';
import { modelHp, type Conditioning, type ModelHp } from '../../src/prompt';
import { LaciHeads } from '../../src/request/laci';
import { controlsDocument, loadSteer } from '../../src/request/steer';
import { createTokenizer } from '../../src/tokenizer';

const root = fileURLToPath(new URL('../../', import.meta.url));
const models = root + 'public/models/';
export const haveModels = (() => { try { readFileSync(models + 'backbone/manifest.json'); return true; } catch { return false; } })();

export const haveEditModels = haveModels &&
  (() => { try { readFileSync(models + 'edit/manifest.json'); readFileSync(models + 'aligner/manifest.json'); return true; } catch { return false; } })();

const NCB = 10, HOP = 2940, EOA = 4096, SPAN_END = 4098;

function file(path: string): ArrayBuffer {
  const b = readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

/** Logits that emit code 7 for `frames` steps, then the stop (EOA, or span-end on the speech-edit vocabulary) on every
 *  codebook; code 7 stays the runner-up so a masked stop keeps the stream deterministic. */
class StubBackbone {
  steerSources: number[];
  private steps: [number, number] = [0, 0];
  steered: (Float32Array | null)[] = [];
  twinSteered: (Float32Array | null)[] = [];
  probes: (ProbeInput | null)[] = [];
  prefills: { slot: 0 | 1; rows: number; cond: Conditioning; ids: Int32Array }[] = [];
  constructor(sources: number[], private frames = 14, private vocab = 4098) { this.steerSources = sources; }
  private logits(slot: 0 | 1): Float32Array {
    const V = this.vocab;
    const l = new Float32Array(V * NCB).fill(-10);
    const stop = this.steps[slot] >= this.frames;
    for (let cb = 0; cb < NCB; cb++) {
      l[cb * V + 7] = 5;
      if (stop) l[cb * V + (V === 4099 ? SPAN_END : EOA)] = 10;
    }
    return l;
  }
  async prefill(slot: 0 | 1, ids: Int32Array, cond: Conditioning) {
    this.steps[slot] = 0;
    this.prefills.push({ slot, rows: ids.length / 11, cond, ids: ids.slice() });
    return { logits: this.logits(slot) };
  }
  private probeOut(probe: ProbeInput | null) {
    return probe ? new Float32Array(2048).fill(0).map((_, j) => (j === 1 ? 1 : 0)) : null;
  }
  async step(slot: 0 | 1, _next: Int32Array, steer: Float32Array | null, probe: ProbeInput | null) {
    this.steps[slot]++;
    if (slot === 0) { this.steered.push(steer); this.probes.push(probe); } else this.twinSteered.push(steer);
    return { logits: this.logits(slot), probe: this.probeOut(probe) };
  }
  async stepPair(_next: Int32Array, steer: Float32Array | null, twinSteer: Float32Array | null, probe: ProbeInput | null) {
    this.steps[0]++;
    this.steps[1]++;
    this.steered.push(steer);
    this.twinSteered.push(twinSteer);
    this.probes.push(probe);
    return { logits: this.logits(0), twinLogits: this.logits(1), probe: this.probeOut(probe) };
  }
}

/** A decoded chunk: its un-sheared indices, first frame, frame count and the backbone decode steps run before it. */
export interface StubChunk { idx: Int32Array; p0: number; n: number; step: number }

class StubCodec {
  readonly gain = Math.fround(10 ** (-3 / 20));
  /** Frames decoded since the last reset. */
  frames = 0;
  chunks: StubChunk[] = [];
  /** Backbone decode steps so far (set by stubApi). */
  clock: () => number = () => 0;
  constructor(readonly partialFrames = true) {}
  reset() { this.frames = 0; this.chunks = []; }
  async chunk(idx: Int32Array, p0: number, n: number) {
    this.frames += n;
    this.chunks.push({ idx: idx.slice(), p0, n, step: this.clock() });
    return new Float32Array(n * HOP).fill(0.5);
  }
  async flush() { return new Float32Array(0); }
}

/** FuseAlign stand-in: every token gets an equal share of the clip, presence 0.9. */
class StubAligner {
  calls: { seconds: number; tokens: string[] }[] = [];
  async align(pcm16k: Float32Array, tokens: string[]): Promise<AlignedWord[]> {
    const seconds = pcm16k.length / 16000;
    this.calls.push({ seconds, tokens });
    const w = seconds / Math.max(1, tokens.length);
    return tokens.map((word, k) => ({ word, start: k * w, end: (k + 1) * w, presence: 0.9, placed: true }));
  }
}

export interface EditStubs { backbone: StubBackbone; aligner: StubAligner; models: EditModels }

export interface StubSet { api: Api; engine: Engine; backbone: StubBackbone; codec: StubCodec; edit: EditStubs | null }

/** The stub speech-edit models over the exported `edit` backbone's hyper-parameters. */
function editStubs(sources: number[], frames: number): EditStubs {
  const hp = modelHp(JSON.parse(readFileSync(models + 'edit/manifest.json', 'utf8')).meta);
  const backbone = new StubBackbone(sources, frames, hp.audioVocab);
  const aligner = new StubAligner();
  return { backbone, aligner, models: { hp, backbone: async () => backbone, align: (pcm, tokens) => aligner.align(pcm, tokens) } };
}

export function stubApi(opts: { frames?: number; editFrames?: number; partialFrames?: boolean } = {}): StubSet {
  const man = JSON.parse(readFileSync(models + 'backbone/manifest.json', 'utf8'));
  const hp = modelHp(man.meta);
  const g = (name: string) => parseGguf(file(models + 'companions/' + name));
  const steer = loadSteer(g('steer.gguf'), { nEmbd: hp.nEmbd, nLayer: hp.nLayer, speakerDim: hp.speakerDim });
  const index: Record<string, string> = JSON.parse(readFileSync(models + 'laci/index.json', 'utf8'));
  const laci = new Map<string, LaciHeads>();
  for (const [lang, f] of Object.entries(index)) laci.set(lang, LaciHeads.parse(readFileSync(models + 'laci/' + f, 'utf8'), f));
  const voices = JSON.parse(readFileSync(models + 'voices/voices.json', 'utf8')) as { id: string; label: string; file: string }[];
  const backbone = new StubBackbone(steer.sources, opts.frames);
  const codec = new StubCodec(opts.partialFrames);
  codec.clock = () => backbone.steered.length;
  const edit = haveEditModels ? editStubs(steer.sources, opts.editFrames ?? 6) : null;
  const engine: Engine = {
    hp, modelFile: 'zonos3-ptq.gguf', backbone, codec, tokenizer: createTokenizer(),
    steer, controlsDoc: controlsDocument(steer, hp.speakerDim), laci,
    voices: voices.map(v => ({
      ...v, pcm: new Float32Array(44100).fill(0.1), codes: new Int32Array(20).fill(5),
      embedding: new Float32Array(hp.speakerDim).fill(0.2),
    })),
    headroomDb: 3,
    async embedSpeaker(pcm) { return new Float32Array(hp.speakerDim).fill(pcm[0] ?? 0.1); },
    async encodeReference(pcm) { return new Int32Array(Math.ceil(pcm.length / HOP) * NCB).fill(3); },
    edit: edit?.models ?? null,
  };
  return { api: new Api(engine, 1000), engine, backbone, codec, edit };
}

/** Release-like checkpoint hyper-parameters (ZONOS3 Mini, zonos3-rate30-20k-r2grpo): condition rows of 10, 10 and 30
 *  buckets, the speaker row, no phoneme tokens. */
const RELEASE_HP: ModelHp = {
  nCodebooks: NCB, audioPad: 4097, textVocab: 8196, audioVocab: 4098, spanEndId: null, textExtraTokens: 0, nCond: 3,
  condBucketCounts: [10, 10, 30],
  condNames: ['loudness', 'fmax', 'speaking_rate'],
  condEdges: [
    [-29.002, -26.861, -25.366, -23.607, -21.916, -20.505, -19.38, -18.151, -16.297],
    [7578.125, 8000, 8000.01, 11062.5, 12445.94, 15328.125, 15773.438, 16623.633, 18758.496],
    [
      6.546, 8.182, 9.273, 10.31, 11.054, 11.667, 12.193, 12.656, 13.083, 13.476, 13.846, 14.196, 14.53, 14.887, 15.193,
      15.517, 15.865, 16.208, 16.579, 16.972, 17.411, 17.898, 18.434, 19.069, 19.818, 20.816, 22.286, 24.722, 29.25,
    ],
  ],
  silenceFrame: [1918, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337], speakerDim: 192, contextLength: 32768,
  condUnits: ['LUFS', 'Hz', 'utf-8 bytes per second'], textTokenizerSpec: '{}', tagCfgTrained: false,
  nEmbd: 1024, nLayer: 11,
};

/** A stub engine without exported models: RELEASE_HP, no steering bundle, no LACI heads, no accent adapters, no
 *  speech editing, one default voice. */
export function bareApi(opts: { frames?: number; partialFrames?: boolean } = {}): StubSet {
  const hp = RELEASE_HP;
  const backbone = new StubBackbone([], opts.frames);
  const codec = new StubCodec(opts.partialFrames);
  codec.clock = () => backbone.steered.length;
  const engine: Engine = {
    hp, modelFile: 'zonos3-ptq.gguf', backbone, codec, tokenizer: createTokenizer(),
    steer: null, controlsDoc: null, laci: new Map(),
    voices: [{ id: 'v0', label: 'Voice 0', file: 'v0.mp3', pcm: new Float32Array(44100).fill(0.1), codes: new Int32Array(20).fill(5), embedding: null }],
    headroomDb: 3,
    async embedSpeaker() { return new Float32Array(hp.speakerDim).fill(0.1); },
    async encodeReference(pcm) { return new Int32Array(Math.ceil(pcm.length / HOP) * NCB).fill(3); },
    edit: null,
  };
  return { api: new Api(engine, 1000), engine, backbone, codec, edit: null };
}

/** A response; `json<T>()` parses the body as the caller's expected shape (tests then assert on it). */
export interface Reply { status: number; headers: Record<string, string>; body: Uint8Array; json<T = Record<string, unknown>>(): T }

export async function call(api: Api, method: string, path: string, body?: unknown,
                           headers: Record<string, string> = {}, pcm?: Float32Array, pcmSeconds?: number): Promise<Reply> {
  let status = 0, hdrs: Record<string, string> = {};
  const parts: Uint8Array[] = [];
  const sink: Sink = { start(s, h) { status = s; hdrs = h; }, chunk(d) { parts.push(d.slice()); }, end() {} };
  const req: EngineRequest = {
    method, path, headers,
    body: body === undefined ? null : new TextEncoder().encode(typeof body === 'string' ? body : JSON.stringify(body)).buffer,
    pcm, pcmSeconds,
  };
  await api.handle(req, sink);
  const n = parts.reduce((a, b) => a + b.length, 0);
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return { status, headers: hdrs, body: out, json: () => JSON.parse(new TextDecoder().decode(out)) };
}

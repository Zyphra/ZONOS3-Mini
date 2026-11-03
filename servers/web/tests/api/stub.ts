// Deterministic stub engines for the API tests: the real tokenizer and prompt layout with the backbone, codec and
// encoders replaced by stand-ins (EOA after a few frames, a sine per decoded frame). stubApi adds the exported models'
// hyper-parameters, steering bundle and LACI heads (public/models); bareApi needs no exported models.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { ProbeInput } from '../../src/backbone';
import type { Engine } from '../../src/api/engine';
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

const VOCAB = 4098, NCB = 10, HOP = 2940, EOA = 4096;

function file(path: string): ArrayBuffer {
  const b = readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

/** Logits that emit codes 7 for `frames` steps, then EOA on every codebook. */
class StubBackbone {
  steerSources: number[];
  private steps: [number, number] = [0, 0];
  steered: (Float32Array | null)[] = [];
  twinSteered: (Float32Array | null)[] = [];
  probes: (ProbeInput | null)[] = [];
  prefills: { slot: 0 | 1; rows: number; cond: Conditioning }[] = [];
  constructor(sources: number[], private frames = 14) { this.steerSources = sources; }
  private logits(slot: 0 | 1): Float32Array {
    const l = new Float32Array(VOCAB * NCB).fill(-10);
    const eoa = this.steps[slot] >= this.frames;
    for (let cb = 0; cb < NCB; cb++) l[cb * VOCAB + (eoa ? EOA : 7)] = 10;
    return l;
  }
  async prefill(slot: 0 | 1, ids: Int32Array, cond: Conditioning) {
    this.steps[slot] = 0;
    this.prefills.push({ slot, rows: ids.length / 11, cond });
    return { logits: this.logits(slot) };
  }
  async step(slot: 0 | 1, _next: Int32Array, steer: Float32Array | null, probe: ProbeInput | null) {
    this.steps[slot]++;
    if (slot === 0) { this.steered.push(steer); this.probes.push(probe); } else this.twinSteered.push(steer);
    return { logits: this.logits(slot), probe: probe ? new Float32Array(2048).fill(0).map((_, j) => (j === 1 ? 1 : 0)) : null };
  }
}

class StubCodec {
  readonly gain = Math.fround(10 ** (-3 / 20));
  private frames = 0;
  reset() { this.frames = 0; }
  async chunk(_idx: Int32Array, _p0: number, n: number) {
    this.frames += n;
    return new Float32Array(n * HOP).fill(0.5);
  }
  flush() { return new Float32Array(0); }
}

export interface StubSet { api: Api; engine: Engine; backbone: StubBackbone }

export function stubApi(opts: { frames?: number } = {}): StubSet {
  const man = JSON.parse(readFileSync(models + 'backbone/manifest.json', 'utf8'));
  const hp = modelHp(man.meta);
  const g = (name: string) => parseGguf(file(models + 'companions/' + name));
  const steer = loadSteer(g('steer.gguf'), { nEmbd: hp.nEmbd, nLayer: hp.nLayer, speakerDim: hp.speakerDim });
  const index: Record<string, string> = JSON.parse(readFileSync(models + 'laci/index.json', 'utf8'));
  const laci = new Map<string, LaciHeads>();
  for (const [lang, f] of Object.entries(index)) laci.set(lang, LaciHeads.parse(readFileSync(models + 'laci/' + f, 'utf8'), f));
  const voices = JSON.parse(readFileSync(models + 'voices/voices.json', 'utf8')) as { id: string; label: string; file: string }[];
  const backbone = new StubBackbone(steer.sources, opts.frames);
  const engine: Engine = {
    hp, modelFile: 'zonos3-ptq.gguf', backbone, codec: new StubCodec(), tokenizer: createTokenizer(),
    steer, controlsDoc: controlsDocument(steer, hp.speakerDim), laci,
    voices: voices.map(v => ({ ...v, pcm: new Float32Array(44100).fill(0.1), codes: new Int32Array(20).fill(5) })),
    headroomDb: 3,
    async embedSpeaker(pcm) { return new Float32Array(hp.speakerDim).fill(pcm[0] ?? 0.1); },
    async encodeReference(pcm) { return new Int32Array(Math.ceil(pcm.length / HOP) * NCB).fill(3); },
  };
  return { api: new Api(engine, 1000), engine, backbone };
}

/** Release-like checkpoint hyper-parameters (ZONOS3 Mini, zonos3-rate30-20k-r2grpo): condition rows of 10, 10 and 30
 *  buckets, the speaker row, no phoneme tokens. */
const RELEASE_HP: ModelHp = {
  nCodebooks: NCB, audioPad: 4097, textVocab: 8196, nCond: 3, condBucketCounts: [10, 10, 30],
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

/** A stub engine without exported models: RELEASE_HP, no steering bundle, no LACI heads, no accent adapters, one
 *  default voice. */
export function bareApi(): StubSet {
  const hp = RELEASE_HP;
  const backbone = new StubBackbone([]);
  const engine: Engine = {
    hp, modelFile: 'zonos3-ptq.gguf', backbone, codec: new StubCodec(), tokenizer: createTokenizer(),
    steer: null, controlsDoc: null, laci: new Map(),
    voices: [{ id: 'v0', label: 'Voice 0', file: 'v0.mp3', pcm: new Float32Array(44100).fill(0.1), codes: new Int32Array(20).fill(5) }],
    headroomDb: 3,
    async embedSpeaker() { return new Float32Array(hp.speakerDim).fill(0.1); },
    async encodeReference(pcm) { return new Int32Array(Math.ceil(pcm.length / HOP) * NCB).fill(3); },
  };
  return { api: new Api(engine, 1000), engine, backbone };
}

/** A response; `json<T>()` parses the body as the caller's expected shape (tests then assert on it). */
export interface Reply { status: number; headers: Record<string, string>; body: Uint8Array; json<T = Record<string, unknown>>(): T }

export async function call(api: Api, method: string, path: string, body?: unknown,
                           headers: Record<string, string> = {}, pcm?: Float32Array): Promise<Reply> {
  let status = 0, hdrs: Record<string, string> = {};
  const parts: Uint8Array[] = [];
  const sink: Sink = { start(s, h) { status = s; hdrs = h; }, chunk(d) { parts.push(d.slice()); }, end() {} };
  const req: EngineRequest = {
    method, path, headers,
    body: body === undefined ? null : new TextEncoder().encode(typeof body === 'string' ? body : JSON.stringify(body)).buffer,
    pcm,
  };
  await api.handle(req, sink);
  const n = parts.reduce((a, b) => a + b.length, 0);
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return { status, headers: hdrs, body: out, json: () => JSON.parse(new TextDecoder().decode(out)) };
}

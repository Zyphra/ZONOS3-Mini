// The GPU sampler (kernels/sample.wgsl): Sampler.sample (src/sampler.ts, its host oracle) plus a frame's cold start,
// recorded once over a decode's logits buffers and encoded after the decode in the same submission. It writes the
// sampled row into the decode input buffer, holds the request's random stream (MT19937 seeded as the host sampler: two
// draws per codebook and frame unless greedy) and the per-codebook repetition rings on the device.
import { CONTEXT_CAPACITY } from '../generation';
import { canonical53, MT19937 } from '../mt19937';
import { TEXT_PAD } from '../prompt';
import { vocabIds, type SamplerVocab, type Sampling } from '../sampler';
import type { Gpu } from './device';
import { Builder, type Program } from './program';

/** A request's sampling: the Sampler settings and vocabulary layout, the stop gate (the stop id waits until
 *  minFrames + cb frames were sampled), the CFG scale (null = the conditional logits alone, else
 *  twin + cfgScale * (cond - twin)) and frames already in the stream that feed the repetition penalty (a LACI resume:
 *  its last rep_window rows count). */
export interface SamplingSetup {
  sp: Sampling;
  vocab?: SamplerVocab;
  minFrames: number;
  cfgScale: number | null;
  history?: ArrayLike<number>[];
}

/** sample.wgsl's uniform Smp: 20 scalar words, u[] as 3 vec4 (at most 12 codebooks), then stop_bias (padded to 36). */
const SMP_WORDS = 36;
const SMP_U = 20;
const SMP_STOP_BIAS = 32;
const MAX_CODEBOOKS = 12;
/** sample.wgsl MAXC: ids per thread of its 256-thread workgroup. */
const MAX_IDS_PER_THREAD = 20;
const NEG_INF_BITS = 0xff800000;
const f = Math.fround;

export class GpuSampler {
  private uni: GPUBuffer;
  private words = new Uint32Array(SMP_WORDS);
  private floats = new Float32Array(this.words.buffer);
  private cnt: GPUBuffer;
  private hist: GPUBuffer | null = null;
  private W = 0;
  /** The recorded dispatch without / with CFG (the ring buffer is bound, so a new ring size re-records). */
  private progs: [Program | null, Program | null] = [null, null];
  private rng: MT19937 | null = null;
  private greedy = false;
  private generated = 0;
  /** Whether the running request samples CFG-guided logits (both decode slots run each step). */
  guided = false;

  /** `logits` / `twin`: the conditional and twin logits [ncb][vocab]; `ids`: the decode input row [ncb + 1] it writes. */
  constructor(private gpu: Gpu, private vocab: number, private ncb: number, private logits: GPUBuffer,
              private twin: GPUBuffer, private ids: GPUBuffer) {
    if (vocab > 256 * MAX_IDS_PER_THREAD) throw new Error(`sampler: the GPU sampler holds at most ${256 * MAX_IDS_PER_THREAD} ids per codebook`);
    if (ncb > MAX_CODEBOOKS) throw new Error(`sampler: the GPU sampler holds at most ${MAX_CODEBOOKS} codebooks`);
    this.uni = gpu.uniform(SMP_WORDS * 4, 'smp');
    this.cnt = gpu.buffer(ncb * 4, 'smp.cnt');
  }

  /** Starts a request's sampling: the random stream from sp.seed, the repetition rings from `history`, 0 frames sampled. */
  start(s: SamplingSetup): void {
    const sp = s.sp, v = vocabIds(this.vocab, s.vocab), ncb = this.ncb;
    const win = Math.min(Math.max(0, Math.floor(sp.rep_window)), CONTEXT_CAPACITY);
    const W = Math.max(1, win);
    if (W !== this.W) {
      this.hist?.destroy();
      for (const p of this.progs) p?.destroy();
      this.progs = [null, null];
      this.hist = this.gpu.buffer(ncb * W * 4, 'smp.hist');
      this.W = W;
    }
    const rows = s.history ?? [];
    const n = Math.min(win, rows.length);
    const ring = new Uint32Array(ncb * W);
    for (let k = 0; k < n; k++) {
      const r = rows[rows.length - n + k];
      for (let cb = 0; cb < ncb; cb++) ring[cb * W + k] = r[cb];
    }
    this.gpu.write(this.hist!, 0, ring);
    this.gpu.write(this.cnt, 0, new Uint32Array(ncb).fill(n));

    const w = this.words, fl = this.floats;
    w.fill(0);
    w[0] = this.vocab;
    w[1] = v.codebookSize;
    w[2] = v.padId;
    w[3] = v.stopId;
    w[4] = Math.max(0, s.minFrames);
    w[7] = TEXT_PAD;
    w[8] = sp.greedy ? 1 : 0;
    fl[9] = f(sp.temperature);
    w[10] = sp.top_k > 0 ? Math.min(Math.floor(sp.top_k), this.vocab) : 0;
    fl[11] = f(sp.top_p);
    fl[12] = f(sp.min_p);
    fl[13] = f(sp.rep_penalty);
    w[14] = win;
    w[15] = Math.max(0, sp.rep_codebooks);
    w[16] = s.cfgScale === null ? 0 : 1;
    fl[17] = f(s.cfgScale ?? 0);
    w[18] = W;
    w[19] = NEG_INF_BITS;
    this.rng = new MT19937(sp.seed >>> 0);
    this.greedy = sp.greedy;
    this.guided = s.cfgScale !== null;
    this.generated = 0;
  }

  /** The next frame's draw at stream row `row` (codebooks past it get the audio pad): writes its uniform (ordered after
   *  every earlier submission) and returns the program to encode after the decode that fills the logits. `stopBias`:
   *  added to the stop id's logit in every codebook before the filters (LACI's stop boost). */
  next(row: number, stopBias = 0): Program {
    if (!this.rng) throw new Error('backbone: startSampling must run before sampling');
    const w = this.words;
    w[5] = this.generated++;
    w[6] = row;
    this.floats[SMP_STOP_BIAS] = f(stopBias);
    if (!this.greedy) for (let cb = 0; cb < this.ncb; cb++) this.floats[SMP_U + cb] = canonical53(this.rng);
    this.gpu.write(this.uni, 0, w);
    const i = this.guided ? 1 : 0;
    let p = this.progs[i];
    if (!p) {
      const b = new Builder(this.gpu);
      b.dispatch('sample', { logits: this.logits, twin: this.guided ? this.twin : null, hist: this.hist, cnt: this.cnt, ids: this.ids, smp: this.uni },
        {}, [this.ncb, 1, 1]);
      p = this.progs[i] = b.finish();
    }
    return p;
  }
}

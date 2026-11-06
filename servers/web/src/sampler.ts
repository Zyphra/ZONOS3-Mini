// Port of zonos3-sampler.cpp: repetition penalty, temperature, top-k, top-p, min-p, multinomial.
import { SAMPLING } from './generation';
import { MT19937, uniform } from './mt19937';

export interface Sampling {
  temperature: number;
  top_k: number;
  top_p: number;
  min_p: number;
  rep_penalty: number;
  rep_window: number;
  rep_codebooks: number;
  seed: number;
  greedy: boolean;
}

/** The registry's sampling defaults (seed 0, sampled). */
export function defaultSampling(): Sampling {
  return {
    temperature: SAMPLING.temperature, top_k: SAMPLING.topk, top_p: SAMPLING.top_p, min_p: SAMPLING.min_p,
    rep_penalty: SAMPLING.repetition_penalty, rep_window: SAMPLING.repetition_window,
    rep_codebooks: SAMPLING.repetition_codebooks, seed: 0, greedy: false,
  };
}

const f = Math.fround;

/** The k-th largest value of `a` (1 ≤ k ≤ a.length): exactly `Float32Array.from(a).sort()[a.length - k]`, ties and
 *  ±Infinity included (NaN-free input). Quickselect (Hoare partition, middle pivot) in `scratch` (length ≥ a.length). */
export function kthLargest(a: Float32Array, k: number, scratch: Float32Array): number {
  const n = a.length;
  if (!(k >= 1 && k <= n)) throw new RangeError(`kthLargest: k=${k} outside 1..${n}`);
  const s = scratch;
  s.set(a);
  const target = n - k;
  let lo = 0, hi = n - 1;
  while (lo < hi) {
    const pivot = s[(lo + hi) >>> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (s[i] < pivot) i++;
      while (s[j] > pivot) j--;
      if (i <= j) {
        const t = s[i]; s[i] = s[j]; s[j] = t;
        i++; j--;
      }
    }
    // now [lo, j] <= pivot <= [i, hi], and every index strictly between j and i holds the pivot
    if (target <= j) hi = j;
    else if (target >= i) lo = i;
    else return s[target];
  }
  return s[target];
}

/** spanEnd: the vocabulary ends with span-end after the pad (the edit vocabulary, audio_vocab 4099). spanEndStop (implies
 *  spanEnd): span-end is the stop id and EOA is never sampled (speech-edit fills, zonos3-sampler.cpp stop_id). */
export interface SamplerVocab { spanEnd?: boolean; spanEndStop?: boolean }

export class Sampler {
  private rng: MT19937;
  private hist: number[][];
  private logit: Float32Array;
  private kthBuf: Float32Array;
  private seen: Uint32Array;     // repetition penalty: seen[t] === seenGen marks t penalized in this call
  private seenGen = 0;
  private codebookSize: number;
  /** EOA (end of audio), the audio pad (never sampled) and span-end (-1 without one). */
  readonly eoaId: number;
  readonly padId: number;
  readonly spanEndId: number;
  stopId: number;
  minFrames = 0;
  generated = 0;

  constructor(private sp: Sampling, private ncb = 10, private vocab = 4098, opts: SamplerVocab = {}) {
    this.rng = new MT19937(sp.seed >>> 0);
    this.hist = Array.from({ length: ncb }, () => []);
    this.logit = new Float32Array(vocab);
    this.kthBuf = new Float32Array(vocab);
    this.seen = new Uint32Array(vocab);
    const spanEnd = opts.spanEnd === true || opts.spanEndStop === true;
    this.codebookSize = vocab - (spanEnd ? 3 : 2);
    if (this.codebookSize < 1) throw new Error(`sampler: vocabulary of ${vocab} has no audio codes`);
    this.eoaId = this.codebookSize;
    this.padId = this.codebookSize + 1;
    this.spanEndId = spanEnd ? this.codebookSize + 2 : -1;
    this.stopId = opts.spanEndStop ? this.spanEndId : this.eoaId;
  }

  accept(frame: ArrayLike<number>): void {
    for (let cb = 0; cb < this.ncb && cb < frame.length; cb++) {
      const h = this.hist[cb];
      h.push(frame[cb]);
      while (h.length > this.sp.rep_window) h.shift();
    }
    this.generated++;
  }

  /** `logits` holds all codebooks; this samples codebook `cb` from [cb*vocab, (cb+1)*vocab).
   *  Order (zonos3-sampler.cpp): pad/special ids masked, repetition penalty, greedy, temperature, top-k with ties,
   *  softmax, top-p on the mass before each token, min-p, multinomial. */
  sample(logits: Float32Array, cb: number, offset = cb * this.vocab): number {
    const sp = this.sp, V = this.vocab;
    const logit = this.logit;
    logit.set(logits.subarray(offset, offset + V));
    // the audio pad is never a target; special ids other than stopId (EOA, span-end) are never sampled, and stopId
    // waits for minFrames
    for (let i = this.codebookSize; i < V; i++)
      if (i === this.padId || i !== this.stopId || this.generated < this.minFrames + cb) logit[i] = -Infinity;

    if (sp.rep_penalty !== 1 && cb < sp.rep_codebooks) {
      const pen = f(sp.rep_penalty);
      const seen = this.seen, gen = ++this.seenGen;
      for (const t of this.hist[cb]) {
        if (t >= 0 && t < this.codebookSize && seen[t] !== gen) {
          seen[t] = gen;
          logit[t] = logit[t] > 0 ? f(logit[t] / pen) : f(logit[t] * pen);
        }
      }
    }

    if (sp.greedy) {
      let best = 0, bv = logit[0];
      for (let i = 1; i < V; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
      return best;
    }

    if (sp.temperature > 0) {
      const temp = f(sp.temperature);
      for (let i = 0; i < V; i++) logit[i] = f(logit[i] / temp);
    }

    // candidate set: every logit >= the k-th largest (ties at the threshold are kept)
    const idx: number[] = [];
    if (sp.top_k > 0 && sp.top_k < V) {
      const kth = kthLargest(logit, sp.top_k, this.kthBuf);
      for (let i = 0; i < V; i++) if (logit[i] >= kth) idx.push(i);
    } else {
      for (let i = 0; i < V; i++) idx.push(i);
    }

    let mx = -Infinity;
    for (const i of idx) if (logit[i] > mx) mx = logit[i];
    const n = idx.length;
    const p = new Float32Array(n);
    let sum = 0;
    for (let j = 0; j < n; j++) { p[j] = f(Math.exp(f(logit[idx[j]] - mx))); sum += p[j]; }
    for (let j = 0; j < n; j++) p[j] = f(p[j] / sum);

    const ord = Array.from({ length: n }, (_, j) => j);
    ord.sort((a, b) => p[b] - p[a] || a - b);

    const keep = new Uint8Array(n).fill(1);
    if (sp.top_p > 0 && sp.top_p < 1) {
      const tp = f(sp.top_p);
      let before = 0;
      for (const r of ord) { if (before > tp) keep[r] = 0; before += p[r]; }
    }
    if (sp.min_p > 0) {
      const thr = f(f(sp.min_p) * p[ord[0]]);
      for (let j = 0; j < n; j++) if (p[j] < thr) keep[j] = 0;
    }

    let s2 = 0;
    for (let j = 0; j < n; j++) { if (!keep[j]) p[j] = 0; s2 += p[j]; }
    if (s2 <= 0) return idx[ord[0]];

    const r = uniform(this.rng, s2);
    let acc = 0;
    for (let j = 0; j < n; j++) { acc += p[j]; if (acc >= r) return idx[j]; }
    return idx[ord[0]];
  }
}

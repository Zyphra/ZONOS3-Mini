// Host-side FuseAlign pieces (no GPU): torchaudio resampling to 16 kHz, the byte tokenizer, waveform standardization,
// RoPE tables and fusealign_hub's post-processing (fill_unplaced, zero-width punctuation). Mirrors
// third_party/fusealign-multilingual/ggml/src/{resample,model,fusealign}.cpp and fusealign_hub/{aligner,longform}.py.

export const SAMPLE_RATE = 16000;
/** Longest clip (seconds) one forward aligns; longer clips need long-form alignment (not available in the browser). */
export const ALIGN_MAX_SECONDS = 30;
/** Longest transcript (UTF-8 bytes of the wordy tokens joined by spaces) one forward aligns. */
export const ALIGN_MAX_BYTES = 1022;

const f = Math.fround;
const LOWPASS_FILTER_WIDTH = 6;
const ROLLOFF = 0.99;

function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b];
  return a;
}

/**
 * torchaudio.functional.resample(pcm, sr, 16000) with its defaults (sinc_interp_hann, lowpass_filter_width 6, rolloff
 * 0.99) for a float32 waveform: the kernel built in float32 as torchaudio builds it, then its strided conv1d (pad width
 * left, width + orig right; ceil(16000 * n / sr) samples kept). Taps whose time argument torchaudio clamps to the filter
 * edge carry a Hann window of cos(pi/2)^2 ~ 2e-15 in float32 and are skipped.
 */
export function resampleTo16k(pcm: Float32Array, sr: number): Float32Array {
  if (!(sr > 0) || !Number.isInteger(sr)) throw new Error(`resampleTo16k: invalid sample rate ${sr}`);
  if (sr === SAMPLE_RATE) return pcm.slice();
  const g = gcd(sr, SAMPLE_RATE);
  const orig = sr / g, target = SAMPLE_RATE / g;
  const baseFreq = Math.min(orig, target) * ROLLOFF;
  const width = Math.ceil(LOWPASS_FILTER_WIDTH * orig / baseFreq);
  const taps = 2 * width + orig;
  const baseF = f(baseFreq), piF = f(Math.PI), scaleF = f(baseFreq / orig), limit = LOWPASS_FILTER_WIDTH;
  const kernels = new Float32Array(target * taps);
  const lo = new Int32Array(target), hi = new Int32Array(target);
  for (let phase = 0; phase < target; phase++) {
    const shift = f(-phase / target);
    let first = taps, last = 0;
    for (let k = 0; k < taps; k++) {
      const idx = f((k - width) / orig);
      let t = f(f(shift + idx) * baseF);
      if (t <= -limit || t >= limit) continue;
      const c = f(Math.cos(f(f(f(t * piF) / limit) / 2)));
      const window = f(c * c);
      t = f(t * piF);
      const sinc = t === 0 ? 1 : f(f(Math.sin(t)) / t);
      kernels[phase * taps + k] = f(sinc * f(window * scaleF));
      first = Math.min(first, k);
      last = k + 1;
    }
    lo[phase] = first;
    hi[phase] = Math.max(first, last);
  }
  const n = pcm.length;
  const paddedLen = width + n + width + orig;
  const frames = Math.floor((paddedLen - taps) / orig) + 1;
  const outLen = Math.min(Math.ceil(target * n / orig), frames * target);
  const out = new Float32Array(outLen);
  for (let o = 0; o < outLen; o++) {
    const frame = Math.floor(o / target), phase = o - frame * target;
    const x0 = frame * orig - width;   // padded index frame*orig is input index frame*orig - width
    const kb = phase * taps;
    let acc = 0;
    for (let k = lo[phase]; k < hi[phase]; k++) {
      const j = x0 + k;
      if (j >= 0 && j < n) acc += kernels[kb + k] * pcm[j];
    }
    out[o] = acc;
  }
  return out;
}

const utf8 = new TextEncoder();

/** fusealign_hub _is_wordy: the token has a letter or a digit (Unicode categories L*, N*). */
export function isWordy(token: string): boolean {
  return /[\p{L}\p{N}]/u.test(token);
}

/** UTF-8 transcript bytes of words joined by spaces (the model's text length, prefix excluded). */
export function transcriptBytes(words: string[]): number {
  let n = Math.max(0, words.length - 1);
  for (const w of words) n += utf8.encode(w).length;
  return n;
}

/** Byte ids [1, 32, transcript...] for the Dia encoder and each word's [start, end) byte span (prefix excluded). */
export function transcriptIds(words: string[]): { ids: Int32Array; spans: Uint32Array } {
  const ids: number[] = [1, 32];
  const spans = new Uint32Array(2 * words.length);
  words.forEach((w, i) => {
    if (i) ids.push(32);
    spans[2 * i] = ids.length - 2;
    for (const b of utf8.encode(w)) ids.push(b);
    spans[2 * i + 1] = ids.length - 2;
  });
  return { ids: Int32Array.from(ids), spans };
}

/** FuseAlign._standardize in float32 (sums accumulated in double, then rounded as torch's fp32 reductions are). */
export function standardize(pcm: Float32Array): Float32Array {
  const n = pcm.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += pcm[i];
  const mean = f(f(sum) / n);
  const out = new Float32Array(n);
  let sq = 0;
  for (let i = 0; i < n; i++) {
    const v = f(pcm[i] - mean);
    out[i] = v;
    sq += v * v;
  }
  const denom = f(f(Math.sqrt(f(f(sq) / n))) + f(1e-5));
  for (let i = 0; i < n; i++) out[i] = f(out[i] / denom);
  return out;
}

/** Round to the nearest bf16 (ties to even), as torch's .to(torch.bfloat16). */
export function bf16Round(x: number): number {
  const b = new Float32Array([x]);
  const u = new Uint32Array(b.buffer);
  if ((u[0] & 0x7fffffff) > 0x7f800000) return x;
  u[0] = ((u[0] + 0x7fff + ((u[0] >>> 16) & 1)) & 0xffff0000) >>> 0;
  return b[0];
}

/** FuseAlign RotaryEmbedding inv_freq: theta ** (-(2i) / hd) in fp32. */
export function faInvFreq(hd: number, theta: number): Float32Array {
  return Float32Array.from({ length: hd / 2 }, (_, i) => f(Math.pow(theta, f(-(2 * i) / hd))));
}

/** transformers' default RoPE inv_freq (Dia): 1 / base ** ((2i) / hd) in fp32. */
export function diaInvFreq(hd: number, theta: number): Float32Array {
  return Float32Array.from({ length: hd / 2 }, (_, i) => f(1 / f(Math.pow(theta, f((2 * i) / hd)))));
}

/** cos / sin tables [positions][hd / 2] of fp32 angles position * inv_freq, optionally rounded to bf16 (Dia). */
export function ropeTables(positions: Float32Array, inv: Float32Array, round: boolean): { cos: Float32Array; sin: Float32Array } {
  const half = inv.length;
  const cos = new Float32Array(positions.length * half), sin = new Float32Array(positions.length * half);
  for (let p = 0; p < positions.length; p++) {
    for (let i = 0; i < half; i++) {
      const a = f(positions[p] * inv[i]);
      const c = Math.cos(a), s = Math.sin(a);
      cos[p * half + i] = round ? bf16Round(c) : c;
      sin[p * half + i] = round ? bf16Round(s) : s;
    }
  }
  return { cos, sin };
}

/** Joint-sequence RoPE positions: audio patches 0..A-1, then text byte j at j * A / max(T - 1, 1) (fp32). */
export function jointPositions(A: number, T: number): Float32Array {
  const pos = new Float32Array(A + T);
  const scale = f(A / Math.max(T - 1, 1));
  for (let a = 0; a < A; a++) pos[a] = a;
  for (let j = 0; j < T; j++) pos[A + j] = f(j * scale);
  return pos;
}

export type Interval = [number, number];

/**
 * fusealign_hub.longform.fill_unplaced: each maximal null run takes equal parts of its surrounding gap (a leading gap
 * begins at 0, a trailing one ends at `duration`); placed entries are kept verbatim.
 */
export function fillUnplaced(intervals: (Interval | null)[], duration: number): Interval[] {
  const filled = intervals.slice();
  let index = 0;
  while (index < filled.length) {
    if (filled[index] !== null) { index++; continue; }
    const lo = index;
    while (index < filled.length && filled[index] === null) index++;
    const left = lo ? filled[lo - 1]![1] : 0;
    const right = index < filled.length ? filled[index]![0] : duration;
    const width = (right - left) / (index - lo);
    for (let o = 0; o < index - lo; o++)
      filled[lo + o] = [left + o * width, lo + o + 1 === index ? right : left + (o + 1) * width];
  }
  return filled as Interval[];
}

export interface AlignedWord {
  word: string;
  start: number;      // seconds
  end: number;        // seconds
  presence: number;   // the model's probability that the word is in the audio (0 for punctuation-only tokens)
  placed: boolean;    // false: uniform fallback (presence under the threshold, audio under one patch) or punctuation
}

/**
 * FuseAlignAligner.align_batch's per-token output: wordy tokens take their (filled) intervals, punctuation-only tokens
 * are zero-width at the previous token's end (the first wordy start when they lead).
 */
export function assembleWords(tokens: string[], wordy: number[], intervals: (Interval | null)[], presence: number[],
                              duration: number): AlignedWord[] {
  const filled = fillUnplaced(intervals, duration);
  const firstStart = filled.length ? filled[0][0] : 0;
  const out: AlignedWord[] = [];
  let previousEnd: number | null = null;
  let j = 0;
  tokens.forEach((word, k) => {
    if (j < wordy.length && wordy[j] === k) {
      out.push({ word, start: filled[j][0], end: filled[j][1], presence: presence[j], placed: intervals[j] !== null });
      previousEnd = filled[j][1];
      j++;
    } else {
      const t: number = previousEnd ?? firstStart;
      out.push({ word, start: t, end: t, presence: 0, placed: false });
      previousEnd = t;
    }
  });
  return out;
}

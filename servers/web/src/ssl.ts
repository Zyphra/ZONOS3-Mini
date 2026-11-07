// w2v-BERT 2.0 SSL teacher (layer 16) for reference encoding: seamless-m4t fbank frontend on the host
// (ssl_features: F64 FFT/mel, F32 statistics with NumPy's pairwise summation) and the conformer on WebGPU.
// Mirrors ssl_features_pcm + build_graph (ssl-encoder.cpp) with its native BF16 autocast recipe: the features are
// rounded to bf16; every linear reads bf16 weights and bf16-rounded inputs with f32 accumulation and rounds after its
// bias (SiLU and GLU outputs rounded too); attention rounds its content/relative scores, their scaled sum, the
// probabilities and the output (ssl_attention_bf16.wgsl); the depthwise conv output is rounded. LayerNorm, softmax and
// the depthwise LayerNorm + SiLU stay f32, and only layer 0 rounds its residual stream (later layers add in f32).
import { bf16RoundInPlace } from './bf16';
import type { Gpu } from './gpu/device';
import { ACT, dwconv, elementwise, EW, gemm, layernorm, RND } from './gpu/ops';
import { Builder, type Program } from './gpu/program';
import { loadModel, type Model, type Progress } from './gpu/weights';
import { Resampler } from './resample';
import { fft512 } from './fft';

const f = Math.fround;
const FRAME_LEN = 400;
const HOP = 160;
const NFFT = 512;
const NMELS = 80;
const STRIDE = 2;
const PREEMPH = 0.97;
const MEL_FLOOR = 1.1920928955078125e-7;
const D = 1024;
const FF = 4096;
const NLAYER = 16;
const CONV_K = 31;
/** Teacher frames for one 300-frame (882000-sample, 44.1 kHz) encoder lane. */
export const SSL_FRAMES = 999;
export const LANE_SAMPLES = 882000;

/** Port of frontend_pairwise_sum: NumPy's 8-accumulator, 128-block pairwise F32 reduction over a strided column. */
function pairwiseSum(x: Float32Array, off: number, n: number, stride: number, squared: boolean, mean: number): number {
  const value = (i: number) => {
    const v = x[off + i * stride];
    if (!squared) return v;
    const d = f(v - mean);
    return f(d * d);
  };
  if (n < 8) {
    let sum = -0;
    for (let i = 0; i < n; i++) sum = f(sum + value(i));
    return sum;
  }
  if (n <= 128) {
    const s = new Float32Array(8);
    for (let j = 0; j < 8; j++) s[j] = value(j);
    const end = n - n % 8;
    let i = 8;
    for (; i < end; i += 8) for (let j = 0; j < 8; j++) s[j] = f(s[j] + value(i + j));
    let sum = f(f(f(s[0] + s[1]) + f(s[2] + s[3])) + f(f(s[4] + s[5]) + f(s[6] + s[7])));
    for (; i < n; i++) sum = f(sum + value(i));
    return sum;
  }
  let half = Math.floor(n / 2);
  half -= half % 8;
  return f(pairwiseSum(x, off, half, stride, squared, mean) + pairwiseSum(x, off + half * stride, n - half, stride, squared, mean));
}

/** Port of ssl_features on 16 kHz audio: normalized log-mel [T][160] (stride-2 stacked, odd tail padded with 1.0). */
export function sslFeatures(wav: Float32Array, window: Float32Array, melFilters: Float32Array): { feats: Float32Array; T: number } {
  const n = wav.length;
  const nf = n < FRAME_LEN ? 0 : 1 + Math.floor((n - FRAME_LEN) / HOP);
  if (nf < 2) throw new Error('ssl: audio too short for normalized fbank');
  const bins = NFFT / 2 + 1;
  const fb = new Float32Array(nf * NMELS);
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
  const power = new Float64Array(bins);
  for (let fr = 0; fr < nf; fr++) {
    const o = fr * HOP;
    let avg = 0;
    for (let i = 0; i < FRAME_LEN; i++) avg += wav[o + i] * 32768;
    avg /= FRAME_LEN;
    for (let i = 0; i < FRAME_LEN; i++) re[i] = wav[o + i] * 32768 - avg;
    for (let i = FRAME_LEN - 1; i > 0; i--) re[i] -= PREEMPH * re[i - 1];
    re[0] *= 1 - PREEMPH;
    for (let i = 0; i < FRAME_LEN; i++) re[i] *= window[i];
    re.fill(0, FRAME_LEN);
    im.fill(0);
    fft512(re, im);
    for (let k = 0; k < bins; k++) {
      const r = f(re[k]), m = f(im[k]);
      power[k] = r * r + m * m;
    }
    for (let mel = 0; mel < NMELS; mel++) {
      let p = 0;
      for (let k = 0; k < bins; k++) p += power[k] * melFilters[k * NMELS + mel];
      fb[fr * NMELS + mel] = Math.log(Math.max(p, MEL_FLOOR));
    }
  }
  const mean = new Float32Array(NMELS), sd = new Float32Array(NMELS);
  for (let j = 0; j < NMELS; j++) {
    mean[j] = f(pairwiseSum(fb, j, nf, NMELS, false, 0) / nf);
    sd[j] = f(Math.sqrt(f(f(pairwiseSum(fb, j, nf, NMELS, true, mean[j]) / (nf - 1)) + f(1e-7))));
  }
  const T = Math.ceil(nf / STRIDE);
  const feats = new Float32Array(T * STRIDE * NMELS).fill(1);
  for (let i = 0; i < nf; i++)
    for (let j = 0; j < NMELS; j++) feats[i * NMELS + j] = f(f(fb[i * NMELS + j] - mean[j]) / sd[j]);
  return { feats, T };
}

export class SslTeacher {
  private prog: Program | null = null;
  private input!: GPUBuffer;
  private output!: GPUBuffer;
  private resampleKernel: Float32Array;

  private constructor(private gpu: Gpu, private w: Model) {
    const k = w.host.get('frontend.resample_44100');
    if (!k || w.meta['n_layer'] !== NLAYER || w.meta['n_embd'] !== D)
      throw new Error('ssl: unsupported teacher; re-export with npm run export');
    this.resampleKernel = k;
  }

  static async load(gpu: Gpu, base: string, onProgress?: Progress, useCache = true): Promise<SslTeacher> {
    return new SslTeacher(gpu, await loadModel(gpu, base, 'teacher', onProgress, useCache));
  }

  /** Features [999][160] of one 44.1 kHz teacher lane (882000 samples), resampled to 16 kHz first. */
  features(pcm44k: Float32Array): Float32Array {
    if (pcm44k.length !== LANE_SAMPLES) throw new Error('ssl: teacher lane must have 882000 samples');
    const rs = new Resampler(this.resampleKernel, 441, 160, 17);
    const a = rs.push(pcm44k), b = rs.flush();
    const wav = new Float32Array(a.length + b.length);
    wav.set(a);
    wav.set(b, a.length);
    const { feats, T } = sslFeatures(wav, this.w.host.get('frontend.window')!, this.w.host.get('frontend.mel_filters')!);
    if (T !== SSL_FRAMES) throw new Error('ssl: unexpected frame count');
    return feats;
  }

  /** Final layer output [999][1024]. */
  async encode(feats: Float32Array): Promise<Float32Array> {
    if (!this.prog) this.record();
    const x = feats.slice();
    bf16RoundInPlace(x);
    this.gpu.write(this.input, 0, x);
    const [out] = await this.prog!.run(this.gpu, [{ buf: this.output, offset: 0, bytes: SSL_FRAMES * D * 4 }]);
    return out;
  }

  private record(): void {
    const T = SSL_FRAMES;
    const w = this.w;
    const W = (n: string) => w.get(n).buf;
    const eps = Number(w.meta['eps'] ?? 1e-5);
    const b = new Builder(this.gpu);
    this.input = b.buf(T * 160 * 4, 'ssl.in');
    const ln = (x: GPUBuffer, p: string, y: GPUBuffer, N = D, act = 0) => layernorm(b, x, W(p + '.weight'), W(p + '.bias'), y, T, N, eps, act);
    const lin = (x: GPUBuffer, p: string, y: GPUBuffer, O: number, IC: number, rnd: number,
                 o: { act?: number; alpha?: number; res?: GPUBuffer; bias?: boolean } = {}) =>
      gemm(b, x, w.get(p + '.weight'), y, { T, O, IC, act: o.act, alpha: o.alpha, rnd },
        { bias: o.bias === false ? undefined : W(p + '.bias'), res: o.res });

    const fpn = b.buf(T * 160 * 4, 'ssl.fpn');
    ln(this.input, 'fp.ln', fpn, 160);
    const xs = [b.buf(T * D * 4, 'ssl.x0'), b.buf(T * D * 4, 'ssl.x1')];
    let cur = 0;
    lin(fpn, 'fp.proj', xs[0], D, 160, RND.bias);
    const xn = b.buf(T * D * 4, 'ssl.xn');
    const hff = b.buf(T * FF * 4, 'ssl.hff');
    const qb = b.buf(T * D * 4, 'ssl.q'), kb = b.buf(T * D * 4, 'ssl.k'), vb = b.buf(T * D * 4, 'ssl.v');
    const att = b.buf(T * D * 4, 'ssl.att');
    const pw1 = b.buf(T * 2 * D * 4, 'ssl.pw1');
    const xf = b.buf((T + CONV_K - 1) * D * 4, 'ssl.xf');
    const dw = b.buf(T * D * 4, 'ssl.dw');
    // each residual add writes a fresh buffer (res != Y); alternate between two stream buffers
    for (let l = 0; l < NLAYER; l++) {
      const p = `enc.${l}.`;
      // Only layer 0's residual additions return bf16 (final_ln is autocast f32, so later adds promote to f32).
      const out = l === 0 ? RND.out : 0;
      const step = (fn: (x: GPUBuffer, y: GPUBuffer) => void) => { fn(xs[cur], xs[cur ^ 1]); cur ^= 1; };
      step((x, y) => {
        ln(x, p + 'ffn1_ln', xn);
        lin(xn, p + 'ffn1.up', hff, FF, D, RND.bias | RND.act, { act: ACT.silu });
        lin(hff, p + 'ffn1.down', y, D, FF, RND.bias | RND.alpha | out, { alpha: 0.5, res: x });
      });
      step((x, y) => {
        ln(x, p + 'attn_ln', xn);
        lin(xn, p + 'attn.q', qb, D, D, RND.bias);
        lin(xn, p + 'attn.k', kb, D, D, RND.bias);
        lin(xn, p + 'attn.v', vb, D, D, RND.bias);
        b.dispatch('ssl_attention_bf16', { qb, kb, vb, dist: W(p + 'attn.dist'), outp: att }, { T, scale: 0.125 }, [T * 16, 1, 1]);
        lin(att, p + 'attn.o', y, D, D, RND.bias | out, { res: x });
      });
      step((x, y) => {
        ln(x, p + 'conv_ln', xn);
        lin(xn, p + 'conv.pw1', pw1, 2 * D, D, RND.acc, { bias: false });
        b.clear(xf, 0, (CONV_K - 1) * D * 4);
        elementwise(b, EW.gluSplitBf16, T * D, pw1, xf, { C: D, y_off: (CONV_K - 1) * D });
        dwconv(b, xf, W(p + 'conv.dw.weight'), dw, T, D, CONV_K, undefined, 0, 4);
        ln(dw, p + 'conv.dw_ln', xn, D, ACT.silu);
        lin(xn, p + 'conv.pw2', y, D, D, RND.acc | out, { bias: false, res: x });
      });
      step((x, y) => {
        ln(x, p + 'ffn2_ln', xn);
        lin(xn, p + 'ffn2.up', hff, FF, D, RND.bias | RND.act, { act: ACT.silu });
        lin(hff, p + 'ffn2.down', y, D, FF, RND.bias | RND.alpha | out, { alpha: 0.5, res: x });
      });
      step((x, y) => ln(x, p + 'final_ln', y));
      b.split();
    }
    this.output = xs[cur];
    this.prog = b.finish();
  }
}

// MECT-B2 + ASTP speaker encoder (spk-encoder.gguf, 192-d) on WebGPU: host Kaldi fbank in f64 and the network as one
// recorded program per frame count. Mirrors fbank_host + builder + spk_embed (servers/ggml/src/spk-encoder.cpp) at f32.
// Activations are channel-last [T][F][C], so ggml's flatten (k = f*C + c) is the row layout itself.
import type { Gpu } from './gpu/device';
import { fft512 } from './fft';
import {
  ACT, astpPool, attention, denseFuse, dwconv, gconv3x3, gemmF32, layernorm, moeDenseMix, subsampleRows, upsampleAdd,
} from './gpu/ops';
import { Builder, type Program } from './gpu/program';
import { loadModel, type Model, type Progress } from './gpu/weights';
import { Resampler } from './resample';

const HIDDEN = 64;
const FEAT = 2560;
const IN_C = 32;
const N_MELS = 80;
const BINS = 257;
const EMB = 192;
const FRAME_LEN = 400;
const FRAME_SHIFT = 160;
const N_FFT = 512;
const N_HEADS = 4;
const POS_K = 113;
const MAX_SAMPLES_16K = 60 * 16000;
/** Largest activation row: stage-1 res blocks, [80][128] floats per frame. */
const MAX_ROW_FLOATS = 10240;
const PROGRAM_CACHE = 4;
const ERROR = 'speaker embedding computation failed';
const LAYOUT_ERROR = 'spk: tensor layout does not match MECT_B2';

// MECT_B2 (backbones/mect.py:759-780): per block (cnn depth, out channels, stride F, stride T, expansion).
interface BlockCfg { cnn: number; outCh: number; sF: number; sT: number; expansion: number }
const rep = (n: number, c: BlockCfg): BlockCfg[] => Array.from({ length: n }, () => c);
const MECT_B2: BlockCfg[][] = [
  rep(2, { cnn: 3, outCh: 32, sF: 1, sT: 1, expansion: 4 }),
  [{ cnn: 3, outCh: 64, sF: 2, sT: 1, expansion: 2 }, ...rep(3, { cnn: 3, outCh: 64, sF: 1, sT: 2, expansion: 2 })],
  [{ cnn: 3, outCh: 128, sF: 2, sT: 1, expansion: 2 }, ...rep(4, { cnn: 3, outCh: 128, sF: 1, sT: 2, expansion: 2 })],
  [{ cnn: 3, outCh: 256, sF: 2, sT: 1, expansion: 1 }, ...rep(2, { cnn: 2, outCh: 256, sF: 1, sT: 2, expansion: 1 })],
];

function checkMeta(meta: Record<string, unknown>): void {
  const want: Record<string, unknown> = {
    'general.architecture': 'mect_spk', backbone: 'MECT_B2', sample_rate: 16000, max_seconds: 60, embedding_length: EMB,
    n_mels: N_MELS, frame_length: FRAME_LEN, frame_shift: FRAME_SHIFT, n_fft: N_FFT, n_heads: N_HEADS, n_experts: 4,
    pos_kernel: POS_K,
  };
  for (const [k, v] of Object.entries(want))
    if (meta[k] !== v) throw new Error('spk: unsupported or incomplete MECT-B2 speaker encoder metadata');
}

interface Recorded { prog: Program; input: GPUBuffer; pooled: GPUBuffer; feats: GPUBuffer[] }
export interface SpeakerRun { embedding: Float32Array; pooled: Float32Array; feats: Float32Array[] }

export class SpeakerEncoder {
  /** Longest clip (seconds) whose stage-1 activations fit the device's storage-buffer limits; longer clips are cropped. */
  readonly maxSeconds: number;
  private readonly maxBytes: number;
  private readonly progs = new Map<string, Recorded>();
  private queue: Promise<unknown> = Promise.resolve();
  private readonly resampleKernel: Float32Array;
  private readonly window: Float32Array;
  private readonly melBanks: Float32Array;
  private readonly preemph: number;
  private readonly waveScale: number;
  private readonly logFloor: number;
  private readonly lnEps: number;
  private readonly varFloor: number;

  private constructor(private gpu: Gpu, private w: Model,
                      private head: { bnScale: Float32Array; bnShift: Float32Array; linear: Float32Array }) {
    const host = (n: string) => {
      const t = w.host.get(n);
      if (!t) throw new Error(`spk: missing ${n}`);
      return t;
    };
    this.resampleKernel = host('frontend.resample_44100');
    this.window = host('frontend.window');
    this.melBanks = host('frontend.mel_banks');
    const m = w.meta;
    this.preemph = Number(m['preemphasis']);
    this.waveScale = Number(m['waveform_scale']);
    this.logFloor = Number(m['log_floor']);
    this.lnEps = Number(m['ln_eps']);
    this.varFloor = Number(m['pool_var_floor']);
    const lim = gpu.device.limits;
    this.maxBytes = Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize);
    const maxT = Math.floor(this.maxBytes / (MAX_ROW_FLOATS * 4));
    this.maxSeconds = Math.min(60, Math.floor(maxT / 100));
    if (this.maxSeconds < 1) throw new Error(`${ERROR} (maximum 0 seconds)`);
  }

  static async load(gpu: Gpu, base: string, onProgress?: Progress, cache = true): Promise<SpeakerEncoder> {
    const w = await loadModel(gpu, base, 'spk', onProgress, cache);
    checkMeta(w.meta);
    const bnScale = await gpu.read(w.get('bn.scale').buf, 0, 2 * FEAT * 4);
    const bnShift = await gpu.read(w.get('bn.shift').buf, 0, 2 * FEAT * 4);
    const linear = await gpu.read(w.get('linear.weight').buf, 0, EMB * 2 * FEAT * 4);
    return new SpeakerEncoder(gpu, w, { bnScale, bnShift, linear });
  }

  /** 192-d speaker embedding (not normalized) of mono 44.1 kHz audio; at most maxSeconds are used. */
  async embed(pcm44k: Float32Array): Promise<Float32Array> {
    const { fbank, T } = this.features(pcm44k);
    const { embedding } = await this.run(fbank, T);
    if (!embedding.every(Number.isFinite)) throw new Error(ERROR);
    return embedding;
  }

  /** Kaldi fbank [T][80] (log mel, before CMN) of the first maxSeconds of 44.1 kHz audio resampled to 16 kHz. */
  features(pcm44k: Float32Array): { fbank: Float32Array; T: number } {
    const rs = new Resampler(this.resampleKernel, 441, 160, 17);
    const a = rs.push(pcm44k), z = rs.flush();
    const n = Math.min(a.length + z.length, MAX_SAMPLES_16K, this.maxSeconds * 16000);
    if (n < FRAME_LEN) throw new Error('spk: audio shorter than 25 ms');
    const wav = new Float32Array(n);
    wav.set(a.subarray(0, n));
    if (n > a.length) wav.set(z.subarray(0, n - a.length), a.length);
    const T = 1 + Math.floor((n - FRAME_LEN) / FRAME_SHIFT);
    const fbank = new Float32Array(T * N_MELS);
    const x = new Float64Array(FRAME_LEN), re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
    const power = new Float64Array(BINS);
    const { preemph, waveScale, window: win, melBanks, logFloor } = this;
    for (let t = 0; t < T; t++) {
      const off = t * FRAME_SHIFT;
      let mean = 0;
      for (let j = 0; j < FRAME_LEN; j++) { x[j] = wav[off + j] * waveScale; mean += x[j]; }
      mean /= FRAME_LEN;
      for (let j = 0; j < FRAME_LEN; j++) x[j] -= mean;
      re.fill(0);
      im.fill(0);
      for (let j = FRAME_LEN - 1; j >= 1; j--) re[j] = (x[j] - preemph * x[j - 1]) * win[j];
      re[0] = (x[0] - preemph * x[0]) * win[0];
      fft512(re, im);
      for (let k = 0; k < BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];
      for (let b = 0; b < N_MELS; b++) {
        const row = b * BINS;
        let acc = 0;
        for (let k = 0; k < BINS; k++) acc += melBanks[row + k] * power[k];
        fbank[t * N_MELS + b] = Math.log(Math.max(acc, logFloor));
      }
    }
    return { fbank, T };
  }

  /** Per-utterance CMN of `fbank` [T][80], the network, then BN + linear on the host in f64. `capture` also reads
   *  back the four stage outputs (feat_1..4, [T][2560]). Runs serially. */
  run(fbank: Float32Array, T: number, capture = false): Promise<SpeakerRun> {
    const job = this.queue.then(() => this.runNow(fbank, T, capture));
    this.queue = job.catch(() => {});
    return job;
  }

  private async runNow(fbank: Float32Array, T: number, capture: boolean): Promise<SpeakerRun> {
    if (fbank.length !== T * N_MELS) throw new Error('spk: fbank must be [T][80]');
    if (T * MAX_ROW_FLOATS * 4 > this.maxBytes) throw new Error(`${ERROR} (maximum ${this.maxSeconds} seconds)`);
    const cmn = new Float32Array(T * N_MELS);
    for (let b = 0; b < N_MELS; b++) {
      let mean = 0;
      for (let t = 0; t < T; t++) mean += fbank[t * N_MELS + b];
      mean /= T;
      for (let t = 0; t < T; t++) cmn[t * N_MELS + b] = fbank[t * N_MELS + b] - mean;
    }
    const rec = this.program(T, capture);
    this.gpu.write(rec.input, 0, cmn);
    const [pooled, ...feats] = await rec.prog.run(this.gpu, [
      { buf: rec.pooled, offset: 0, bytes: 2 * FEAT * 4 },
      ...rec.feats.map(buf => ({ buf, offset: 0, bytes: T * FEAT * 4 })),
    ]);
    const { bnScale, bnShift, linear } = this.head;
    const pbn = new Float64Array(2 * FEAT);
    for (let i = 0; i < 2 * FEAT; i++) pbn[i] = pooled[i] * bnScale[i] + bnShift[i];
    const embedding = new Float32Array(EMB);
    for (let o = 0; o < EMB; o++) {
      const row = o * 2 * FEAT;
      let acc = 0;
      for (let i = 0; i < 2 * FEAT; i++) acc += linear[row + i] * pbn[i];
      embedding[o] = acc;
    }
    return { embedding, pooled, feats };
  }

  /** Recorded program for T frames; the last PROGRAM_CACHE are kept. */
  private program(T: number, capture: boolean): Recorded {
    const key = `${T}:${capture ? 1 : 0}`;
    let rec = this.progs.get(key);
    if (rec) {
      this.progs.delete(key);
    } else {
      rec = this.record(T, capture);
      if (this.progs.size >= PROGRAM_CACHE) {
        const [oldKey, old] = this.progs.entries().next().value!;
        old.prog.destroy();
        this.progs.delete(oldKey);
      }
    }
    this.progs.set(key, rec);
    return rec;
  }

  private record(T: number, capture: boolean): Recorded {
    const b = new Builder(this.gpu);
    const B = (n: string) => this.w.get(n).buf;
    const input = b.buf(T * N_MELS * 4, 'spk.in');
    const streams = [b.buf(T * FEAT * 4, 'spk.s0'), b.buf(T * FEAT * 4, 'spk.s1')];
    const fused = b.buf(T * FEAT * 4, 'spk.fused');
    const pooled = b.buf(2 * FEAT * 4, 'spk.pooled');
    const feats = capture ? MECT_B2.map((_, s) => b.buf(T * FEAT * 4, `spk.feat${s + 1}`)) : [];

    gconv3x3(b, input, B('fe.head.conv.weight'), streams[0], { T_in: T, F_in: N_MELS, IC: 1, OC: IN_C, a: 1, s_f: 1, s_t: 1 },
      { bias: B('fe.head.conv.bias'), scale: B('fe.head.bn.scale'), shift: B('fe.head.bn.shift') });
    let cur = 0, C = IN_C, F = N_MELS;
    MECT_B2.forEach((stage, s) => {
      stage.forEach((cfg, k) => {
        [C, F] = this.block(b, streams[cur], streams[cur ^ 1], T, C, F, cfg, `fe.b${s + 1}.${k}.`);
        cur ^= 1;
        b.split();
      });
      denseFuse(b, streams[cur], B('fe.dense_w'), fused, T, FEAT, s);
      if (capture) b.copy(streams[cur], 0, feats[s], 0, T * FEAT * 4);
    });

    // ASTP over time; the spare stream buffer holds the attention logits.
    const p1 = b.tmp(T * 128 * 4, 'spk.pool1');
    const att = streams[cur ^ 1];
    gemmF32(b, fused, this.w.get('pool.l1.weight'), p1, { T, O: 128, IC: FEAT, act: ACT.tanh }, { bias: B('pool.l1.bias') });
    gemmF32(b, p1, this.w.get('pool.l2.weight'), att, { T, O: FEAT, IC: 128 }, { bias: B('pool.l2.bias') });
    astpPool(b, att, fused, pooled, T, FEAT, this.varFloor);
    return { prog: b.finish(), input, pooled, feats };
  }

  /** One MECT block on stream X ([T][F][C], F*C = 2560) into stream Y; returns the output [C, F]. */
  private block(b: Builder, X: GPUBuffer, Y: GPUBuffer, T: number, cIn: number, fIn: number, cfg: BlockCfg,
                p: string): [number, number] {
    const w = this.w;
    const B = (n: string) => w.get(n).buf;
    let y = X, Tc = T, C = cIn, F = fIn;
    for (let r = 0; r < cfg.cnn; r++) {
      const out = this.resBlock(b, y, Tc, F, C, `${p}cnn.${r}.`, r === 0 ? cfg.sF : 1, r === 0 ? cfg.sT : 1);
      if (y !== X) b.release(y);
      y = out.buf; Tc = out.T; F = out.F; C = out.C;
    }
    if (w.has(p + 'reduce.weight')) {
      const red = w.get(p + 'reduce.weight'), cOut = red.shape[3];
      const R = b.tmp(Tc * F * cOut * 4, 'spk.reduce');
      gemmF32(b, y, red, R, { T: Tc * F, O: cOut, IC: C },
        { bias: B(p + 'reduce.bias'), scale: B(p + 'reduce.bn.scale'), shift: B(p + 'reduce.bn.shift') });
      b.release(y);
      y = R; C = cOut;
    }
    if (C * F !== FEAT || cIn * fIn !== FEAT) throw new Error(LAYOUT_ERROR);
    const x2 = y, T2 = Tc;

    const hs = T2 * HIDDEN * 4, pad = POS_K >> 1;
    const hA = b.tmp(hs, 'spk.hA'), hB = b.tmp(hs, 'spk.hB'), hC = b.tmp(hs, 'spk.hC');
    const q = b.tmp(hs, 'spk.q'), k = b.tmp(hs, 'spk.k'), v = b.tmp(hs, 'spk.v'), at = b.tmp(hs, 'spk.att');
    const hp = b.tmp((T2 + 2 * pad) * HIDDEN * 4, 'spk.hp');
    const lg = b.tmp(T2 * 4 * 4, 'spk.gate'), ex = b.tmp(T2 * 4 * HIDDEN * 4, 'spk.experts'), mix = b.tmp(hs, 'spk.mix');
    const lin = (x: GPUBuffer, n: string, out: GPUBuffer, O: number, IC: number,
                 o: { act?: number; alpha?: number; res?: GPUBuffer; bias?: boolean } = {}) =>
      gemmF32(b, x, w.get(n + '.weight'), out, { T: T2, O, IC, act: o.act, alpha: o.alpha },
        { bias: o.bias === false ? undefined : B(n + '.bias'), res: o.res });
    const ln = (x: GPUBuffer, n: string, out: GPUBuffer) =>
      layernorm(b, x, B(n + '.weight'), B(n + '.bias'), out, T2, HIDDEN, this.lnEps);

    lin(x2, p + 'rproj', hA, HIDDEN, FEAT);
    ln(hA, p + 'rproj.ln', hB);
    // h + pos_conv(h): zero-padded depthwise K=113 over time (BN folded on export), exact GELU, pointwise.
    b.clear(hp, 0, pad * HIDDEN * 4);
    b.copy(hB, 0, hp, pad * HIDDEN * 4, hs);
    b.clear(hp, (pad + T2) * HIDDEN * 4, pad * HIDDEN * 4);
    dwconv(b, hp, B(p + 'pos.dw.weight'), hA, T2, HIDDEN, POS_K, B(p + 'pos.dw.bias'), ACT.gelu);
    lin(hA, p + 'pos.pw', hC, HIDDEN, HIDDEN, { res: hB });
    // Full (non-causal) attention: pos0 = T2-1 with window = ring = T2 visits every key once per row.
    const hd = HIDDEN / N_HEADS;
    lin(hC, p + 'attn.q', q, HIDDEN, HIDDEN, { alpha: 1 / Math.sqrt(hd) });
    lin(hC, p + 'attn.k', k, HIDDEN, HIDDEN);
    lin(hC, p + 'attn.v', v, HIDDEN, HIDDEN);
    attention(b, q, k, v, at, { n: T2, nh: N_HEADS, nkv: N_HEADS, hd, window: T2, ring: T2, scale: 1, pos0: T2 - 1 });
    lin(at, p + 'attn.o', hA, HIDDEN, HIDDEN, { res: hC });
    ln(hA, p + 'ln1', hB);
    // Dense 4-expert MoE FFN.
    gemmF32(b, hB, w.get(p + 'moe.gate.weight'), lg, { T: T2, O: 4, IC: HIDDEN });
    gemmF32(b, hB, w.get(p + 'moe.experts.weight'), ex, { T: T2, O: 4 * HIDDEN, IC: HIDDEN, act: ACT.newgelu });
    moeDenseMix(b, lg, ex, mix, T2);
    lin(mix, p + 'ffn_out', hA, HIDDEN, HIDDEN, { res: hB });
    ln(hA, p + 'ln2', hC);
    const E = b.tmp(T2 * FEAT * 4, 'spk.e');
    lin(hC, p + 'eproj', E, FEAT, HIDDEN, { res: x2 });
    upsampleAdd(b, E, X, Y, T, FEAT, cfg.sT);
    for (const t of [x2, hA, hB, hC, q, k, v, at, hp, lg, ex, mix, E]) b.release(t);
    return [C, F];
  }

  /** torch BasicResBlock: relu(BN(pw1(gconv1(x)))) -> relu(BN(pw2(gconv2(.))) + shortcut). */
  private resBlock(b: Builder, x: GPUBuffer, T: number, F: number, IC: number, q: string, sF: number, sT: number):
    { buf: GPUBuffer; T: number; F: number; C: number } {
    const w = this.w;
    const B = (n: string) => w.get(n).buf;
    const c1 = w.get(q + 'conv1.weight'), c2 = w.get(q + 'conv2.weight');
    const oc = c1.shape[3];
    const T2 = Math.floor((T - 1) / sT) + 1, F2 = Math.floor((F - 1) / sF) + 1;
    const n = T2 * F2 * oc * 4;
    const A = b.tmp(n, 'spk.rA'), Bf = b.tmp(n, 'spk.rB');
    gconv3x3(b, x, c1.buf, A, { T_in: T, F_in: F, IC, OC: oc, a: c1.shape[2], s_f: sF, s_t: sT });
    gemmF32(b, A, w.get(q + 'pw1.weight'), Bf, { T: T2 * F2, O: oc, IC: oc, act: ACT.relu, actAfterShift: true },
      { bias: B(q + 'pw1.bias'), scale: B(q + 'bn1.scale'), shift: B(q + 'bn1.shift') });
    gconv3x3(b, Bf, c2.buf, A, { T_in: T2, F_in: F2, IC: oc, OC: oc, a: c2.shape[2], s_f: 1, s_t: 1 });
    let res = x, sc: GPUBuffer | null = null;
    if (w.has(q + 'sc.weight')) {
      let src = x;
      if (sT > 1) {
        src = b.tmp(T2 * F * IC * 4, 'spk.sub');
        subsampleRows(b, x, src, T2, F * IC, sT);
      }
      sc = b.tmp(n, 'spk.sc');
      gemmF32(b, src, w.get(q + 'sc.weight'), sc, { T: T2 * F2, O: oc, IC, x_stride: sF * IC },
        { scale: B(q + 'sc.bn.scale'), shift: B(q + 'sc.bn.shift') });
      if (src !== x) b.release(src);
      res = sc;
    } else if (sF > 1 || sT > 1 || IC !== oc) {
      throw new Error(LAYOUT_ERROR);
    }
    gemmF32(b, A, w.get(q + 'pw2.weight'), Bf, { T: T2 * F2, O: oc, IC: oc, act: ACT.relu, actAfterShift: true },
      { bias: B(q + 'pw2.bias'), scale: B(q + 'bn2.scale'), shift: B(q + 'bn2.shift'), res });
    b.release(A);
    if (sc) b.release(sc);
    return { buf: Bf, T: T2, F: F2, C: oc };
  }
}

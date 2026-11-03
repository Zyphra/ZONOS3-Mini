// mossdual_istft (15 Hz) streaming codec decoder on WebGPU over a fixed 16-frame graph: RVQ lookup, causal
// ConvNeXt semantic head, 5 MOSS transformer stages with patch-up, the causal ConvNeXt iSTFT head projecting two
// spectrogram resolutions at 240 rows/s, then the host iSTFT + FIR mixer (44160 Hz) and sinc resampler (44100 Hz).
// Mirrors decode_chunk + dac_stream_decode/flush (dac.cpp).
import type { Gpu } from './gpu/device';
import { ACT, attentionWin, dwconv, gemm, layernorm, patchUp, ropeKv, type GemmOpts, type Weight } from './gpu/ops';
import { Builder, type Program, type Step } from './gpu/program';
import { ropeTable } from './gpu/rope';
import type { Model } from './gpu/weights';
import { IstftHead } from './istft';
import { Resampler } from './resample';

export const SAMPLE_RATE = 44100;
export const HOP = 2940;
export const BLOCK_FRAMES = 16;
/** Stream frames per utterance: 30 silence-prefill + 900 decode-context + 2000 generated frames fit. */
export const MAX_FRAMES = 3072;

interface Stage { inDim: number; outDim: number; d: number; nl: number; ff: number; ctx: number; patch: number }
const STAGES: Stage[] = [
  { inDim: 1024, outDim: 1280, d: 1280, nl: 32, ff: 5120, ctx: 150, patch: 2 },
  { inDim: 640, outDim: 768, d: 768, nl: 12, ff: 3072, ctx: 300, patch: 2 },
  { inDim: 384, outDim: 768, d: 768, nl: 12, ff: 3072, ctx: 480, patch: 2 },
  { inDim: 384, outDim: 768, d: 768, nl: 12, ff: 3072, ctx: 480, patch: 2 },
  { inDim: 384, outDim: 0, d: 768, nl: 12, ff: 3072, ctx: 480, patch: 1 },
];
const HEAD_DIM = 64;
const SEM = 768;
const LATENT = 1024;
const SEM_DEPTH = 4;
const SEM_FF = 2048;
const HEAD_DEPTH = 8;
const HEAD_FF = 2048;
const N_FFT: [number, number] = [736, 2208];
const HEAD_HOP = 184;
const FIR = 127;
const CPF = STAGES.reduce((a, s) => a * s.patch, 1);   // 16 head rows per frame
const LN_EPS = 1e-5;
const CNX_EPS = 1e-6;

interface HistOp {
  hist: GPUBuffer;
  step: Extract<Step, { k: 'c' }>;   // export copy; srcOff depends on the valid frame count
  rowBytes: number;
  cpf: number;
}

/** History-prefixed activation: `xf` rows [0,R) are the previous chunk's tail, the producer writes at float `off`. */
interface HistBuf { xf: GPUBuffer; hist: GPUBuffer; off: number }

function checkMeta(meta: Record<string, unknown>): void {
  const want: Record<string, unknown> = {
    'general.architecture': 'mossdual_istft',
    'stage.d_model': STAGES.map(s => s.d),
    'stage.n_layer': STAGES.map(s => s.nl),
    'stage.context': STAGES.map(s => s.ctx),
    'stage.patch': STAGES.map(s => s.patch),
    'stage.out_dim': STAGES.map(s => s.outDim),
    'head.depth': HEAD_DEPTH,
    'head.hidden': HEAD_FF,
    'head.n_fft': N_FFT,
    'head.hop': HEAD_HOP,
    'head.fir_kernel': FIR,
  };
  for (const [k, v] of Object.entries(want))
    if (JSON.stringify(meta[k]) !== JSON.stringify(v)) throw new Error('dac: unsupported codec; re-export with pnpm export');
  if (typeof meta['head.max_log_magnitude'] !== 'number' || typeof meta['headroom_db'] !== 'number')
    throw new Error('dac: unsupported codec; re-export with pnpm export');
}

export class DacStream {
  /** Output gain 10^(-headroom_db/20); applied by the caller (finish_block). */
  readonly gain: number;
  private prog: Program;
  private idx: GPUBuffer;
  private dyn: GPUBuffer;
  private dynData = new Uint32Array(4);
  private spec: [GPUBuffer, GPUBuffer];
  private rope: GPUBuffer;
  private hists: HistOp[] = [];
  private needReset = true;
  private istft: IstftHead;
  private rs: Resampler;

  constructor(private gpu: Gpu, private w: Model) {
    checkMeta(w.meta);
    const host = (n: string): Float32Array => {
      const v = w.host.get(n);
      if (!v) throw new Error(`dac: missing ${n}; re-export with pnpm export`);
      return v;
    };
    this.gain = Math.fround(Math.pow(10, -Number(w.meta['headroom_db']) / 20));
    this.rs = new Resampler(host('resample.kernel'));
    this.istft = new IstftHead({
      nFft: N_FFT, hop: HEAD_HOP, windows: [host('head.window.0'), host('head.window.1')],
      mixer: host('head.mixer.weight'), maxLogMag: Number(w.meta['head.max_log_magnitude']),
    });
    this.idx = gpu.buffer(10 * BLOCK_FRAMES * 4, 'dac.idx');
    this.dyn = gpu.uniform(16, 'dac.dyn');
    this.rope = gpu.upload(ropeTable((MAX_FRAMES + BLOCK_FRAMES) * CPF, 10000, HEAD_DIM), 'dac.rope');
    const b = new Builder(gpu);
    this.spec = this.record(b);
    this.prog = b.finish();
  }

  private record(b: Builder): [GPUBuffer, GPUBuffer] {
    const T = (n: string) => this.w.get(n).buf;
    const M = (n: string): Weight => this.w.get(n);   // matrix operand (f16 or q8)
    const dyn = this.dyn;
    const F = BLOCK_FRAMES;
    let cpf = 1;
    // Codec gemms skip row tiles past the dyn.b valid frames (cpf rows each, read at call time).
    const g = (X: GPUBuffer, W: Weight, Y: GPUBuffer, o: GemmOpts, ex: { bias?: GPUBuffer; scale?: GPUBuffer; res?: GPUBuffer } = {}) =>
      gemm(b, X, W, Y, { ...o, rowsDyn: cpf }, { ...ex, dyn });

    // History-prefixed input buffer: rows [0,R) come from the previous chunk, [R, R+rows) are filled by
    // the producer at float offset R*ch. After the consumer, rows [Tv, Tv+R) export to the history.
    const withHistory = (ch: number, R: number): HistBuf => {
      const rows = F * cpf;
      const xf = b.tmp((R + rows) * ch * 4, 'dac.xf');
      const hist = b.buf(R * ch * 4, 'dac.hist');
      b.copy(hist, 0, xf, 0, R * ch * 4);
      return { xf, hist, off: R * ch };
    };
    const exportHistory = (xf: GPUBuffer, hist: GPUBuffer, ch: number, R: number) => {
      b.copy(xf, 0, hist, 0, R * ch * 4);
      this.hists.push({ hist, step: b.steps[b.steps.length - 1] as HistOp['step'], rowBytes: ch * 4, cpf });
    };

    // ---- semantic lookup + causal ConvNeXt head (768 ch, 15 Hz)
    let hx = withHistory(SEM, 6);
    b.dispatch('dac_embed', { idx: this.idx, tables: T('sem.table'), bias: T('sem.bias'), outp: hx.xf },
      { n: F, C: SEM, F, y_off: hx.off, mode: 0 }, [Math.ceil(SEM / 512), F, 1]);
    const s = b.buf(F * SEM * 4, 'dac.sem');
    for (let bi = 0; bi < SEM_DEPTH; bi++) {
      const p = `sem_head.${bi}.`;
      const h = b.tmp(F * SEM * 4, 'dac.cn');
      dwconv(b, hx.xf, T(p + 'dw.weight'), h, F, SEM, 7, T(p + 'dw.bias'));
      exportHistory(hx.xf, hx.hist, SEM, 6);
      const hn = b.tmp(F * SEM * 4, 'dac.cn');
      layernorm(b, h, T(p + 'norm.weight'), T(p + 'norm.bias'), hn, F, SEM, CNX_EPS);
      b.release(h);
      const h1 = b.tmp(F * SEM_FF * 4, 'dac.pw1');
      g(hn, M(p + 'pw1.weight'), h1, { T: F, O: SEM_FF, IC: SEM, act: ACT.gelu }, { bias: T(p + 'pw1.bias') });
      b.release(hn);
      const next = bi + 1 < SEM_DEPTH ? withHistory(SEM, 6) : null;
      g(h1, M(p + 'pw2.weight'), next ? next.xf : s, { T: F, O: SEM, IC: SEM_FF, r_off: hx.off, y_off: next ? next.off : 0 },
        { bias: T(p + 'pw2.bias'), res: hx.xf });
      b.release(h1);
      b.release(hx.xf);
      if (next) hx = next;
    }
    const z = b.buf(F * LATENT * 4, 'dac.z');
    g(s, M('sem_head.out.weight'), z, { T: F, O: LATENT, IC: SEM }, { bias: T('sem_head.out.bias') });
    b.dispatch('dac_embed', { idx: this.idx, tables: T('res.tables'), bias: T('res.bias'), outp: z },
      { n: F, C: LATENT, F, y_off: 0, mode: 1 }, [Math.ceil(LATENT / 512), F, 1]);

    // ---- MOSS stages: windowed causal attention on absolute per-stage positions p0*cpf + i
    let x = z;
    let headIn: HistBuf | null = null;   // head block 0 input, written by the last stage
    for (const [si, st] of STAGES.entries()) {
      const rows = F * cpf;
      const { d, ff } = st;
      const nh = d / HEAD_DIM;
      const ring = st.ctx - 1 + rows;
      const xa = b.buf(rows * d * 4, 'dac.xa');
      const xb = b.buf(rows * d * 4, 'dac.xb');
      const xn = b.buf(rows * d * 4, 'dac.xn');
      const qkv = b.buf(rows * 3 * d * 4, 'dac.qkv');
      const q = b.buf(rows * d * 4, 'dac.q');
      const ao = b.buf(rows * d * 4, 'dac.ao');
      const f1 = b.buf(rows * ff * 4, 'dac.f1');
      const sp = `stage.${si}.`;
      // outDim 0 and patch 1: the last layer writes straight into the head's history-prefixed input.
      const last = st.outDim === 0 && st.patch === 1;
      g(x, M(sp + 'in.weight'), xa, { T: rows, O: d, IC: st.inDim });
      for (let l = 0; l < st.nl; l++) {
        const p = `${sp}l${l}.`;
        const kr = b.buf(ring * nh * HEAD_DIM * 4, `dac.k${si}.${l}`);
        const vr = b.buf(ring * nh * HEAD_DIM * 4, `dac.v${si}.${l}`);
        layernorm(b, xa, T(p + 'norm1.weight'), T(p + 'norm1.bias'), xn, rows, d, LN_EPS);
        g(xn, M(p + 'qkv.weight'), qkv, { T: rows, O: 3 * d, IC: d });
        ropeKv(b, qkv, this.rope, q, kr, vr,
          { n: rows, nh, nkv: nh, hd: HEAD_DIM, in_stride: 3 * d, qkNorm: false, ring, posDyn: true, nDyn: true, posMul: cpf }, { dyn });
        attentionWin(b, q, kr, vr, ao, { n: rows, nh, window: st.ctx, ring, scale: 1 / Math.sqrt(HEAD_DIM), posMul: cpf }, dyn);
        g(ao, M(p + 'wo.weight'), xb, { T: rows, O: d, IC: d }, { scale: T(p + 'attn_scale'), res: xa });
        layernorm(b, xb, T(p + 'norm2.weight'), T(p + 'norm2.bias'), xn, rows, d, LN_EPS);
        g(xn, M(p + 'ffn1.weight'), f1, { T: rows, O: ff, IC: d, act: ACT.gelu });
        if (last && l === st.nl - 1) {
          headIn = withHistory(d, 6);
          g(f1, M(p + 'ffn2.weight'), headIn.xf, { T: rows, O: d, IC: ff, y_off: headIn.off }, { scale: T(p + 'ffn_scale'), res: xb });
        } else {
          g(f1, M(p + 'ffn2.weight'), xa, { T: rows, O: d, IC: ff }, { scale: T(p + 'ffn_scale'), res: xb });
        }
      }
      if (last) continue;
      const y = b.buf(rows * st.outDim * 4, 'dac.sout');
      g(xa, M(sp + 'out.weight'), y, { T: rows, O: st.outDim, IC: d });
      const xp = b.buf(rows * st.outDim * 4, 'dac.patch');
      patchUp(b, y, xp, rows, st.outDim / st.patch, st.patch);
      x = xp;
      cpf *= st.patch;
    }
    if (!headIn || cpf !== CPF) throw new Error('dac: unsupported codec; re-export with pnpm export');

    // ---- iSTFT head: causal ConvNeXt blocks at F*16 rows (768 ch), LayerNorm, two spectrogram projections
    const rows = F * cpf;
    let cur: HistBuf = headIn;
    const hn = b.buf(rows * SEM * 4, 'dac.hnorm');
    for (let bi = 0; bi < HEAD_DEPTH; bi++) {
      const p = `head.${bi}.`;
      const h = b.tmp(rows * SEM * 4, 'dac.hcn');
      dwconv(b, cur.xf, T(p + 'dw.weight'), h, rows, SEM, 7, T(p + 'dw.bias'));
      exportHistory(cur.xf, cur.hist, SEM, 6);
      const hl = b.tmp(rows * SEM * 4, 'dac.hcn');
      layernorm(b, h, T(p + 'norm.weight'), T(p + 'norm.bias'), hl, rows, SEM, CNX_EPS);
      b.release(h);
      const h1 = b.tmp(rows * HEAD_FF * 4, 'dac.hpw1');
      g(hl, M(p + 'pw1.weight'), h1, { T: rows, O: HEAD_FF, IC: SEM, act: ACT.gelu }, { bias: T(p + 'pw1.bias') });
      b.release(hl);
      const next = bi + 1 < HEAD_DEPTH ? withHistory(SEM, 6) : null;
      const out = next ? next.xf : b.tmp(rows * SEM * 4, 'dac.hout');
      g(h1, M(p + 'pw2.weight'), out, { T: rows, O: SEM, IC: HEAD_FF, r_off: cur.off, y_off: next ? next.off : 0 },
        { bias: T(p + 'pw2.bias'), res: cur.xf });
      b.release(h1);
      b.release(cur.xf);
      if (next) cur = next;
      else {
        layernorm(b, out, T('head.norm.weight'), T('head.norm.bias'), hn, rows, SEM, CNX_EPS);
        b.release(out);
      }
    }
    const spec = N_FFT.map((n, i) => {
      const y = b.buf(rows * (n + 2) * 4, `dac.spec${i}`);
      g(hn, M(`head.out.${i}.weight`), y, { T: rows, O: n + 2, IC: SEM }, { bias: T(`head.out.${i}.bias`) });
      return y;
    });
    return [spec[0], spec[1]];
  }

  /** Starts a new utterance: clears conv histories (on the next chunk), the iSTFT head and the resampler. */
  reset(): void {
    this.needReset = true;
    this.istft.reset();
    this.rs.reset();
  }

  /**
   * Decode frames [p0, p0+n) (1 ≤ n ≤ 16) from un-sheared indices idx[10][16]. Returns the newly final
   * 44.1 kHz samples (variable length: the FIR mixer and resampler hold lookahead until flush()). No gain applied.
   */
  async chunk(idx: Int32Array, p0: number, n: number): Promise<Float32Array> {
    if (n < 1 || n > BLOCK_FRAMES) throw new Error(`dac chunk size ${n} out of range`);
    if (p0 + n > MAX_FRAMES) throw new Error(`dac: utterance exceeds ${MAX_FRAMES} frames`);
    this.gpu.write(this.idx, 0, idx);
    this.dynData[0] = p0;
    this.dynData[1] = n;
    this.gpu.write(this.dyn, 0, this.dynData);
    for (const h of this.hists) h.step.srcOff = n * h.cpf * h.rowBytes;
    const reset = this.needReset;
    this.needReset = false;
    const cols = n * CPF;
    const [s0, s1] = await this.gpu.submitRead(enc => {
      if (reset) for (const h of this.hists) enc.clearBuffer(h.hist, 0, h.hist.size);
      this.prog.encode(enc);
    }, [
      { buf: this.spec[0], offset: 0, bytes: cols * (N_FFT[0] + 2) * 4 },
      { buf: this.spec[1], offset: 0, bytes: cols * (N_FFT[1] + 2) * 4 },
    ]);
    return this.rs.push(this.istft.push(s0, s1, cols));
  }

  /** Ends the utterance: the iSTFT/mixer tail, then the resampler tail; total output is frames * HOP samples. */
  flush(): Float32Array {
    const a = this.rs.push(this.istft.flush());
    const z = this.rs.flush();
    const out = new Float32Array(a.length + z.length);
    out.set(a);
    out.set(z, a.length);
    return out;
  }
}

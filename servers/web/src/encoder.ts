// Reference-audio encoder (mossdual_encoder + w2v-BERT SSL teacher) on WebGPU: 44.1 kHz audio ->
// ordinary [T][10] codec codes. Mirrors zac_encode_diagnostic (zac-encoder.cpp) with its native BF16 recipe:
// 300-frame lanes with 150 frames of left context; per lane a causal acoustic path (SEANet front in
// 10 tiles with carried conv histories, then 5 MOSS stages), the SSL teacher pooled to 300 frames
// through the semantic encoder and RVQ level 0, the semantic head prediction, and 9 residual levels.
// BF16 recipe (zac.cpp encoder_conv / encoder_round / convnext_block / moss_layer): every GEMM reads bf16 weights and
// bf16-rounded inputs with f32 accumulation. Convolutions (and the semantic in/out projections) round the product,
// add the bias and round again; residual units round after their residual add. MOSS linears round their product
// (ffn1 also after GELU), q/k round after RoPE, while attention, LayerNorm, layer scales and the MOSS residual stream
// stay f32. ConvNeXt blocks round the depthwise input, its tap sum and its biased output, pw1/pw2 after bias (pw1 also
// after GELU), and the semantic encoder's residual output. Snake and the RVQ are f32.
import { EDIT_MAX_SECONDS, HOP, SAMPLE_RATE } from './generation';
import type { Gpu } from './gpu/device';
import { ACT, attentionWin, dwconv, gemm, layernorm, patchDown, RND, ropeKv, snake, type Weight } from './gpu/ops';
import { Builder, type Program } from './gpu/program';
import { ropeTable } from './gpu/rope';
import { loadModel, type Model, type Progress } from './gpu/weights';
import { Resampler } from './resample';
import { LANE, poolSsl, Rvq } from './rvq';
import { LANE_SAMPLES, SSL_FRAMES, SslTeacher } from './ssl';

const INTERNAL_HOP = 2944;
const CONTEXT = 150;
const NCB = 10;
const LANE_INTERNAL = LANE * INTERNAL_HOP;   // 883200
const TILES = 10;
const TILE = LANE_INTERNAL / TILES;          // 88320 samples = 480 stage-0 rows
const TILE_ROWS = TILE / 2 / 4 / 23;         // 480
const LATENT = 1024;
const SEM = 768;
const SEM_FF = 2048;
const HEAD_DIM = 64;
const LN_EPS = 1e-5;
const CNX_EPS = 1e-6;
/** Conv1d boundary: bf16(bf16(W.x) + bias). */
const CONV = RND.acc | RND.bias;

interface Stage { inDim: number; outDim: number; d: number; nh: number; nl: number; ff: number; ctx: number; patch: number }
const STAGES: Stage[] = [
  { inDim: 5888, outDim: 384, d: 768, nh: 12, nl: 12, ff: 3072, ctx: 480, patch: 23 },
  { inDim: 768, outDim: 384, d: 768, nh: 12, nl: 12, ff: 3072, ctx: 480, patch: 2 },
  { inDim: 768, outDim: 384, d: 768, nh: 12, nl: 12, ff: 3072, ctx: 480, patch: 2 },
  { inDim: 768, outDim: 640, d: 768, nh: 12, nl: 12, ff: 3072, ctx: 300, patch: 2 },
  { inDim: 1280, outDim: 1024, d: 1280, nh: 20, nl: 32, ff: 5120, ctx: 150, patch: 2 },
];
const STAGE_T = [4800, 2400, 1200, 600, 300];

/** The longest clip the encoder takes: a speech-edit source (reference clips are capped at REFERENCE_AUDIO_MAX_SECONDS
 *  by the speaker routes before they get here). Lanes of 300 frames make length no structural limit. */
const ENCODE_ERROR = `reference audio encoding failed (maximum ${EDIT_MAX_SECONDS} seconds)`;

function checkMeta(meta: Record<string, unknown>): void {
  const want: Record<string, unknown> = {
    'stage.in_dim': STAGES.map(s => s.inDim),
    'stage.out_dim': STAGES.map(s => s.outDim),
    'stage.d_model': STAGES.map(s => s.d),
    'stage.n_head': STAGES.map(s => s.nh),
    'stage.n_layer': STAGES.map(s => s.nl),
    'stage.n_ff': STAGES.map(s => s.ff),
    'stage.context': STAGES.map(s => s.ctx),
    'stage.patch': STAGES.map(s => s.patch),
    lane_frames: LANE,
    context_frames: CONTEXT,
  };
  for (const [k, v] of Object.entries(want))
    if (JSON.stringify(meta[k]) !== JSON.stringify(v)) throw new Error('encoder: unsupported model; re-export with npm run export');
}

interface Semantic { prog: Program; input: GPUBuffer; output: GPUBuffer; outFloats: number }

export class RefEncoder {
  private front!: Program;
  private frontIn!: GPUBuffer;
  private frontOut!: GPUBuffer;
  private frontHists: GPUBuffer[] = [];
  private stages!: Program;
  private stageIn!: GPUBuffer;
  private stageOut!: GPUBuffer;
  private sem: Record<'enc' | 'dec', Semantic>;
  private rvq: Rvq[] = [];
  private internalKernel: Float32Array;
  private windowRatio: number;

  private constructor(private gpu: Gpu, private w: Model, readonly teacher: SslTeacher) {
    checkMeta(w.meta);
    const k = w.host.get('resample.kernel');
    if (!k) throw new Error('encoder: unsupported model; re-export with npm run export');
    this.internalKernel = k;
    this.windowRatio = Number(w.meta['window_ratio'] ?? 2);
    for (let c = 0; c < NCB; c++) this.rvq.push(new Rvq(w.host, c));
    this.recordFront();
    this.recordStages();
    this.sem = { enc: this.recordSemantic(true), dec: this.recordSemantic(false) };
  }

  static async load(gpu: Gpu, base: string, onProgress?: Progress, useCache = true): Promise<RefEncoder> {
    const [w, teacher] = await Promise.all([loadModel(gpu, base, 'encoder', onProgress, useCache),
      SslTeacher.load(gpu, base, onProgress, useCache)]);
    return new RefEncoder(gpu, w, teacher);
  }

  /** SEANet front for one 88320-sample tile: enc.in, two downsampling blocks, stage-0 patch-down + in-proj.
   *  Each causal conv reads a history-prefixed buffer; its last R input rows export to the history. */
  private recordFront(): void {
    const b = new Builder(this.gpu);
    const T = (n: string) => this.w.get(n).buf;
    const M = (n: string): Weight => this.w.get(n);
    // enc.in (K=7, 1 -> 64) as a 2-tap gemm over 16-sample rows; the host writes [previous 16 | tile]
    this.frontIn = b.buf((TILE / 16 + 1) * 16 * 4, 'enc.xin');
    let x = b.tmp(TILE * 64 * 4, 'enc.act');
    gemm(b, this.frontIn, M('enc.in.weight'), x, { T: TILE / 16, O: 1024, IC: 16, K: 2, d: 1, nb_bias: 64, rnd: CONV }, { bias: T('enc.in.bias') });
    let rows = TILE;
    const withHistory = (ch: number, R: number, n: number) => {
      const xf = b.tmp((R + n) * ch * 4, 'enc.xf');
      const hist = b.buf(R * ch * 4, 'enc.hist');
      this.frontHists.push(hist);
      b.copy(hist, 0, xf, 0, R * ch * 4);
      return { xf, hist };
    };
    for (let bi = 0; bi < 2; bi++) {
      const p = `enc.b${bi}`;
      const C = 64 << bi, s = bi === 0 ? 2 : 4;
      [1, 3, 9].forEach((dil, r) => {
        const u = `${p}.res${r}`;
        const R = 6 * dil;
        const h1 = withHistory(C, R, rows);
        snake(b, x, T(u + '.snake1.alpha'), T(u + '.snake1.inv_alpha'), h1.xf, rows, C, R * C);
        const h = b.tmp(rows * C * 4, 'enc.act');
        gemm(b, h1.xf, M(u + '.conv1.weight'), h, { T: rows, O: C, IC: C, K: 7, d: dil, rnd: CONV }, { bias: T(u + '.conv1.bias') });
        b.copy(h1.xf, rows * C * 4, h1.hist, 0, R * C * 4);
        b.release(h1.xf);
        const s2 = b.tmp(rows * C * 4, 'enc.act');
        snake(b, h, T(u + '.snake2.alpha'), T(u + '.snake2.inv_alpha'), s2, rows, C);
        b.release(h);
        const y = b.tmp(rows * C * 4, 'enc.act');
        gemm(b, s2, M(u + '.conv2.weight'), y, { T: rows, O: C, IC: C, rnd: CONV | RND.out }, { bias: T(u + '.conv2.bias'), res: x });
        b.release(s2);
        b.release(x);
        x = y;
      });
      // strided causal conv K=2s, left pad s: two taps over rows of s frames [previous | current]
      const n = rows / s, IC = s * C;
      const h = withHistory(IC, 1, n);
      snake(b, x, T(p + '.snake.alpha'), T(p + '.snake.inv_alpha'), h.xf, rows, C, IC);
      b.release(x);
      const y = b.tmp(n * 2 * C * 4, 'enc.act');
      gemm(b, h.xf, M(p + '.conv.weight'), y, { T: n, O: 2 * C, IC, K: 2, d: 1, rnd: CONV }, { bias: T(p + '.conv.bias') });
      b.copy(h.xf, n * IC * 4, h.hist, 0, IC * 4);
      b.release(h.xf);
      x = y;
      rows = n;
    }
    const pd = b.buf(TILE_ROWS * STAGES[0].inDim * 4, 'enc.pd');
    patchDown(b, x, pd, TILE_ROWS, 256, STAGES[0].patch);
    this.frontOut = b.buf(TILE_ROWS * STAGES[0].d * 4, 'enc.tileOut');
    gemm(b, pd, M('stage.0.in.weight'), this.frontOut, { T: TILE_ROWS, O: STAGES[0].d, IC: STAGES[0].inDim, rnd: RND.acc });
    this.front = b.finish();
  }

  /** MOSS stages over the whole lane; role buffers are sized for stage 0 and reused by later stages. */
  private recordStages(): void {
    const gpu = this.gpu;
    const b = new Builder(gpu);
    const T = (n: string) => this.w.get(n).buf;
    const M = (n: string): Weight => this.w.get(n);
    const maxTD = Math.max(...STAGES.map((s, i) => STAGE_T[i] * s.d));
    const maxTF = Math.max(...STAGES.map((s, i) => STAGE_T[i] * s.ff));
    const maxOut = Math.max(...STAGES.map((s, i) => STAGE_T[i] * s.outDim));
    const maxIn = Math.max(...STAGES.slice(1).map((s, i) => STAGE_T[i + 1] * s.inDim));
    const xa = b.buf(maxTD * 4, 'enc.xa'), xb = b.buf(maxTD * 4, 'enc.xb'), xn = b.buf(maxTD * 4, 'enc.xn');
    const q = b.buf(maxTD * 4, 'enc.q'), ao = b.buf(maxTD * 4, 'enc.ao');
    const kc = b.buf(maxTD * 4, 'enc.k'), vc = b.buf(maxTD * 4, 'enc.v');
    const qkv = b.buf(3 * maxTD * 4, 'enc.qkv');
    const f1 = b.buf(maxTF * 4, 'enc.f1');
    const y = b.buf(maxOut * 4, 'enc.sout');
    const pd = b.buf(maxIn * 4, 'enc.spd');
    const rope = gpu.upload(ropeTable(STAGE_T[0], 10000, HEAD_DIM), 'enc.rope');
    this.stageIn = xa;
    this.stageOut = y;
    STAGES.forEach((st, si) => {
      const n = STAGE_T[si], d = st.d, nh = d / HEAD_DIM;
      const sp = `stage.${si}.`;
      if (si > 0) {
        patchDown(b, y, pd, n, STAGES[si - 1].outDim, st.patch);
        gemm(b, pd, M(sp + 'in.weight'), xa, { T: n, O: d, IC: st.inDim, rnd: RND.acc });
      }
      const dyn = gpu.uniform(16, `enc.dyn${si}`);
      gpu.write(dyn, 0, new Uint32Array([0, n, 0, 0]));
      for (let l = 0; l < st.nl; l++) {
        const p = `${sp}l${l}.`;
        layernorm(b, xa, T(p + 'norm1.weight'), T(p + 'norm1.bias'), xn, n, d, LN_EPS);
        gemm(b, xn, M(p + 'qkv.weight'), qkv, { T: n, O: 3 * d, IC: d, rnd: RND.acc });
        ropeKv(b, qkv, rope, q, kc, vc, { n, nh, nkv: nh, hd: HEAD_DIM, in_stride: 3 * d, qkNorm: false, ring: n, pos0: 0, bf16: true });
        attentionWin(b, q, kc, vc, ao, { n, nh, window: st.ctx, ring: n, scale: 1 / Math.sqrt(HEAD_DIM), posMul: 1 }, dyn);
        gemm(b, ao, M(p + 'wo.weight'), xb, { T: n, O: d, IC: d, rnd: RND.acc }, { scale: T(p + 'attn_scale'), res: xa });
        layernorm(b, xb, T(p + 'norm2.weight'), T(p + 'norm2.bias'), xn, n, d, LN_EPS);
        gemm(b, xn, M(p + 'ffn1.weight'), f1, { T: n, O: st.ff, IC: d, act: ACT.gelu, rnd: RND.acc | RND.act });
        gemm(b, f1, M(p + 'ffn2.weight'), xa, { T: n, O: d, IC: st.ff, rnd: RND.acc }, { scale: T(p + 'ffn_scale'), res: xb });
        b.split();
      }
      gemm(b, xa, M(sp + 'out.weight'), y, { T: n, O: st.outDim, IC: d, rnd: RND.acc });
    });
    this.stages = b.finish();
  }

  /** Semantic encoder (in-proj + 4 ConvNeXt) or head (4 ConvNeXt + out-proj) over one lane; causal
   *  depthwise convs see 6 zero rows of left padding. */
  private recordSemantic(encode: boolean): Semantic {
    const b = new Builder(this.gpu);
    const T = (n: string) => this.w.get(n).buf;
    const M = (n: string): Weight => this.w.get(n);
    const n = LANE;
    const input = b.buf(n * (encode ? LATENT : SEM) * 4, 'sem.in');
    const xs = [b.buf((6 + n) * SEM * 4, 'sem.x0'), b.buf((6 + n) * SEM * 4, 'sem.x1')];
    for (const x of xs) b.clear(x, 0, 6 * SEM * 4);
    const off = 6 * SEM;
    let cur = 0;
    if (encode) gemm(b, input, M('sem_enc.in.weight'), xs[0], { T: n, O: SEM, IC: LATENT, y_off: off, rnd: CONV }, { bias: T('sem_enc.in.bias') });
    else b.copy(input, 0, xs[0], off * 4, n * SEM * 4);
    const h = b.buf(n * SEM * 4, 'sem.h'), hn = b.buf(n * SEM * 4, 'sem.hn'), h1 = b.buf(n * SEM_FF * 4, 'sem.h1');
    for (let bi = 0; bi < 4; bi++) {
      const p = `${encode ? 'sem_enc' : 'sem_head'}.${bi}.`;
      const x = xs[cur], nx = xs[cur ^ 1];
      dwconv(b, x, T(p + 'dw.weight'), h, n, SEM, 7, T(p + 'dw.bias'), 0, 1 | 2 | 4);
      layernorm(b, h, T(p + 'norm.weight'), T(p + 'norm.bias'), hn, n, SEM, CNX_EPS);
      gemm(b, hn, M(p + 'pw1.weight'), h1, { T: n, O: SEM_FF, IC: SEM, act: ACT.gelu, rnd: RND.bias | RND.act }, { bias: T(p + 'pw1.bias') });
      gemm(b, h1, M(p + 'pw2.weight'), nx, { T: n, O: SEM, IC: SEM_FF, r_off: off, y_off: off, rnd: RND.bias | (encode ? RND.out : 0) }, { bias: T(p + 'pw2.bias'), res: x });
      cur ^= 1;
    }
    let output: GPUBuffer, outFloats: number;
    if (encode) {
      output = xs[cur];
      outFloats = n * SEM;
    } else {
      output = b.buf(n * LATENT * 4, 'sem.out');
      outFloats = n * LATENT;
      gemm(b, xs[cur], M('sem_head.out.weight'), output, { T: n, O: LATENT, IC: SEM, x_off: off, rnd: CONV }, { bias: T('sem_head.out.bias') });
    }
    return { prog: b.finish(), input, output: encode ? xs[cur] : output, outFloats };
  }

  /** Acoustic latents [300][1024] of one 883200-sample internal-rate lane. */
  async acoustic(lane: Float32Array): Promise<Float32Array> {
    if (lane.length !== LANE_INTERNAL) throw new Error('encoder: acoustic lane must have 883200 samples');
    const gpu = this.gpu;
    const tile = new Float32Array(16 + TILE);
    const rowBytes = TILE_ROWS * STAGES[0].d * 4;
    for (let t = 0; t < TILES; t++) {
      const s0 = t * TILE;
      tile.fill(0, 0, 16);
      if (t > 0) tile.set(lane.subarray(s0 - 16, s0), 0);
      tile.set(lane.subarray(s0, s0 + TILE), 16);
      gpu.write(this.frontIn, 0, tile);
      await gpu.submitRead(enc => {
        if (t === 0) for (const h of this.frontHists) enc.clearBuffer(h);
        this.front.encode(enc);
        enc.copyBufferToBuffer(this.frontOut, 0, this.stageIn, t * rowBytes, rowBytes);
      });
    }
    const [out] = await this.stages.run(gpu, [{ buf: this.stageOut, offset: 0, bytes: LANE * LATENT * 4 }]);
    return out;
  }

  /** encode: pooled SSL [300][1024] -> semantic [300][768]; otherwise quantized semantic [300][768] -> prediction [300][1024]. */
  async semantic(x: Float32Array, encode: boolean): Promise<Float32Array> {
    const s = encode ? this.sem.enc : this.sem.dec;
    this.gpu.write(s.input, 0, x);
    const off = encode ? 6 * SEM * 4 : 0;
    const [out] = await this.gpu.submitRead(enc => s.prog.encode(enc), [{ buf: s.output, offset: off, bytes: s.outFloats * 4 }]);
    return out;
  }

  quantize(c: number, input: Float32Array): { ids: Int32Array; quantized: Float32Array } {
    return this.rvq[c].quantize(input, LANE);
  }

  /** Ordinary reference codes [frames][10] for mono 44.1 kHz audio (at most EDIT_MAX_SECONDS). */
  async encode(pcm44k: Float32Array): Promise<Int32Array> {
    const n = pcm44k.length;
    if (n < 1 || n > EDIT_MAX_SECONDS * SAMPLE_RATE || !pcm44k.every(Number.isFinite))
      throw new Error(ENCODE_ERROR);
    const frames = Math.ceil(n / HOP);
    const rs = new Resampler(this.internalKernel, 735, 736, 17);
    const a = rs.push(pcm44k), z = rs.flush();
    const internal = new Float32Array(frames * INTERNAL_HOP);
    internal.set(a.subarray(0, internal.length));
    internal.set(z.subarray(0, Math.max(0, internal.length - a.length)), Math.min(a.length, internal.length));
    const result = new Int32Array(frames * NCB);
    const laneAudio = new Float32Array(LANE_INTERNAL), teacherAudio = new Float32Array(LANE_SAMPLES);
    for (let start = 0; start < frames;) {
      const context = start === 0 ? 0 : CONTEXT;
      const count = Math.min(frames - start, LANE - context), left = start - context;
      laneAudio.fill(0);
      teacherAudio.fill(0);
      laneAudio.set(internal.subarray(left * INTERNAL_HOP, (left + context + count) * INTERNAL_HOP));
      const ts = left * HOP;
      teacherAudio.set(pcm44k.subarray(ts, ts + Math.min(n - ts, (context + count) * HOP)));
      const ssl = await this.teacher.encode(this.teacher.features(teacherAudio));
      const acoustic = await this.acoustic(laneAudio);
      const semantic = await this.semantic(poolSsl(ssl, SSL_FRAMES, this.windowRatio), true);
      const r0 = this.quantize(0, semantic);
      for (let t = 0; t < count; t++) result[(start + t) * NCB] = r0.ids[context + t];
      const prediction = await this.semantic(r0.quantized, false);
      const residual = new Float32Array(acoustic.length);
      for (let i = 0; i < residual.length; i++) residual[i] = acoustic[i] - prediction[i];
      for (let c = 1; c < NCB; c++) {
        const r = this.quantize(c, residual);
        for (let t = 0; t < count; t++) result[(start + t) * NCB + c] = r.ids[context + t];
        for (let i = 0; i < residual.length; i++) residual[i] -= r.quantized[i];
      }
      start += count;
    }
    return result;
  }
}

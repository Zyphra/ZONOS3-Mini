// Typed wrappers that record kernel dispatches into a Builder.
import { h8Grid } from './device';
import { linearWg, type Builder } from './program';
import type { GpuTensor } from './weights';

export const ACT = { none: 0, silu: 1, gelu: 2, relu: 3, tanh: 4, softcap: 5, newgelu: 6 } as const;
/** gemmF32 flags bit: apply `act` after the scale/shift epilogue (relu(BN(W.x + b))). */
export const ACT_AFTER_SHIFT = 16;
type Buf = GPUBuffer;
/** Matrix operand: f16 packed in u32, bf16 packed in u32, q8_0 (int8 quants + f32 per-32 block scales), q4_0
 *  (nibbles + f32 scales) or h8 (k-major f16 [K][IC][ceil32(O)], src/gpu/h8.ts). */
export type Weight = Pick<GpuTensor, 'buf' | 'scales' | 'dtype'>;
/** gemm `rnd` bits (bf16 weights only): round the epilogue to bf16 after the accumulation, +bias, act, *alpha, +res. */
export const RND = { acc: 1, bias: 2, act: 4, alpha: 8, out: 16 } as const;

export interface GemmOpts {
  T: number; O: number; IC: number; K?: number; d?: number;
  x_stride?: number; x_off?: number; y_stride?: number; y_off?: number; r_stride?: number; r_off?: number;
  nb_bias?: number; act?: number; alpha?: number; cap?: number;
  rowsDyn?: number;   // rows per valid frame: row tiles at or past dyn.b * rowsDyn are skipped
  rnd?: number;       // RND bits; bf16 weights only (their X operand is always rounded to bf16 as it is loaded)
  kernel?: string;    // h8 weights only: the gemm_h8 variant (gemm_h8_r<R>_<LX>x<LY> or gemm_h8_ks<S>)
}

/** Y = (act(conv(X, W) + bias) * scale + shift) * alpha + res; W f16, bf16, q8 [K][O][IC] or h8. Offsets in floats.
 *  T ≤ 32 with IC % 32 == 0 uses the skinny 16×64 tile (f16/q8); q8 weights need IC % 32 == 0. bf16 weights run the
 *  native BF16 autocast arithmetic: bf16 operands, f32 accumulation, bf16 rounding at the `rnd` points. h8 weights run
 *  `o.kernel` (IC % 16 == 0, x_stride and x_off multiples of 4; the split-K `ks` variants need T ≤ 32). */
export function gemm(b: Builder, X: Buf, W: Weight, Y: Buf, o: GemmOpts,
                     ex: { bias?: Buf; scale?: Buf; shift?: Buf; res?: Buf; dyn?: Buf } = {}): void {
  if (o.IC % 16) throw new Error(`gemm: IC=${o.IC} must be a multiple of 16`);
  if (o.rowsDyn && !ex.dyn) throw new Error('gemm: rowsDyn requires ex.dyn');
  const q8 = W.dtype === 'q8';
  const bf16 = W.dtype === 'bf16';
  if (W.dtype === 'h8') {
    const xs = o.x_stride ?? o.IC, xo = o.x_off ?? 0;
    if (!o.kernel || xs % 4 || xo % 4)
      throw new Error(`gemm: h8 needs a kernel and x_stride, x_off multiples of 4 (x_stride=${xs}, x_off=${xo})`);
    const ks = /^gemm_h8_ks(\d+)$/.exec(o.kernel);
    if (ks && (o.T > 32 || o.IC % (4 * Number(ks[1]))))
      throw new Error(`gemm: h8 needs T <= 32 and IC % ${4 * Number(ks[1])} == 0 for ${o.kernel} (T=${o.T}, IC=${o.IC})`);
  } else if (o.kernel) {
    throw new Error('gemm: kernel needs h8 weights');
  }
  if (q8 && (o.IC % 32 || !W.scales)) throw new Error(`gemm: q8 weights need IC % 32 == 0 (IC=${o.IC}) and scales`);
  if (!q8 && !bf16 && W.dtype !== 'f16' && W.dtype !== 'h8') throw new Error(`gemm: unsupported weight dtype ${W.dtype}`);
  if (o.rnd && !bf16) throw new Error('gemm: rnd needs bf16 weights');
  const flags = (ex.bias ? 1 : 0) | (ex.scale ? 2 : 0) | (ex.shift ? 4 : 0) | (ex.res ? 8 : 0);
  const skinny = !bf16 && o.T <= 32 && o.IC % 32 === 0;
  const io = { X, bias: ex.bias, scl: ex.scale, shf: ex.shift, res: ex.res, Y, dyn: ex.dyn ?? b.gpu.dummyUniform() };
  const params = {
    T: o.T, O: o.O, IC: o.IC, K: o.K ?? 1, d: o.d ?? 1,
    x_stride: o.x_stride ?? o.IC, x_off: o.x_off ?? 0, y_stride: o.y_stride ?? o.O, y_off: o.y_off ?? 0,
    r_stride: o.r_stride ?? o.O, r_off: o.r_off ?? 0, nb_bias: o.nb_bias ?? o.O, flags,
    act: o.act ?? 0, alpha: o.alpha ?? 1, cap: o.cap ?? 0, rows_dyn: o.rowsDyn ?? 0,
  };
  if (bf16) {
    b.dispatch('gemm_bf16', { ...io, Wt: W.buf }, { ...params, rnd: o.rnd ?? 0 }, [Math.ceil(o.O / 64), Math.ceil(o.T / 64), 1]);
    return;
  }
  if (W.dtype === 'h8') {
    b.dispatch(o.kernel!, { ...io, W: W.buf }, params, h8Grid(o.kernel!, o.T, o.O));
    return;
  }
  b.dispatch((skinny ? 'gemm_skinny' : 'gemm') + (q8 ? '_q8' : ''), q8 ? { ...io, Wq: W.buf, Wsc: W.scales } : { ...io, Wt: W.buf },
    params, skinny ? [Math.ceil(o.O / 64), Math.ceil(o.T / 16), 1] : [Math.ceil(o.O / 64), Math.ceil(o.T / 64), 1]);
}

/** y[y_off + o] = act(W[o] . x[x_off:] + bias) + res[o]; W f16 [O][I] (I % 8 == 0) or q8 (I % 32 == 0); x_off % 4 == 0
 *  (x is read as vec4). 16 rows per workgroup. */
export function matvec(b: Builder, x: Buf, W: Weight, y: Buf,
                       o: { O: number; I: number; x_off?: number; y_off?: number; nb_bias?: number; act?: number; cap?: number },
                       ex: { bias?: Buf; res?: Buf } = {}): void {
  const q8 = W.dtype === 'q8';
  if (q8) {
    if (o.I % 32 !== 0 || !W.scales) throw new Error(`matvec: q8 weights need I % 32 == 0 (I=${o.I}) and scales`);
  } else if (W.dtype !== 'f16') {
    throw new Error(`matvec: unsupported weight dtype ${W.dtype}`);
  } else if (o.I % 8) {
    throw new Error(`matvec: I=${o.I} must be a multiple of 8`);
  }
  if ((o.x_off ?? 0) % 4) throw new Error('matvec: x_off must be a multiple of 4');
  const bufs = q8
    ? { x, Wq: W.buf, Wsc: W.scales, bias: ex.bias, res: ex.res, y }
    : { x, Wt: W.buf, bias: ex.bias, res: ex.res, y };
  b.dispatch(q8 ? 'matvec_q8' : 'matvec', bufs, {
    O: o.O, I: o.I, x_off: o.x_off ?? 0, y_off: o.y_off ?? 0, nb_bias: o.nb_bias ?? o.O,
    flags: (ex.bias ? 1 : 0) | (ex.res ? 8 : 0), act: o.act ?? 0, cap: o.cap ?? 0,
  }, linearWg(Math.ceil(o.O / 16)));
}

export function matvecGlu(b: Builder, x: Buf, Wu: Buf, Wg: Buf, y: Buf, O: number, I: number): void {
  b.dispatch('matvec_glu', { x, Wu, Wg, y }, { O, I }, linearWg(Math.ceil(O / 8)));
}

/** out[i] = rmsnorm(res[row0+i] (+= x)) * w, i < rows. */
export function addRmsnorm(b: Builder, res: Buf, out: Buf,
                           o: { N: number; rows: number; eps: number; row0?: number },
                           ex: { x?: Buf; w?: Buf } = {}): void {
  b.dispatch('add_rmsnorm', { res, x: ex.x, w: ex.w, outp: out }, {
    N: o.N, eps: o.eps, has_x: ex.x ? 1 : 0, has_w: ex.w ? 1 : 0, row0: o.row0 ?? 0,
  }, [o.rows, 1, 1]);
}

/** Full AttnRes route + weighted RMSNorm: out[i] = rmsnorm(softmax_s(q[qi].rms(src_s)) . src_s) * w, row r = row0 + i.
 *  Sources: bank[0..ns) ([ns][n][D]); q f16 [23][D]; D <= 1024, ns <= 32. */
export function attnres(b: Builder, bank: Buf, q: Buf, w: Buf, out: Buf,
                        o: { D: number; n: number; ns: number; qi: number; rows: number; row0?: number; eps: number }): void {
  if (o.D > 1024 || o.ns > 32 || o.ns < 1) throw new Error(`attnres: D=${o.D} ns=${o.ns} out of range`);
  b.dispatch('attnres', { bank, q, w, outp: out },
    { D: o.D, n: o.n, ns: o.ns, qi: o.qi, row0: o.row0 ?? 0, eps: o.eps }, [o.rows, 1, 1]);
}

/** GDN (4 heads x 128, conv kernel 4) on n rows of `mixed` [n][2056] = q|k|v|z|a|b; carries `state`
 *  [4][128][128] and `hist` [1536][3] across calls. Writes gated [n][512] (input of attn_output). */
export function gdn(b: Builder, mixed: Buf, n: number, wts: { conv: Buf; dtBias: Buf; aLog: Buf; norm: Buf },
                    state: Buf, hist: Buf, gated: Buf,
                    scratch: { conv: Buf; qn: Buf; kn: Buf; decay: Buf; beta: Buf; o: Buf }): void {
  const s = scratch;
  b.dispatch('gdn_conv', { mixed, w: wts.conv, hist, outp: s.conv }, { n, stride: 2056 }, linearWg(6));
  b.dispatch('gdn_prep', { conv: s.conv, mixed, dt_bias: wts.dtBias, a_log: wts.aLog, qn: s.qn, kn: s.kn, decay: s.decay, beta: s.beta },
    { n }, linearWg(n * 4));
  b.dispatch('gdn_scan', { qn: s.qn, kn: s.kn, conv: s.conv, decay: s.decay, beta: s.beta, state, outp: s.o }, { n }, [8, 1, 1]);
  b.dispatch('gdn_out', { o: s.o, mixed, nw: wts.norm, outp: gated }, { n }, linearWg(n * 4));
}

/** Top-2 latent MoE experts (64 experts, latent 256, expert ff 1536, q4): router logits -> latent out [n][256]. */
export function moe(b: Builder, lg: Buf, rbias: Buf, z: Buf, up: Weight, gate: Weight, down: Weight, out: Buf, n: number,
                    scratch: { sel: Buf; wts: Buf; hbuf: Buf }): void {
  if (up.dtype !== 'q4' || gate.dtype !== 'q4' || down.dtype !== 'q4') throw new Error('moe: experts must be q4; re-export with npm run export');
  const { sel, wts, hbuf } = scratch;
  b.dispatch('moe_topk', { lg, rbias, sel, wts }, {}, [n, 1, 1]);
  b.dispatch('moe_up', { x: z, sel, gq: gate.buf, gs: gate.scales, uq: up.buf, us: up.scales, hbuf }, {}, [48, 2, n]);
  b.dispatch('moe_down', { hbuf, sel, wts, dq: down.buf, ds: down.scales, outp: out }, {}, [32, 1, n]);
}

/** Time-major patch up: y[t*h + j][ch] = x[t][ch*h + j], C = output channels. */
export function patchUp(b: Builder, x: Buf, y: Buf, T: number, C: number, h: number): void {
  const n = T * h * C;
  b.dispatch('patch_up', { x, y }, { n, C, h }, linearWg(Math.ceil(n / 256)));
}

/** Time-major patch down: y[t][ch*h + j] = x[t*h + j][ch]; T output rows, C input channels. */
export function patchDown(b: Builder, x: Buf, y: Buf, T: number, C: number, h: number): void {
  const n = T * h * C;
  b.dispatch('patch_down', { x, y }, { n, C, h }, linearWg(Math.ceil(n / 256)));
}

export function layernorm(b: Builder, x: Buf, w: Buf, bias: Buf, y: Buf, rows: number, N: number, eps: number, act = 0): void {
  b.dispatch('layernorm', { x, w, b: bias, y }, { N, eps, act }, linearWg(rows));
}

export const EW = { siluMul: 0, gluSplit: 1, add: 2, snake: 3, gluSplitBf16: 4 } as const;

export function elementwise(b: Builder, op: number, n: number, a: Buf, y: Buf,
                            o: { C?: number; a_off?: number; y_off?: number } = {}, bb?: Buf, cc?: Buf): void {
  b.dispatch('elementwise', { a, b: bb, c: cc, y }, { op, n, C: o.C ?? 1, a_off: o.a_off ?? 0, y_off: o.y_off ?? 0 },
    linearWg(Math.ceil(n / 256)));
}

export function snake(b: Builder, x: Buf, alpha: Buf, invAlpha: Buf, y: Buf, T: number, C: number, y_off = 0): void {
  elementwise(b, EW.snake, T * C, x, y, { C, y_off }, alpha, invAlpha);
}

/** Causal depthwise conv; `rnd` bits round to bf16: 1 the inputs as loaded, 2 the tap sum, 4 the output (after bias/act). */
export function dwconv(b: Builder, Xf: Buf, w: Buf, y: Buf, T: number, C: number, K: number, bias?: Buf, act = 0, rnd = 0): void {
  b.dispatch('dwconv', { Xf, w, b: bias, y }, { T, C, K, has_bias: bias ? 1 : 0, act, rnd }, linearWg(Math.ceil(T * C / 256)));
}

export interface AttnOpts {
  n: number; nh: number; nkv: number; hd: number; window: number; ring: number; scale: number;
  pos0?: number; posDyn?: boolean; gate_stride?: number; gate_off?: number;
}

/** Head dim 32 (the backbone's) runs the register-resident attention_hd32 kernel; others (≤ 64) the generic one. */
export function attention(b: Builder, q: Buf, kc: Buf, vc: Buf, out: Buf, o: AttnOpts, ex: { gate?: Buf; dyn?: Buf } = {}): void {
  if (o.hd > 64) throw new Error(`attention: hd=${o.hd} > 64`);
  b.dispatch(o.hd === 32 ? 'attention_hd32' : 'attention', { q, kc, vc, gate: ex.gate, outp: out, dyn: ex.dyn ?? b.gpu.dummyUniform() }, {
    n: o.n, nh: o.nh, nkv: o.nkv, pos_dyn: o.posDyn ? 1 : 0, pos0: o.pos0 ?? 0, window: o.window, ring: o.ring,
    scale: o.scale, has_gate: ex.gate ? 1 : 0, gate_stride: o.gate_stride ?? 0, gate_off: o.gate_off ?? 0,
    hd: o.hd,
  }, linearWg(o.n * o.nh));
}

export interface WinAttnOpts { n: number; nh: number; window: number; ring: number; scale: number; posMul: number }

/** Codec windowed attention (head dim 64, nkv == nh, no gate): positions dyn.a*posMul + i; 16-row tiles past dyn.b*posMul skipped. */
export function attentionWin(b: Builder, q: Buf, kc: Buf, vc: Buf, out: Buf, o: WinAttnOpts, dyn: Buf): void {
  b.dispatch('attention_win', { q, kc, vc, outp: out, dyn },
    { n: o.n, nh: o.nh, window: o.window, ring: o.ring, scale: o.scale, pos_mul: o.posMul }, [Math.ceil(o.n / 16), o.nh, 1]);
}

export interface RopeOpts {
  n: number; nh: number; nkv: number; hd: number; in_stride: number; qkNorm: boolean; eps?: number; ring: number;
  pos0?: number; posDyn?: boolean; nDyn?: boolean; posMul?: number;
  bf16?: boolean;   // round the roped q and k to bf16 (native BF16 encoder recipe)
}

export function ropeKv(b: Builder, qkv: Buf, rope: Buf, q: Buf, kc: Buf, vc: Buf, o: RopeOpts,
                       ex: { temp?: Buf; dyn?: Buf } = {}): void {
  if (o.hd > 64 || o.hd % 2) throw new Error(`ropeKv: hd=${o.hd} must be even and <= 64`);
  b.dispatch('rope_kv', { qkv, temp: ex.temp, rope, qo: q, kc, vc, dyn: ex.dyn ?? b.gpu.dummyUniform() }, {
    n: o.n, nh: o.nh, nkv: o.nkv, in_stride: o.in_stride, qk_norm: o.qkNorm ? 1 : 0, has_temp: ex.temp ? 1 : 0,
    eps: o.eps ?? 0, ring: o.ring, pos_dyn: o.posDyn ? 1 : 0, pos0: o.pos0 ?? 0, n_dyn: o.nDyn ? 1 : 0,
    hd: o.hd, pos_mul: o.posMul ?? 1, rnd: o.bf16 ? 1 : 0,
  }, linearWg(o.n * (o.nh + 2 * o.nkv)));
}

/** Conditioning rows of n embedding rows at positions dyn.a + t (cond_rows.wgsl): speaker row / cond_embd.i[bucket]. */
export function condRows(b: Builder, emb: Buf, cond: Buf, tables: [Buf, Buf, Buf], n: number, dyn: Buf): void {
  b.dispatch('cond_rows', { emb, cond, t0: tables[0], t1: tables[1], t2: tables[2], dyn }, { n }, [4, n, 1]);
}

/** x[x_off + i] += d[d_off + i], i < n, when dyn.b != 0. */
export function steerAdd(b: Builder, x: Buf, d: Buf, n: number, x_off: number, d_off: number, dyn: Buf): void {
  b.dispatch('steer_add', { x, d, dyn }, { n, x_off, d_off }, linearWg(Math.ceil(n / 256)));
}

export interface ProbeOpts { nl: number; nh: number; nkv: number; hd: number; ring: number; scale: number; lmax: number }

/** Probe readout softmax of probe layer `layer` over each weighted head's text window into sc [nl][nh][lmax]. */
export function probeReadout(b: Builder, q: Buf, kc: Buf, pin: Buf, sc: Buf, layer: number, o: ProbeOpts): void {
  if (o.hd > 64) throw new Error(`probeReadout: hd=${o.hd} > 64`);
  b.dispatch('probe', { q, kc, pin, sc, acc: null }, { mode: 0, layer, ...o }, [o.nh, 1, 1]);
}

/** acc[j] = sum over probe layers and weighted heads of w * sc (j < lmax). */
export function probeAccumulate(b: Builder, pin: Buf, sc: Buf, acc: Buf, o: ProbeOpts): void {
  b.dispatch('probe', { pin, sc, acc }, { mode: 1, ...o }, [Math.ceil(o.lmax / 256), 1, 1]);
}

export interface MaskedAttnOpts { nh: number; nkv: number; hd: number; ring: number; scale: number; gate_off: number }

/** Decode-row attention of the window-flagged heads of probe layer `layer` with out-of-window text keys blocked. */
export function attentionMasked(b: Builder, q: Buf, kc: Buf, vc: Buf, gate: Buf, pin: Buf, out: Buf, layer: number,
                                o: MaskedAttnOpts, dyn: Buf): void {
  if (o.hd > 64) throw new Error(`attentionMasked: hd=${o.hd} > 64`);
  b.dispatch('attention_masked', { q, kc, vc, gate, pin, outp: out, dyn }, { layer, ...o }, [o.nh, 1, 1]);
}

/** gemm with f32 weights [K][O][IC]; same epilogue as gemm, with `act` moved after scale/shift when actAfterShift. */
export function gemmF32(b: Builder, X: Buf, W: Weight, Y: Buf, o: GemmOpts & { actAfterShift?: boolean },
                        ex: { bias?: Buf; scale?: Buf; shift?: Buf; res?: Buf; dyn?: Buf } = {}): void {
  if (o.IC % 16) throw new Error(`gemmF32: IC=${o.IC} must be a multiple of 16`);
  if (W.dtype !== 'f32') throw new Error(`gemmF32: unsupported weight dtype ${W.dtype}`);
  if (o.rowsDyn && !ex.dyn) throw new Error('gemmF32: rowsDyn requires ex.dyn');
  const flags = (ex.bias ? 1 : 0) | (ex.scale ? 2 : 0) | (ex.shift ? 4 : 0) | (ex.res ? 8 : 0) | (o.actAfterShift ? ACT_AFTER_SHIFT : 0);
  b.dispatch('gemm_f32', { X, Wt: W.buf, bias: ex.bias, scl: ex.scale, shf: ex.shift, res: ex.res, Y, dyn: ex.dyn ?? b.gpu.dummyUniform() }, {
    T: o.T, O: o.O, IC: o.IC, K: o.K ?? 1, d: o.d ?? 1,
    x_stride: o.x_stride ?? o.IC, x_off: o.x_off ?? 0, y_stride: o.y_stride ?? o.O, y_off: o.y_off ?? 0,
    r_stride: o.r_stride ?? o.O, r_off: o.r_off ?? 0, nb_bias: o.nb_bias ?? o.O, flags,
    act: o.act ?? 0, alpha: o.alpha ?? 1, cap: o.cap ?? 0, rows_dyn: o.rowsDyn ?? 0,
  }, [Math.ceil(o.O / 64), Math.ceil(o.T / 64), 1]);
}

export interface Gconv3x3Opts { T_in: number; F_in: number; IC: number; OC: number; a: number; s_f: number; s_t: number; act?: number }

/** Grouped 3x3 conv, pad 1, channel-last [T][F][C]; w f32 [OC][a][3][3] (kh along F, kw along T), group of oc =
 *  oc / (OC / (IC / a)) reading input channels group*a + j. Epilogue: +bias, *scale + shift, act. Returns [T_out, F_out]. */
export function gconv3x3(b: Builder, x: Buf, w: Buf, y: Buf, o: Gconv3x3Opts,
                         ex: { bias?: Buf; scale?: Buf; shift?: Buf } = {}): [number, number] {
  if (o.IC % o.a || o.OC % (o.IC / o.a)) throw new Error(`gconv3x3: IC=${o.IC} a=${o.a} OC=${o.OC} do not form groups`);
  const T_out = Math.floor((o.T_in - 1) / o.s_t) + 1, F_out = Math.floor((o.F_in - 1) / o.s_f) + 1;
  b.dispatch('gconv3x3', { x, w, bias: ex.bias, scl: ex.scale, shf: ex.shift, y }, {
    T_in: o.T_in, F_in: o.F_in, IC: o.IC, OC: o.OC, a: o.a, bg: o.OC / (o.IC / o.a), s_f: o.s_f, s_t: o.s_t,
    T_out, F_out, flags: (ex.bias ? 1 : 0) | (ex.scale ? 2 : 0) | (ex.shift ? 4 : 0), act: o.act ?? 0,
  }, linearWg(Math.ceil(T_out * F_out * o.OC / 256)));
  return [T_out, F_out];
}

/** y[t][c] = x[t*s][c] for T_out rows of `row` floats. */
export function subsampleRows(b: Builder, x: Buf, y: Buf, T_out: number, row: number, s: number): void {
  const n = T_out * row;
  b.dispatch('subsample_rows', { x, y }, { n, row, s }, linearWg(Math.ceil(n / 256)));
}

/** Dense 4-expert mix: y[t][k] = newgelu(sum_i softmax(lg[t])[i] * e[t][64i + k]); lg [T][4], e [T][256], y [T][64]. */
export function moeDenseMix(b: Builder, lg: Buf, e: Buf, y: Buf, T: number): void {
  b.dispatch('moe_dense_mix', { lg, e, y }, { T }, linearWg(Math.ceil(T * 64 / 256)));
}

/** y[t][k] = e[floor(t / s)][k] + xf[t][k] for T rows of N floats. */
export function upsampleAdd(b: Builder, e: Buf, xf: Buf, y: Buf, T: number, N: number, s: number): void {
  const n = T * N;
  b.dispatch('upsample_add', { e, xf, y }, { n, N, s }, linearWg(Math.ceil(n / 256)));
}

/** acc[t][k] = (S == 0 ? 0 : acc[t][k]) + feat[t][k] * w[S*N + k] for T rows of N floats. */
export function denseFuse(b: Builder, feat: Buf, w: Buf, acc: Buf, T: number, N: number, S: number): void {
  const n = T * N;
  b.dispatch('dense_fuse', { feat, w, acc }, { n, N, S }, linearWg(Math.ceil(n / 256)));
}

/** Attentive statistics pooling over T rows of N columns: pooled [2N] = mean | sqrt(clamp(var, varFloor, 3e38)). */
export function astpPool(b: Builder, att: Buf, x: Buf, pooled: Buf, T: number, N: number, varFloor: number): void {
  b.dispatch('astp_pool', { att, x, pooled }, { T, N, var_floor: varFloor }, linearWg(N));
}

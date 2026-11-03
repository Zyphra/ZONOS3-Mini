// zonos3 GDN backbone on WebGPU: 9 Gated-DeltaNet + 2 global-attention layers, Full AttnRes over every
// sublayer output, dense layer-0 FFN and 64-expert top-2 latent MoE. Chunked prefill (gemm) and a
// pre-recorded single-row decode (matvec). Weights: Q4_0 experts, Q8_0 mixers, F16 spine. Mirrors build_graph in
// zonos3-graph.cpp: speaker/condition rows replace the summed embedding before emb_norm (prefill), hidden steering
// deltas join their Full AttnRes sources (decode), and the alignment probe reads the global layers (slot 0 decode).
// Two state sets: slot 0 = conditional, slot 1 = CFG twin.
import type { Gpu } from './gpu/device';
import {
  ACT, addRmsnorm, attention, attentionMasked, attnres, condRows, elementwise, EW, gdn, gemm, matvec, matvecGlu, moe,
  probeAccumulate, probeReadout, ropeKv, steerAdd, type ProbeOpts,
} from './gpu/ops';
import { Builder, type Program } from './gpu/program';
import { ropeTable } from './gpu/rope';
import type { Model } from './gpu/weights';
import { clipWindow } from './probe';
import type { Conditioning } from './prompt';

export const N_LAYER = 11;
export const D = 1024;
export const HD = 32;
export const NH = 16;
export const NKV = 4;
export const QKVG = NH * HD + 2 * NKV * HD + NH; // 784: q | k | v | gate
const GATE_OFF = NH * HD + 2 * NKV * HD;         // 768
export const GDN_H = 4;
export const GDN_S = 128;
export const GDN_I = GDN_H * GDN_S;              // 512
export const GDN_PROJ = 4 * GDN_I + 2 * GDN_H;   // 2056: q | k | v | z | a | b
export const GDN_CONV = 3 * GDN_I;               // 1536
export const N_FF = 2048;
export const N_LATENT = 256;
export const N_EXP_FF = 1536;
export const N_EXPERT = 64;
export const TOPK = 2;
export const VOCAB = 4098;
export const NCB = 10;
export const LOGITS = VOCAB * NCB;
export const MAX_FRAMES = 2000;
/** Rows per slot: native context_capacity = min(context_length, max_frames + 4096). */
export const CAP = MAX_FRAMES + 4096;
export const CHUNK = 256;
/** Text rows the alignment probe reads (sgl --align-probe-max-text). */
export const PROBE_LMAX = 2048;
const SPK_DIM = 192;
const N_COND = 3;
/** cond_embd rows per condition: buckets 0..9 plus the unknown row 10. */
const N_BUCKETS = 10;
const AUDIO_PAD = 4097;
const TEXT_PAD = 8196;
/** Global attention layers in probe order. */
const PROBE_LAYERS = [3, 7];
/** condBuf: speaker_row, condition_row, n_cond, bucket[3], 0, 0 (u32 words), then the projected speaker row [1024]. */
const COND_HDR = 8;
/** probeBuf: text_start, text_len, 0, 0, then per probe layer and head: weight (f32), lo, hi, flag. */
const PROBE_WORDS = 4 + PROBE_LAYERS.length * NH * 4;
/** AttnRes sources: the normalized embedding plus every attention and FFN output. */
const N_SRC = 2 * N_LAYER + 1;
const ATTN_SCALE = 1 / Math.sqrt(HD);
const RMS_EPS = 1e-5;
const QK_EPS = 1e-6;
const SOFTCAP = 15;
const KV_BYTES = CAP * NKV * HD * 4;
const STATE_BYTES = GDN_H * GDN_S * GDN_S * 4;
const HIST_BYTES = GDN_CONV * 3 * 4;

export const isGlobal = (L: number) => (L + 1) % 4 === 0;

interface Capture { name: string; buf: GPUBuffer; off: number; floats: number }

export interface PrefillResult { logits: Float32Array; caps?: Map<string, Float32Array> }
export interface PrefillOpts { chunk?: number; capture?: boolean }

interface SlotGpuState { kc: Map<number, GPUBuffer>; vc: Map<number, GPUBuffer>; gdnState: Map<number, GPUBuffer>; gdnHist: Map<number, GPUBuffer> }

function isPadRow(ids: Int32Array, r: number): boolean {
  for (let c = 0; c < NCB; c++) if (ids[r * 11 + c] !== AUDIO_PAD) return false;
  return ids[r * 11 + NCB] === TEXT_PAD;
}

/** zonos3_check_conditioning + features_ok (speaker delta). */
function checkConditioning(ids: Int32Array, n: number, c: Conditioning, condNames: string[]): void {
  const padRow = (row: number) => {
    if (row < 0 || row >= n || !isPadRow(ids, row)) throw new Error(`zonos3: conditioning row ${row} is not an all-pad row`);
  };
  if (c.speakerRow >= 0) {
    if (!c.speaker || c.speaker.length !== SPK_DIM) throw new Error(`zonos3: expected a ${SPK_DIM}-value speaker embedding`);
    for (const v of c.speaker) if (!Number.isFinite(v)) throw new Error('zonos3: speaker embedding contains a non-finite value');
    padRow(c.speakerRow);
  }
  if (c.conditionRow >= 0) {
    if (c.buckets.length !== N_COND) throw new Error(`zonos3: expected ${N_COND} condition buckets`);
    for (let i = 0; i < N_COND; i++) {
      if (!Number.isInteger(c.buckets[i]) || c.buckets[i] < 0 || c.buckets[i] > N_BUCKETS)
        throw new Error(`zonos3: condition bucket ${c.buckets[i]} for ${condNames[i]} outside 0..${N_BUCKETS - 1} (${N_BUCKETS} = unknown)`);
      padRow(c.conditionRow + i);
    }
  }
  for (let r = 0; r < n; r++) {
    if (!isPadRow(ids, r)) continue;
    const speaker = c.speakerRow >= 0 && r === c.speakerRow;
    const cond = c.conditionRow >= 0 && r >= c.conditionRow && r < c.conditionRow + N_COND;
    if (!speaker && !cond) throw new Error(`zonos3: all-pad row ${r} has no speaker/condition embedding`);
  }
  if (c.speakerDelta) {
    if (c.speakerRow < 0 || c.speakerDelta.length !== D)
      throw new Error(`zonos3: a speaker-row delta needs a speaker row and ${D} values`);
    for (const v of c.speakerDelta) if (!Number.isFinite(v)) throw new Error('zonos3: speaker-row delta contains a non-finite value');
  }
}

function checkMeta(meta: Record<string, unknown>): void {
  if (meta['general.architecture'] !== 'zonos3' || meta['residual.type'] !== 'full_attnres' ||
      meta['attention.pattern'] !== 'gdn3_global1' || Number(meta['speaker.embedding_length']) !== SPK_DIM ||
      Number(meta['condition.count']) !== N_COND || Number(meta['condition.bucket_count']) !== N_BUCKETS)
    throw new Error('backbone: unsupported model; re-export with pnpm export');
}

/** Per-step alignment probe (slot 0 decode): readout heads (weight 1) and ACI windows (unclipped text indices). */
export interface ProbeInput {
  textStart: number; textLen: number;
  heads: { layer: number; head: number }[];
  windows: { head: { layer: number; head: number }; lo: number; hi: number }[];
}

export class Backbone {
  readonly steerSources: number[];
  private rope: GPUBuffer;
  private slots: [SlotGpuState, SlotGpuState];
  private ids: GPUBuffer;
  private chunkIds: GPUBuffer;
  private dyn: GPUBuffer;
  private dynData = new Uint32Array(4);   // a = position of the first row, b = steering on (decode)
  private condBuf: GPUBuffer;
  private spkIn: GPUBuffer;
  private spkDelta: GPUBuffer;
  private spkProg: Program;
  private steerBuf: GPUBuffer;
  private probeBuf: GPUBuffer;
  private probeSc: GPUBuffer;
  private probeAcc: GPUBuffer;
  private decodeProg: [Program, Program];
  private decodeLogits: [GPUBuffer, GPUBuffer];
  private chunkProg: [Program, Program];
  private chunkLogits: [GPUBuffer, GPUBuffer];
  nPast: [number, number] = [0, 0];
  private condNames: string[];

  constructor(private gpu: Gpu, private w: Model, steerSources: number[]) {
    checkMeta(w.meta);
    const names = w.meta['condition.names'];
    this.condNames = Array.isArray(names) ? names.map(String) : [];
    for (let i = 0; i < steerSources.length; i++) {
      const s = steerSources[i];
      if (!Number.isInteger(s) || s < 0 || s > 2 * N_LAYER || (i > 0 && s <= steerSources[i - 1]))
        throw new Error(`zonos3: steering sources must be ascending Full AttnRes bank indices in 0..${2 * N_LAYER}`);
    }
    this.steerSources = steerSources.slice();
    this.rope = gpu.upload(ropeTable(CAP, Number(w.meta['rope.freq_base'] ?? 10000), HD), 'bb.rope');
    const state = (k: number): SlotGpuState => {
      const st: SlotGpuState = { kc: new Map(), vc: new Map(), gdnState: new Map(), gdnHist: new Map() };
      for (let L = 0; L < N_LAYER; L++) {
        if (isGlobal(L)) {
          st.kc.set(L, gpu.buffer(KV_BYTES, `s${k}.kc${L}`));
          st.vc.set(L, gpu.buffer(KV_BYTES, `s${k}.vc${L}`));
        } else {
          st.gdnState.set(L, gpu.buffer(STATE_BYTES, `s${k}.gdnState${L}`));
          st.gdnHist.set(L, gpu.buffer(HIST_BYTES, `s${k}.gdnHist${L}`));
        }
      }
      return st;
    };
    this.slots = [state(0), state(1)];
    this.ids = gpu.buffer(11 * 4, 'bb.ids');
    this.chunkIds = gpu.buffer(CHUNK * 11 * 4, 'bb.chunkIds');
    this.dyn = gpu.uniform(16, 'bb.dyn');
    this.condBuf = gpu.buffer((COND_HDR + D) * 4, 'bb.cond');
    this.spkIn = gpu.buffer(SPK_DIM * 4, 'bb.spkIn');
    this.spkDelta = gpu.buffer(D * 4, 'bb.spkDelta');
    this.steerBuf = gpu.buffer(Math.max(1, this.steerSources.length) * D * 4, 'bb.steer');
    this.probeBuf = gpu.buffer(PROBE_WORDS * 4, 'bb.probe');
    this.probeSc = gpu.buffer(PROBE_LAYERS.length * NH * PROBE_LMAX * 4, 'bb.probeSc');
    this.probeAcc = gpu.buffer(PROBE_LMAX * 4, 'bb.probeAcc');
    let b = new Builder(gpu);
    // Projected speaker row = speaker_proj.weight . speaker + bias (+ delta) into condBuf (zonos3-graph.cpp:515-521).
    matvec(b, this.spkIn, w.get('speaker_proj.weight'), this.condBuf, { O: D, I: SPK_DIM, y_off: COND_HDR },
      { bias: w.get('speaker_proj.bias').buf, res: this.spkDelta });
    this.spkProg = b.finish();
    const decodeLogits: GPUBuffer[] = [], decodeProg: Program[] = [], chunkLogits: GPUBuffer[] = [], chunkProg: Program[] = [];
    for (const s of [0, 1] as const) {
      b = new Builder(gpu);
      decodeLogits.push(this.record(b, 1, true, this.ids, null, s));
      decodeProg.push(b.finish());
      b = new Builder(gpu);
      chunkLogits.push(this.record(b, CHUNK, false, this.chunkIds, null, s));
      chunkProg.push(b.finish());
    }
    this.decodeLogits = [decodeLogits[0], decodeLogits[1]];
    this.decodeProg = [decodeProg[0], decodeProg[1]];
    this.chunkLogits = [chunkLogits[0], chunkLogits[1]];
    this.chunkProg = [chunkProg[0], chunkProg[1]];
  }

  /**
   * Record the forward pass of slot `slot` for n rows at absolute positions dyn.a + i. decode=true uses matvec,
   * adds the hidden steering deltas (gated by dyn.b) and, on slot 0, runs the alignment probe. Prefill applies the
   * conditioning rows of condBuf. Returns the logits buffer [40980] (last row only).
   */
  private record(b: Builder, n: number, decode: boolean, ids: GPUBuffer, caps: Capture[] | null, slot: 0 | 1): GPUBuffer {
    const w = this.w;
    const T = (name: string) => w.get(name).buf;
    const dyn = this.dyn;
    const st = this.slots[slot];
    const probing = decode && slot === 0;
    const cap = (name: string, buf: GPUBuffer, floats: number, off = 0) => {
      if (!caps) return;
      const c = b.buf(floats * 4, name);
      b.copy(buf, off * 4, c, 0, floats * 4);
      caps.push({ name, buf: c, off: 0, floats });
    };
    const lin = (x: GPUBuffer, W: string, y: GPUBuffer, O: number, I: number, y_off = 0) => {
      if (decode) matvec(b, x, w.get(W), y, { O, I, y_off });
      else gemm(b, x, w.get(W), y, { T: n, O, IC: I, y_off });
    };
    const src = (s: number) => s * n * D;   // bank float offset of source s
    // Hidden-space steering: the delta joins source s as it enters the bank (zonos3-graph.cpp:610-615).
    const steer = (s: number) => {
      const k = this.steerSources.indexOf(s);
      if (decode && k >= 0) steerAdd(b, bank, this.steerBuf, D, src(s), k * D, dyn);
    };
    const po: ProbeOpts = { nl: PROBE_LAYERS.length, nh: NH, nkv: NKV, hd: HD, ring: CAP, scale: ATTN_SCALE, lmax: PROBE_LMAX };

    const emb = b.buf(n * D * 4, 'emb');
    b.dispatch('embed', { ids, E: T('embd_all'), outp: emb }, { n }, [2, n, 1]);
    cap('emb_sum', emb, n * D);
    if (!decode) {
      condRows(b, emb, this.condBuf, [T('cond_embd.0.weight'), T('cond_embd.1.weight'), T('cond_embd.2.weight')], n, dyn);
      cap('emb_cond', emb, n * D);
    }
    const bank = b.buf(N_SRC * n * D * 4, 'bank');
    addRmsnorm(b, emb, bank, { N: D, rows: n, eps: RMS_EPS });
    steer(0);
    cap('emb_norm', bank, n * D);
    const aq = T('attnres_q.weight');
    const cur = b.buf(n * D * 4, 'cur');
    const qkvg = b.buf(n * QKVG * 4, 'qkvg');
    const q = b.buf(n * NH * HD * 4, 'q');
    const attnH = b.buf(n * NH * HD * 4, 'attnH');
    const mixed = b.buf(n * GDN_PROJ * 4, 'gdnMixed');
    const gs = {
      conv: b.buf(n * GDN_CONV * 4, 'gdnConv'), qn: b.buf(n * GDN_I * 4, 'gdnQ'), kn: b.buf(n * GDN_I * 4, 'gdnK'),
      decay: b.buf(n * GDN_H * 4, 'gdnDecay'), beta: b.buf(n * GDN_H * 4, 'gdnBeta'), o: b.buf(n * GDN_I * 4, 'gdnO'),
    };
    const gated = b.buf(n * GDN_I * 4, 'gdnGated');
    const ffA = decode ? null : b.buf(n * N_FF * 4, 'ffA');
    const ffG = decode ? null : b.buf(n * N_FF * 4, 'ffG');
    const ffH = b.buf(n * N_FF * 4, 'ffH');
    const rn = b.buf(n * D * 4, 'routerIn');
    const lg = b.buf(n * N_EXPERT * 4, 'routerLogits');
    const ms = { sel: b.buf(n * TOPK * 4, 'sel'), wts: b.buf(n * TOPK * 4, 'wts'), hbuf: b.buf(n * TOPK * N_EXP_FF * 4, 'hbuf') };
    const z = b.buf(n * N_LATENT * 4, 'z');
    const moeLat = b.buf(n * N_LATENT * 4, 'moeLat');

    for (let L = 0; L < N_LAYER; L++) {
      const p = `blk.${L}.`;
      attnres(b, bank, aq, T(p + 'attn_norm.weight'), cur, { D, n, ns: 2 * L + 1, qi: 2 * L, rows: n, eps: RMS_EPS });
      cap(`attn_in_${L}`, cur, n * D);
      if (isGlobal(L)) {
        const kc = st.kc.get(L)!, vc = st.vc.get(L)!;
        lin(cur, p + 'attn_qkvg.weight', qkvg, QKVG, D);
        ropeKv(b, qkvg, this.rope, q, kc, vc,
          { n, nh: NH, nkv: NKV, hd: HD, in_stride: QKVG, qkNorm: true, eps: QK_EPS, ring: CAP, posDyn: true },
          { temp: T(p + 'attn_temp'), dyn });
        attention(b, q, kc, vc, attnH,
          { n, nh: NH, nkv: NKV, hd: HD, window: 0xffffffff, ring: CAP, scale: ATTN_SCALE, posDyn: true,
            gate_stride: QKVG, gate_off: GATE_OFF },
          { gate: qkvg, dyn });
        const pl = PROBE_LAYERS.indexOf(L);
        if (probing && pl >= 0) {
          probeReadout(b, q, kc, this.probeBuf, this.probeSc, pl, po);
          attentionMasked(b, q, kc, vc, qkvg, this.probeBuf, attnH, pl,
            { nh: NH, nkv: NKV, hd: HD, ring: CAP, scale: ATTN_SCALE, gate_off: GATE_OFF }, dyn);
        }
        lin(attnH, p + 'attn_output.weight', bank, D, NH * HD, src(2 * L + 1));
      } else {
        const state = st.gdnState.get(L)!, hist = st.gdnHist.get(L)!;
        lin(cur, p + 'gdn_in.weight', mixed, GDN_PROJ, D);
        gdn(b, mixed, n, { conv: T(p + 'gdn_conv.weight'), dtBias: T(p + 'gdn_dt_bias'), aLog: T(p + 'gdn_A_log'), norm: T(p + 'gdn_norm.weight') },
          state, hist, gated, gs);
        if (caps) {
          cap(`gdn_conv_${L}`, gs.conv, n * GDN_CONV);
          cap(`gdn_q_${L}`, gs.qn, n * GDN_I);
          cap(`gdn_k_${L}`, gs.kn, n * GDN_I);
          cap(`gdn_decay_${L}`, gs.decay, n * GDN_H);
          cap(`gdn_beta_${L}`, gs.beta, n * GDN_H);
          cap(`gdn_recurrent_${L}`, gs.o, n * GDN_I);
          cap(`gdn_gated_${L}`, gated, n * GDN_I);
          cap(`gdn_state_${L}`, state, GDN_H * GDN_S * GDN_S);
          cap(`gdn_conv_state_${L}`, hist, GDN_CONV * 3);
        }
        lin(gated, p + 'attn_output.weight', bank, D, GDN_I, src(2 * L + 1));
      }
      steer(2 * L + 1);
      cap(`attn_out_${L}`, bank, n * D, src(2 * L + 1));
      attnres(b, bank, aq, T(p + 'ffn_norm.weight'), cur, { D, n, ns: 2 * L + 2, qi: 2 * L + 1, rows: n, eps: RMS_EPS });
      cap(`ffn_in_${L}`, cur, n * D);
      if (!w.has(p + 'router.weight')) {
        if (decode) {
          matvecGlu(b, cur, T(p + 'ffn_up.weight'), T(p + 'ffn_gate.weight'), ffH, N_FF, D);
        } else {
          gemm(b, cur, w.get(p + 'ffn_up.weight'), ffA!, { T: n, O: N_FF, IC: D });
          gemm(b, cur, w.get(p + 'ffn_gate.weight'), ffG!, { T: n, O: N_FF, IC: D });
          elementwise(b, EW.siluMul, n * N_FF, ffA!, ffH, {}, ffG!);
        }
        lin(ffH, p + 'ffn_down.weight', bank, D, N_FF, src(2 * L + 2));
      } else {
        addRmsnorm(b, cur, rn, { N: D, rows: n, eps: RMS_EPS }, { w: T(p + 'router_norm.weight') });
        lin(rn, p + 'router.weight', lg, N_EXPERT, D);
        cap(`router_logits_${L}`, lg, n * N_EXPERT);
        lin(cur, p + 'ffn_latent_down.weight', z, N_LATENT, D);
        moe(b, lg, T(p + 'router_bias'), z, w.get(p + 'ffn_up_exps.weight'), w.get(p + 'ffn_gate_exps.weight'),
          w.get(p + 'ffn_down_exps.weight'), moeLat, n, ms);
        lin(moeLat, p + 'ffn_latent_up.weight', bank, D, N_LATENT, src(2 * L + 2));
      }
      steer(2 * L + 2);
      cap(`ffn_out_${L}`, bank, n * D, src(2 * L + 2));
    }
    if (probing) probeAccumulate(b, this.probeBuf, this.probeSc, this.probeAcc, po);
    const hLast = b.buf(D * 4, 'hLast');
    attnres(b, bank, aq, T('output_norm.weight'), hLast, { D, n, ns: N_SRC, qi: 2 * N_LAYER, rows: 1, row0: n - 1, eps: RMS_EPS });
    const logits = b.buf(LOGITS * 4, 'logits');
    matvec(b, hLast, w.get('output.weight'), logits, { O: LOGITS, I: D, act: ACT.softcap, cap: SOFTCAP });
    return logits;
  }

  private clearRecurrent(enc: GPUCommandEncoder, slot: 0 | 1): void {
    for (const s of this.slots[slot].gdnState.values()) enc.clearBuffer(s);
    for (const h of this.slots[slot].gdnHist.values()) enc.clearBuffer(h);
  }

  /** Writes the conditioning header (and the speaker inputs); returns whether the speaker row must be projected. */
  private writeCond(c: Conditioning): boolean {
    const hdr = new Int32Array(COND_HDR);
    const speaker = c.speakerRow >= 0;
    hdr[0] = speaker ? c.speakerRow : -1;
    hdr[1] = c.conditionRow >= 0 ? c.conditionRow : -1;
    hdr[2] = N_COND;
    for (let i = 0; i < N_COND; i++) hdr[3 + i] = c.conditionRow >= 0 ? c.buckets[i] : 0;
    this.gpu.write(this.condBuf, 0, hdr);
    if (!speaker) return false;
    this.gpu.write(this.spkIn, 0, c.speaker!);
    this.gpu.write(this.spkDelta, 0, c.speakerDelta ?? new Float32Array(D));
    return true;
  }

  /** Runs the prompt of `slot` from position 0 in chunks (absolute positions, recurrent state carried across chunks),
   *  with the speaker/condition rows of `cond` replaced (zonos3_batch_slot_prefill). */
  async prefill(slot: 0 | 1, ids: Int32Array, cond: Conditioning, opts: PrefillOpts = {}): Promise<PrefillResult> {
    const n0 = ids.length / 11;
    if (!Number.isInteger(n0) || n0 < 1) throw new Error('empty prompt');
    if (n0 > CAP) throw new Error(`prompt has ${n0} rows; maximum is ${CAP}`);
    checkConditioning(ids, n0, cond, this.condNames);
    const gpu = this.gpu;
    const chunk = opts.capture ? n0 : Math.max(1, opts.chunk ?? CHUNK);
    const adhoc = new Map<number, { prog: Program; ids: GPUBuffer; logits: GPUBuffer; caps: Capture[] | null }>();
    const program = (len: number) => {
      if (len === CHUNK && !opts.capture) return { prog: this.chunkProg[slot], ids: this.chunkIds, logits: this.chunkLogits[slot], caps: null };
      let p = adhoc.get(len);
      if (!p) {
        const b = new Builder(gpu);
        const idsBuf = b.buf(len * 11 * 4, 'prefill.ids');
        const caps: Capture[] | null = opts.capture ? [] : null;
        const logits = this.record(b, len, false, idsBuf, caps, slot);
        p = { prog: b.finish(), ids: idsBuf, logits, caps };
        adhoc.set(len, p);
      }
      return p;
    };
    const project = this.writeCond(cond);
    this.nPast[slot] = 0;
    try {
      for (let start = 0; start < n0; start += chunk) {
        const len = Math.min(chunk, n0 - start);
        const last = start + len >= n0;
        const p = program(len);
        gpu.write(p.ids, 0, ids.subarray(start * 11, (start + len) * 11));
        this.dynData[0] = start;
        this.dynData[1] = 0;
        gpu.write(this.dyn, 0, this.dynData);
        const first = start === 0;
        const encode = (enc: GPUCommandEncoder) => {
          if (first) {
            this.clearRecurrent(enc, slot);
            if (project) this.spkProg.encode(enc);
          }
          p.prog.encode(enc);
        };
        if (!last) {
          const enc = gpu.device.createCommandEncoder();
          encode(enc);
          gpu.device.queue.submit([enc.finish()]);
          continue;
        }
        const caps = p.caps ?? [];
        const out = await gpu.submitRead(encode,
          [{ buf: p.logits, offset: 0, bytes: LOGITS * 4 }, ...caps.map(c => ({ buf: c.buf, offset: 0, bytes: c.floats * 4 }))]);
        this.nPast[slot] = n0;
        const res: PrefillResult = { logits: out[0] };
        if (p.caps) res.caps = new Map(p.caps.map((c, i) => [c.name, out[i + 1]]));
        return res;
      }
      throw new Error('unreachable');
    } finally {
      for (const p of adhoc.values()) p.prog.destroy();
    }
  }

  /** Writes probeBuf for this step (probe_ok + set_feature_inputs); returns the text length read back (0 = off). */
  private writeProbe(slot: 0 | 1, probe: ProbeInput | null): number {
    const words = new Uint32Array(PROBE_WORDS);
    const f = new Float32Array(words.buffer);
    const on = probe !== null && probe.heads.length > 0 && probe.textLen > 0;
    if (on) {
      if (slot !== 0) throw new Error('backbone: the alignment probe runs on slot 0 only');
      const n = this.nPast[slot];
      if (probe.textLen > PROBE_LMAX || probe.textStart < 0 || probe.textStart + probe.textLen > n)
        throw new Error(`zonos3: alignment probe text span [${probe.textStart}, +${probe.textLen}) must lie in the slot's ${n} cached rows and ` +
          `hold at most ${PROBE_LMAX} rows (zonos3_batch_features::probe_max_text)`);
      const at = (layer: number, head: number) => 4 + (PROBE_LAYERS.indexOf(layer) * NH + head) * 4;
      const headOk = (layer: number, head: number) => PROBE_LAYERS.includes(layer) && Number.isInteger(head) && head >= 0 && head < NH;
      for (const h of probe.heads) if (!headOk(h.layer, h.head))
        throw new Error(`zonos3: align probe head (${h.layer}, ${h.head}) is not an attention head of this model`);
      for (const win of probe.windows) if (!headOk(win.head.layer, win.head.head))
        throw new Error(`zonos3: align probe window head (${win.head.layer}, ${win.head.head}) is not an attention head of this model`);
      words[0] = probe.textStart;
      words[1] = probe.textLen;
      for (let l = 0; l < PROBE_LAYERS.length; l++)
        for (let h = 0; h < NH; h++) words[4 + (l * NH + h) * 4 + 2] = probe.textLen - 1;
      for (const h of probe.heads) f[at(h.layer, h.head)] = 1;
      for (const win of probe.windows) {
        const [lo, hi] = clipWindow(win.lo, win.hi, probe.textLen);
        const e = at(win.head.layer, win.head.head);
        words[e + 1] = lo;
        words[e + 2] = hi;
        words[e + 3] = 1;
      }
    }
    if (slot === 0) this.gpu.write(this.probeBuf, 0, words);
    return on ? probe.textLen : 0;
  }

  /** One decode row for `slot` at position nPast[slot]. `steer` = steerSources.length x 1024 hidden deltas (null = off).
   *  `probe` (slot 0): returns the summed readout distribution probeAcc[0..textLen). */
  async step(slot: 0 | 1, next: Int32Array, steer: Float32Array | null, probe: ProbeInput | null):
      Promise<{ logits: Float32Array; probe: Float32Array | null }> {
    if (this.nPast[slot] >= CAP) throw new Error('KV cache full');
    const K = this.steerSources.length;
    if (steer && steer.length !== K * D)
      throw new Error(`zonos3: slot steering delta must hold ${K} sources x ${D} values`);
    const textLen = this.writeProbe(slot, probe);
    this.gpu.write(this.ids, 0, next);
    if (steer && K) this.gpu.write(this.steerBuf, 0, steer);
    this.dynData[0] = this.nPast[slot];
    this.dynData[1] = steer && K ? 1 : 0;
    this.gpu.write(this.dyn, 0, this.dynData);
    const reads = [{ buf: this.decodeLogits[slot], offset: 0, bytes: LOGITS * 4 }];
    if (textLen) reads.push({ buf: this.probeAcc, offset: 0, bytes: textLen * 4 });
    const out = await this.gpu.submitRead(enc => this.decodeProg[slot].encode(enc), reads);
    this.nPast[slot]++;
    return { logits: out[0], probe: textLen ? out[1] : null };
  }
}

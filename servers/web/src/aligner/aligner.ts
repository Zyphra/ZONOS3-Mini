// FuseAlign word-level forced aligner on WebGPU: the frozen Dia byte-level text encoder, the FuseAlign joint
// transformer over [audio patches | transcript bytes], the two refinement heads (16x upsampling to a 1 ms grid), word
// mean pooling and the word MLPs, for one clip per forward (no padding, so every mask of the Python forward is a no-op).
// Mirrors third_party/fusealign-multilingual/ggml/src/model.cpp (Model::Impl::forward) on the q8_0 checkpoint exported
// by scripts/export_web_aligner.py (model directory `aligner`): q8 matrices for both transformers, f32 heads and MLPs,
// F32 activations, bf16 Dia RoPE tables and a bf16-rounded Dia output as the CPU reference.
import { fail } from '../api/http';
import type { Gpu } from '../gpu/device';
import { ACT, addRmsnorm, gemm, gemmF32 } from '../gpu/ops';
import { Builder, linearWg } from '../gpu/program';
import { loadModel, type Model, type Progress } from '../gpu/weights';
import {
  ALIGN_MAX_BYTES, ALIGN_MAX_SECONDS, assembleWords, diaInvFreq, faInvFreq, isWordy, jointPositions, ropeTables,
  SAMPLE_RATE, standardize, transcriptBytes, transcriptIds, type AlignedWord, type Interval,
} from './host';

export { ALIGN_MAX_BYTES, ALIGN_MAX_SECONDS };

interface Hp {
  dim: number; nl: number; nh: number; ffn: number; feat: number; win: number; hop: number; nup: number;
  ropeTheta: number; eps: number; threshold: number;
  D: number; dl: number; dnh: number; dhd: number; inter: number; diaEps: number; diaTheta: number;
}

function readHp(meta: Record<string, unknown>): Hp {
  const num = (k: string) => {
    const v = meta[k];
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`aligner: missing or invalid metadata ${k}`);
    return v;
  };
  if (meta['general.architecture'] !== 'fusealign' || meta['format'] !== 'fusealign-gguf/1')
    throw new Error('aligner: not a FuseAlign export (scripts/export_web_aligner.py)');
  const hp: Hp = {
    dim: num('dim'), nl: num('n_layers'), nh: num('n_heads'), ffn: num('ffn_dim'), feat: num('feat_dim'),
    win: num('patch_window'), hop: num('patch_hop'), nup: num('n_upsample'), ropeTheta: num('rope_theta'),
    eps: num('norm_eps'), threshold: num('presence_threshold'),
    D: num('dia_hidden'), dl: num('dia_layers'), dnh: num('dia_heads'), dhd: num('dia_head_dim'),
    inter: num('dia_intermediate'), diaEps: num('dia_norm_eps'), diaTheta: num('dia_rope_theta'),
  };
  const hd = hp.dim / hp.nh;
  if (hp.win !== 2 * hp.hop || hp.hop % 16 || hp.feat % 16 || (hd !== 64 && hd !== 128) ||
      (hp.dhd !== 64 && hp.dhd !== 128) || num('max_text_bytes') < ALIGN_MAX_BYTES || num('max_seconds') < ALIGN_MAX_SECONDS)
    throw new Error('aligner: unsupported FuseAlign hyperparameters');
  return hp;
}

type Buf = GPUBuffer;

/** In-place rotate_half RoPE of the q and k parts of `qkv` rows (fa_rope.wgsl). */
function rope(b: Builder, qkv: Buf, cos: Buf, sin: Buf, o: { n: number; nh: number; hd: number; stride: number; kOff: number }): void {
  b.dispatch('fa_rope', { qkv, cs: cos, sn: sin }, { n: o.n, nh: o.nh, hd: o.hd, stride: o.stride, q_off: 0, k_off: o.kOff },
    linearWg(Math.ceil(o.n * o.nh * (o.hd / 2) / 256)));
}

/** Full softmax attention over n rows of fused q|k|v (fa_attention.wgsl); out [n][nh * hd]. */
function attention(b: Builder, qkv: Buf, out: Buf, o: { n: number; nh: number; hd: number; scale: number }): void {
  const w = o.nh * o.hd;
  b.dispatch('fa_attention', { qkv, outp: out }, {
    n: o.n, nh: o.nh, hd: o.hd, stride: 3 * w, q_off: 0, k_off: w, v_off: 2 * w, out_stride: w, scale: o.scale,
  }, [Math.ceil(o.n / (256 / (o.hd / 16))), o.nh, 1]);
}

const EW_SWIGLU = 0, EW_BF16 = 1;
function ew(b: Builder, op: number, x: Buf, y: Buf, n: number, C = 1): void {
  b.dispatch('fa_ew', { x, y }, { op, n, C }, linearWg(Math.ceil(n / 256)));
}

export class Aligner {
  private queue: Promise<unknown> = Promise.resolve();
  private readonly faInv: Float32Array;
  private readonly diaInv: Float32Array;
  private readonly embedding: Float32Array;

  private constructor(private gpu: Gpu, private w: Model, private hp: Hp) {
    this.faInv = faInvFreq(hp.dim / hp.nh, hp.ropeTheta);
    this.diaInv = diaInvFreq(hp.dhd, hp.diaTheta);
    const e = w.host.get('dia.embedding');
    if (!e || e.length !== 256 * hp.D) throw new Error('aligner: missing dia.embedding');
    this.embedding = e;
  }

  /** Loads the model directory `aligner` (scripts/export_web_aligner.py). */
  static async load(gpu: Gpu, base: string, onProgress?: Progress, cache = true): Promise<Aligner> {
    const w = await loadModel(gpu, base, 'aligner', onProgress, cache);
    return new Aligner(gpu, w, readHp(w.meta));
  }

  /**
   * Word alignments of 16 kHz mono audio for `tokens` (zonos3.tts.align_text.align_tokens: non-empty, no whitespace),
   * one per token in order, as fusealign_hub's FuseAlignAligner.align: tokens with a letter or digit go to the model,
   * words under the presence threshold take the uniform fallback (fill_unplaced), punctuation-only tokens are
   * zero-width at the previous token's end. Over ALIGN_MAX_SECONDS of audio or ALIGN_MAX_BYTES of transcript: 400.
   */
  async align(pcm16k: Float32Array, tokens: string[]): Promise<AlignedWord[]> {
    for (const t of tokens)
      if (typeof t !== 'string' || !t || /\s/u.test(t)) fail(400, 'Alignment tokens must be non-empty strings without whitespace');
    const duration = pcm16k.length / SAMPLE_RATE;
    if (duration > ALIGN_MAX_SECONDS)
      fail(400, `Clip is ${duration.toFixed(2)} s; in-browser alignment covers at most ${ALIGN_MAX_SECONDS} s of audio`);
    const wordy: number[] = [];
    tokens.forEach((t, k) => { if (isWordy(t)) wordy.push(k); });
    const words = wordy.map(k => tokens[k]);
    const bytes = transcriptBytes(words);
    if (bytes > ALIGN_MAX_BYTES)
      fail(400, `Transcript is ${bytes} UTF-8 bytes; in-browser alignment covers at most ${ALIGN_MAX_BYTES} bytes`);
    const run = this.queue.then(async () => {
      let intervals: (Interval | null)[] = words.map(() => null);
      let presence = words.map(() => 0);
      if (words.length && pcm16k.length >= this.hp.win) ({ intervals, presence } = await this.forward(pcm16k, words));
      return assembleWords(tokens, wordy, intervals, presence, duration);
    });
    this.queue = run.catch(() => undefined);
    return run;
  }

  /** One forward (model.cpp Model::Impl::forward): per word its [start, end] (seconds) or null, and presence. */
  private async forward(pcm: Float32Array, words: string[]): Promise<{ intervals: (Interval | null)[]; presence: number[] }> {
    const { gpu, w, hp } = this;
    const { ids, spans } = transcriptIds(words);
    const T = ids.length - 2, L = T + 2, W = words.length;
    const A = Math.floor((pcm.length - hp.win) / hp.hop) + 1;
    const N = A + T;
    const { dim, nh, feat, D, dnh, dhd } = hp;
    const hd = dim / nh, HD = dnh * dhd;
    const b = new Builder(gpu);
    const f4 = (n: number) => n * 4;
    const upload = (data: Float32Array | Uint32Array) => {
      const buf = b.buf(data.byteLength);
      gpu.write(buf, 0, data);
      return buf;
    };

    // ── Dia text encoder ────────────────────────────────────────────────
    const x0 = new Float32Array(L * D);
    for (let i = 0; i < L; i++) x0.set(this.embedding.subarray(ids[i] * D, (ids[i] + 1) * D), i * D);
    const dx = upload(x0);
    const dTab = ropeTables(Float32Array.from({ length: L }, (_, i) => i), this.diaInv, true);
    const dCos = upload(dTab.cos), dSin = upload(dTab.sin);
    const dh = b.buf(f4(L * D)), dqkv = b.buf(f4(L * 3 * HD)), datt = b.buf(f4(L * HD));
    const dgu = b.buf(f4(L * 2 * hp.inter)), dact = b.buf(f4(L * hp.inter)), ddelta = b.buf(f4(L * D));
    for (let i = 0; i < hp.dl; i++) {
      addRmsnorm(b, dx, dh, { N: D, rows: L, eps: hp.diaEps }, { x: i ? ddelta : undefined, w: w.get(`dia.${i}.pre_norm`).buf });
      gemm(b, dh, w.get(`dia.${i}.qkv`), dqkv, { T: L, O: 3 * HD, IC: D });
      rope(b, dqkv, dCos, dSin, { n: L, nh: dnh, hd: dhd, stride: 3 * HD, kOff: HD });
      attention(b, dqkv, datt, { n: L, nh: dnh, hd: dhd, scale: 1 });   // DiaSelfAttention scaling 1
      gemm(b, datt, w.get(`dia.${i}.o`), ddelta, { T: L, O: D, IC: HD });
      addRmsnorm(b, dx, dh, { N: D, rows: L, eps: hp.diaEps }, { x: ddelta, w: w.get(`dia.${i}.post_norm`).buf });
      gemm(b, dh, w.get(`dia.${i}.gate_up`), dgu, { T: L, O: 2 * hp.inter, IC: D });
      ew(b, EW_SWIGLU, dgu, dact, L * hp.inter, hp.inter);
      gemm(b, dact, w.get(`dia.${i}.down`), ddelta, { T: L, O: D, IC: hp.inter });
      b.split();
    }
    addRmsnorm(b, dx, dh, { N: D, rows: L, eps: hp.diaEps }, { x: ddelta, w: w.get('dia.norm').buf });
    const textFeats = dqkv;   // reused: the encoder's bf16 output [L][D]
    ew(b, EW_BF16, dh, textFeats, L * D);

    // ── FuseAlign joint transformer ─────────────────────────────────────
    const std = standardize(pcm);
    const x = b.buf(f4(N * dim));
    const frames = upload(std.subarray(0, (A + 1) * hp.hop));
    // Conv1d(1, dim, 2 * hop, stride hop) as a K=2 gemm over hop-sample rows (+ mod_emb[0], folded into the bias).
    gemmF32(b, frames, w.get('fa.audio_patch'), x, { T: A, O: dim, IC: hp.hop, K: 2 }, { bias: w.get('fa.audio_bias').buf });
    gemm(b, textFeats, w.get('fa.text_proj'), x, { T, O: dim, IC: D, x_off: 2 * D, y_off: A * dim },
      { bias: w.get('fa.text_bias').buf });
    const fTab = ropeTables(jointPositions(A, T), this.faInv, false);
    const fCos = upload(fTab.cos), fSin = upload(fTab.sin);
    const h = b.buf(f4(N * dim)), qkv = b.buf(f4(N * 3 * dim)), att = b.buf(f4(N * dim));
    const w13 = b.buf(f4(N * 2 * hp.ffn)), act = b.buf(f4(N * hp.ffn)), delta = b.buf(f4(N * dim));
    for (let i = 0; i < hp.nl; i++) {
      addRmsnorm(b, x, h, { N: dim, rows: N, eps: hp.eps }, { x: i ? delta : undefined, w: w.get(`fa.${i}.attn_norm`).buf });
      gemm(b, h, w.get(`fa.${i}.qkv`), qkv, { T: N, O: 3 * dim, IC: dim });
      rope(b, qkv, fCos, fSin, { n: N, nh, hd, stride: 3 * dim, kOff: dim });
      attention(b, qkv, att, { n: N, nh, hd, scale: 1 / Math.sqrt(hd) });
      gemm(b, att, w.get(`fa.${i}.out`), delta, { T: N, O: dim, IC: dim });
      addRmsnorm(b, x, h, { N: dim, rows: N, eps: hp.eps }, { x: delta, w: w.get(`fa.${i}.ffn_norm`).buf });
      gemm(b, h, w.get(`fa.${i}.w13`), w13, { T: N, O: 2 * hp.ffn, IC: dim });
      ew(b, EW_SWIGLU, w13, act, N * hp.ffn, hp.ffn);
      gemm(b, act, w.get(`fa.${i}.w2`), delta, { T: N, O: dim, IC: hp.ffn });
      b.split();
    }
    const out = h;   // final_norm(x) [N][dim]: audio rows, then text rows
    addRmsnorm(b, x, out, { N: dim, rows: N, eps: hp.eps }, { x: delta, w: w.get('fa.final_norm').buf });

    // ── Word pooling and MLPs ───────────────────────────────────────────
    const wv = b.buf(f4(W * dim));
    b.dispatch('fa_pool', { x: out, spans: upload(spans), y: wv }, { dim, row0: A }, [W, 1, 1]);
    const hm = b.buf(f4(W * 3 * dim));
    gemmF32(b, wv, w.get('fa.mlp0'), hm, { T: W, O: 3 * dim, IC: dim, act: ACT.gelu }, { bias: w.get('fa.mlp0_b').buf });
    const svec = b.buf(f4(W * feat)), evec = b.buf(f4(W * feat));
    const result = b.buf(f4(3 * W));   // start ms | end ms | presence logit
    gemmF32(b, hm, w.get('fa.start_mlp2'), svec, { T: W, O: feat, IC: dim, x_stride: 3 * dim }, { bias: w.get('fa.start_mlp2_b').buf });
    gemmF32(b, hm, w.get('fa.end_mlp2'), evec, { T: W, O: feat, IC: dim, x_stride: 3 * dim, x_off: dim }, { bias: w.get('fa.end_mlp2_b').buf });
    gemmF32(b, hm, w.get('fa.presence_mlp2'), result, { T: W, O: 1, IC: dim, x_stride: 3 * dim, x_off: 2 * dim, y_off: 2 * W },
      { bias: w.get('fa.presence_mlp2_b').buf });

    // ── Refinement heads, logits over the 1 ms grid, argmax ─────────────
    const F = A << hp.nup;
    // Ping-pong buffers with zero pad rows: one each side for the transposed convs, two for the kernel-5 conv.
    const padA = b.buf(f4((F + 4) * feat)), padB = b.buf(f4((F + 4) * feat));
    const hs = b.buf(f4(F * feat)), hf = b.buf(f4(F * feat)), lg = b.buf(f4(W * F));
    b.split();
    for (const [name, vec, off] of [['start', svec, 0], ['end', evec, W]] as const) {
      let n = A, src = padA, dst = padB;
      b.clear(src, 0, f4(feat));
      b.clear(src, f4((n + 1) * feat), f4(feat));
      gemmF32(b, out, w.get(`fa.${name}.in`), src, { T: A, O: feat, IC: dim, y_off: feat }, { bias: w.get(`fa.${name}.in_b`).buf });
      for (let s = 0; s < hp.nup; s++) {
        // ConvTranspose1d(k4, s2, p1): y[2m] = W3.h[m-1] + W1.h[m], y[2m+1] = W2.h[m] + W0.h[m+1]; then GELU.
        const last = s === hp.nup - 1, pad = last ? 2 : 1;
        b.clear(dst, 0, f4(pad * feat));
        b.clear(dst, f4((2 * n + pad) * feat), f4(pad * feat));
        const bias = w.get(`fa.${name}.up${s}.b`).buf;
        const common = { T: n, O: feat, IC: feat, K: 2, y_stride: 2 * feat, act: ACT.gelu };
        gemmF32(b, src, w.get(`fa.${name}.up${s}.even`), dst, { ...common, y_off: pad * feat }, { bias });
        gemmF32(b, src, w.get(`fa.${name}.up${s}.odd`), dst, { ...common, x_off: feat, y_off: (pad + 1) * feat }, { bias });
        n *= 2;
        [src, dst] = [dst, src];
      }
      gemmF32(b, src, w.get(`fa.${name}.smooth`), hs, { T: F, O: feat, IC: feat, K: 5, act: ACT.gelu },
        { bias: w.get(`fa.${name}.smooth_b`).buf });
      gemmF32(b, hs, w.get(`fa.${name}.out`), hf, { T: F, O: feat, IC: feat }, { bias: w.get(`fa.${name}.out_b`).buf });
      gemmF32(b, vec, { buf: hf, dtype: 'f32' }, lg, { T: W, O: F, IC: feat });
      b.dispatch('fa_argmax', { lg, outp: result }, { F, out_off: off }, [W, 1, 1]);
      b.split();
    }

    const prog = b.finish();
    try {
      const [res] = await prog.run(gpu, [{ buf: result, offset: 0, bytes: f4(3 * W) }]);
      const intervals: (Interval | null)[] = [];
      const presence: number[] = [];
      for (let i = 0; i < W; i++) {
        let p = 1 / (1 + Math.exp(-res[2 * W + i]));
        if (Number.isNaN(p)) p = 0;
        presence.push(p);
        const s = res[i], e = Math.max(res[W + i], s);
        intervals.push(p >= hp.threshold ? [s / 1000, e / 1000] : null);
      }
      return { intervals, presence };
    } finally {
      prog.destroy();
    }
  }
}

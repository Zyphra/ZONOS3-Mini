// Backbone parity vs zonos3-cli --cpu goldens: conditioned prefill, teacher-forced decode, alignment probe, CFG-slot
// isolation, hidden steering and masked probe heads; plus the speech-edit backbone (model 'edit') on an edit prompt.
import { Backbone, loadBackbone, NCB, type ProbeInput } from '../../src/backbone';
import { fetchJson, loadModel } from '../../src/gpu/weights';
import { locateTextSpan, probeReduce } from '../../src/probe';
import { conditionBucketCounts, type Conditioning } from '../../src/prompt';
import { argmax, cosine, maxAbs, type Check, type Ctx } from './util';

const W = 11;
/** Steering sources of the shipped steer.gguf (its hidden sets). */
const STEER_SOURCES = [15, 19];

interface CondJson { speaker_row: number; condition_row: number; condition_buckets: number[]; speaker_embedding: number[] }

interface Fixture {
  bb: Backbone;
  ids: Int32Array;           // prompt [n0][11]
  full: Int32Array;          // prompt + generated rows [69][11]
  n0: number;
  cond: Conditioning;
  unknown: number[];         // per-table unknown bucket ids (the bucket counts)
  logits: Float32Array;      // [50][4098 * 10]: row 0 prompt, row k decode step k
  probe: Float32Array;       // [49][2]
  probeIn: ProbeInput;
}

let fixture: Promise<Fixture> | null = null;

function load(ctx: Ctx): Promise<Fixture> {
  return fixture ??= (async () => {
    const idsNpy = await ctx.npy('ids.npy');
    const ids = Int32Array.from(idsNpy.data, Math.round);
    const full = Int32Array.from((await ctx.npy('full_ids.npy')).data, Math.round);
    const cj = JSON.parse(new TextDecoder().decode(await ctx.golden('ids.cond.json'))) as CondJson;
    const cond: Conditioning = {
      speakerRow: cj.speaker_row, speaker: Float32Array.from(cj.speaker_embedding), conditionRow: cj.condition_row,
      buckets: cj.condition_buckets,
    };
    const heads = (await fetchJson<{ heads: { layer: number; head: number }[] }>(new URL('laci/zonos3_mini_mect60k_heads.json', ctx.base).href)).heads;
    const span = locateTextSpan(ids);
    if (!span) throw new Error('ids.npy has no BOS..EOS text span');
    const model = await loadModel(ctx.gpu, ctx.base, 'backbone');
    const bb = new Backbone(ctx.gpu, model, STEER_SOURCES);
    return {
      bb, ids, full, n0: idsNpy.shape[0], cond, unknown: conditionBucketCounts(model.meta),
      logits: (await ctx.npy('logits.npy')).data, probe: (await ctx.npy('probe.npy')).data,
      probeIn: { textStart: span.start, textLen: span.len, heads: heads.map(h => ({ layer: h.layer, head: h.head })), windows: [] },
    };
  })();
}

const LOGITS = 4098 * NCB;
const row = (f: Fixture, r: number) => f.full.subarray(r * W, (r + 1) * W);
const want = (f: Fixture, k: number) => f.logits.subarray(k * LOGITS, (k + 1) * LOGITS);

/** Codebooks whose argmax agrees. */
function argmaxAgree(a: Float32Array, b: Float32Array, V = 4098): number {
  let n = 0;
  for (let cb = 0; cb < NCB; cb++) if (argmax(a, cb * V, V) === argmax(b, cb * V, V)) n++;
  return n;
}

interface DecodeRun { logits: Float32Array[]; probe: Float32Array[] }
let decodeRun: Promise<DecodeRun> | null = null;

/** Prefill full_ids[0..n0) then teacher-force rows n0..67 with the probe on (49 steps). */
function decode(f: Fixture): Promise<DecodeRun> {
  return decodeRun ??= (async () => {
    await f.bb.prefill(0, f.full.subarray(0, f.n0 * W), f.cond);
    const steps = f.logits.length / LOGITS - 1;
    const out: DecodeRun = { logits: [], probe: [] };
    for (let k = 1; k <= steps; k++) {
      const r = await f.bb.step(0, row(f, f.n0 + k - 1), null, f.probeIn);
      out.logits.push(r.logits);
      out.probe.push(r.probe!);
    }
    return out;
  })();
}

export const checks: Record<string, Check> = {
  async backbone_prefill(ctx) {
    const f = await load(ctx);
    const r = await f.bb.prefill(0, f.ids, f.cond);
    const cos = cosine(r.logits, want(f, 0));
    const agree = argmaxAgree(r.logits, want(f, 0));
    const rc = await f.bb.prefill(0, f.ids, f.cond, { capture: true });
    const embCond = (await ctx.npy('validate/emb_cond.npy')).data.subarray(0, f.n0 * 1024);
    const embSum = (await ctx.npy('validate/emb_sum.npy')).data.subarray(0, f.n0 * 1024);
    const gotCond = rc.caps!.get('emb_cond')!;
    const gotSum = rc.caps!.get('emb_sum')!;
    let minRowCos = 1;
    for (let t = 0; t < f.n0; t++)
      minRowCos = Math.min(minRowCos, cosine(gotCond.subarray(t * 1024, (t + 1) * 1024), embCond.subarray(t * 1024, (t + 1) * 1024)));
    const capCos = cosine(rc.logits, want(f, 0));
    return {
      ok: cos >= 0.995 && agree >= 9 && minRowCos >= 0.999 && capCos >= 0.995,
      metric: { cosine: cos, argmax: `${agree}/10`, embCondMinRowCos: minRowCos, embCondMaxAbs: maxAbs(gotCond, embCond),
        embSumCos: cosine(gotSum, embSum), captureLogitsCos: capCos },
    };
  },

  async backbone_decode(ctx) {
    const f = await load(ctx);
    const run = await decode(f);
    // Argmax bar vs ggml Metal (validate_metal: full-sequence logits, decode step k = row n0 + k - 1): the CPU golden's
    // NEON fp16-accumulated F16 dots flip near-tie MoE expert picks, so ggml Metal itself agrees with it on ~94.5%.
    const metal = (await ctx.npy('validate_metal/logits.npy')).data;
    let minCos = 1, agreeCpu = 0, agreeMetal = 0, minCosMetal = 1;
    run.logits.forEach((lg, i) => {
      const m = metal.subarray((f.n0 + i) * LOGITS, (f.n0 + i + 1) * LOGITS);
      minCos = Math.min(minCos, cosine(lg, want(f, i + 1)));
      minCosMetal = Math.min(minCosMetal, cosine(lg, m));
      agreeCpu += argmaxAgree(lg, want(f, i + 1));
      agreeMetal += argmaxAgree(lg, m);
    });
    const total = run.logits.length * NCB;
    return {
      ok: minCos >= 0.99 && agreeMetal / total >= 0.95,
      metric: { steps: run.logits.length, minCos, argmaxCpu: agreeCpu / total, minCosMetal, argmaxMetal: agreeMetal / total },
    };
  },

  async backbone_probe(ctx) {
    const f = await load(ctx);
    const run = await decode(f);
    let agree = 0, maxCentroid = 0;
    run.probe.forEach((acc, k) => {
      const r = probeReduce(acc, f.probeIn.textLen);
      if (r.argmax === Math.round(f.probe[k * 2])) agree++;
      maxCentroid = Math.max(maxCentroid, Math.abs(r.centroid - f.probe[k * 2 + 1]));
    });
    const steps = f.probe.length / 2;
    return {
      ok: run.probe.length === steps && agree / steps >= 0.95,
      metric: { steps, argmaxAgree: agree / steps, maxCentroidDiff: maxCentroid, textStart: f.probeIn.textStart, textLen: f.probeIn.textLen },
    };
  },

  async backbone_twin_and_steer(ctx) {
    const f = await load(ctx);
    const ref = await decode(f);
    // (a) slot 1 holds the CFG twin (speaker row dropped, unknown buckets) and steps with slot 0 in one submission.
    const twinIds = f.ids.slice(W);
    const twinCond: Conditioning = { speakerRow: -1, speaker: null, conditionRow: 0, buckets: f.unknown.slice() };
    await f.bb.prefill(0, f.ids, f.cond);
    const tw = await f.bb.prefill(1, twinIds, twinCond);
    const n = 8;
    let isoMax = 0;
    for (let k = 1; k <= n; k++) {
      const r0 = await f.bb.stepPair(row(f, f.n0 + k - 1), null, null, f.probeIn);
      isoMax = Math.max(isoMax, maxAbs(r0.logits, ref.logits[k - 1]));
    }
    const twinDiffers = 1 - cosine(tw.logits, want(f, 0)) > 1e-4;
    // (b) one step after a fresh prefill: unsteered, zero steer, nonzero steer.
    const K = STEER_SOURCES.length;
    const once = async (steer: Float32Array | null) => {
      await f.bb.prefill(0, f.ids, f.cond);
      return (await f.bb.step(0, row(f, f.n0), steer, null)).logits;
    };
    const plain = await once(null);
    const zero = await once(new Float32Array(K * 1024));
    const delta = new Float32Array(K * 1024);
    for (let i = 0; i < delta.length; i++) delta[i] = Math.sin(i * 0.37) * 0.5;
    const steered = await once(delta);
    const zeroMax = maxAbs(plain, zero);
    const steerCos = cosine(plain, steered);
    return {
      ok: isoMax <= 1e-5 && twinDiffers && zeroMax <= 1e-5 && steerCos < 0.9999,
      metric: { isolationMaxAbs: isoMax, twinVsCondCos: cosine(tw.logits, want(f, 0)), zeroSteerMaxAbs: zeroMax, steeredCos: steerCos },
    };
  },

  async backbone_masked(ctx) {
    const f = await load(ctx);
    const ref = await decode(f);
    const once = async (windows: ProbeInput['windows']) => {
      await f.bb.prefill(0, f.ids, f.cond);
      return (await f.bb.step(0, row(f, f.n0), null, { ...f.probeIn, windows })).logits;
    };
    // Full span on one head = no blocked key: identical attention.
    const full = await once([{ head: f.probeIn.heads[0], lo: 0, hi: f.probeIn.textLen - 1 }]);
    // A one-key window on every probe head blocks most text keys of those heads.
    const narrow = await once(f.probeIn.heads.map(head => ({ head, lo: 2, hi: 2 })));
    const fullCos = cosine(full, ref.logits[0]);
    const narrowMax = maxAbs(narrow, ref.logits[0]);
    return {
      ok: fullCos >= 0.9999 && narrowMax > 1e-2 && narrow.every(Number.isFinite),
      metric: { fullWindowCos: fullCos, fullWindowMaxAbs: maxAbs(full, ref.logits[0]), narrowWindowCos: cosine(narrow, ref.logits[0]),
        narrowWindowMaxAbs: narrowMax },
    };
  },

  /** The speech-edit backbone (model 'edit', adapter merged at export) vs MLX with the adapter applied at runtime
   *  (scripts/check_edit_export.py --goldens): an edit prompt prefill, then teacher-forced fill rows through span-end.
   *  The bar is the export's own deviation from the runtime adapter (MLX on the exported tensors: min cos 0.998,
   *  argmax 95 % on these rows) plus the WebGPU kernels' rounding. */
  async backbone_edit(ctx) {
    const ids = Int32Array.from((await ctx.npy('edit/ids.npy')).data, Math.round);
    const ref = (await ctx.npy('edit/logits.npy')).data;
    const cj = JSON.parse(new TextDecoder().decode(await ctx.golden('edit/cond.json'))) as CondJson & { n0: number };
    const bb = await loadBackbone(ctx.gpu, ctx.base, 'edit');
    const V = bb.hp.audioVocab, L = bb.logitsLen;
    const cond: Conditioning = {
      speakerRow: cj.speaker_row, speaker: Float32Array.from(cj.speaker_embedding), conditionRow: cj.condition_row,
      buckets: cj.condition_buckets,
    };
    const rows = [(await bb.prefill(0, ids.subarray(0, cj.n0 * W), cond)).logits];
    for (let r = cj.n0; r < ids.length / W; r++) rows.push((await bb.step(0, ids.subarray(r * W, (r + 1) * W), null, null)).logits);
    let minCos = 1, agree = 0;
    rows.forEach((lg, k) => {
      const want = ref.subarray(k * L, (k + 1) * L);
      for (let cb = 0; cb < NCB; cb++)
        minCos = Math.min(minCos, cosine(lg.subarray(cb * V, (cb + 1) * V), want.subarray(cb * V, (cb + 1) * V)));
      agree += argmaxAgree(lg, want, V);
    });
    const total = rows.length * NCB;
    return {
      ok: V === 4099 && bb.hp.spanEndId === 4098 && rows.length * L === ref.length && minCos >= 0.99 && agree / total >= 0.9,
      metric: { rows: rows.length, audioVocab: V, spanEndId: bb.hp.spanEndId, textExtraTokens: bb.hp.textExtraTokens,
        minCodebookCos: minCos, argmax: agree / total },
    };
  },
};

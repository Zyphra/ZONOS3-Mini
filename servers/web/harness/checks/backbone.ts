// Backbone parity vs zonos3-cli --cpu goldens: conditioned prefill, teacher-forced decode, alignment probe, CFG-slot
// isolation, hidden steering and masked probe heads; plus the speech-edit LM (the edit/ adapter on the backbone's
// weights, Backbone.withEdit) on an edit prompt, and the GPU sampler (sample.wgsl) against its host oracle (Sampler).
import { Backbone, NCB, type ProbeInput } from '../../src/backbone';
import { GpuSampler, type SamplingSetup } from '../../src/gpu/sampler';
import { fetchJson, loadModel } from '../../src/gpu/weights';
import { MT19937 } from '../../src/mt19937';
import { locateTextSpan, probeReduce } from '../../src/probe';
import { conditionBucketCounts, TEXT_PAD, type Conditioning } from '../../src/prompt';
import { Sampler, type Sampling } from '../../src/sampler';
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

/** The host oracle of one GpuSampler draw: Sampler over `cond` (CFG: twin + s * (cond - twin) in f32 ops), cold start
 *  past `row`, the text pad. */
function hostDraw(s: SamplingSetup, row: number, cond: Float32Array, twin: Float32Array): Int32Array {
  const V = cond.length / NCB;
  const smp = new Sampler({ ...s.sp }, NCB, V, s.vocab ?? {});
  smp.minFrames = s.minFrames;
  for (const h of s.history ?? []) smp.accept(h);
  smp.generated = 0;
  let src = cond;
  if (s.cfgScale !== null) {
    const f = Math.fround, sc = f(s.cfgScale);
    src = cond.map((c, i) => f(twin[i] + f(sc * f(c - twin[i]))));
  }
  const out = new Int32Array(NCB + 1);
  for (let cb = 0; cb < NCB; cb++) out[cb] = cb > row ? smp.padId : smp.sample(src, cb);
  out[NCB] = TEXT_PAD;
  return out;
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

  /** The speech-edit LM (backbone/ weights + the edit/ adapter at runtime) vs MLX with the adapter applied at runtime
   *  (scripts/check_edit_export.py --goldens): an edit prompt prefill, then teacher-forced fill rows through span-end. */
  async backbone_edit(ctx) {
    const ids = Int32Array.from((await ctx.npy('edit/ids.npy')).data, Math.round);
    const ref = (await ctx.npy('edit/logits.npy')).data;
    const cj = JSON.parse(new TextDecoder().decode(await ctx.golden('edit/cond.json'))) as CondJson & { n0: number };
    const f = await load(ctx);
    const bb = f.bb.withEdit(await loadModel(ctx.gpu, ctx.base, 'edit'));
    const V = bb.hp.audioVocab, L = bb.logitsLen;
    const cond: Conditioning = {
      speakerRow: cj.speaker_row, speaker: Float32Array.from(cj.speaker_embedding), conditionRow: cj.condition_row,
      buckets: cj.condition_buckets,
    };
    const rows = [(await bb.prefill(0, ids.subarray(0, cj.n0 * W), cond)).logits];
    for (let r = cj.n0; r < ids.length / W; r++) rows.push((await bb.step(0, ids.subarray(r * W, (r + 1) * W), null, null)).logits);
    let minCos = 1, agree = 0, maxDiff = 0;
    rows.forEach((lg, k) => {
      const want = ref.subarray(k * L, (k + 1) * L);
      for (let cb = 0; cb < NCB; cb++)
        minCos = Math.min(minCos, cosine(lg.subarray(cb * V, (cb + 1) * V), want.subarray(cb * V, (cb + 1) * V)));
      agree += argmaxAgree(lg, want, V);
      maxDiff = Math.max(maxDiff, maxAbs(lg, want));
    });
    const total = rows.length * NCB;
    return {
      ok: V === 4099 && bb.hp.spanEndId === 4098 && rows.length * L === ref.length && minCos >= 0.999 && agree / total >= 0.98,
      metric: { rows: rows.length, audioVocab: V, spanEndId: bb.hp.spanEndId, textExtraTokens: bb.hp.textExtraTokens,
        minCodebookCos: minCos, argmax: agree / total, maxAbs: maxDiff },
    };
  },

  /** sample.wgsl vs Sampler on fixed random logits (vocabulary 4098, 10 codebooks): greedy draws (repetition penalty,
   *  CFG, the stop gate, cold start) must match exactly; sampled draws (2000 seeds per setting, the same MT19937
   *  stream on both sides) must give per-codebook histograms within total variation 0.05. Then the backbone: the first
   *  greedy frame from the prefill's logits must match the host sampler, and stepSample calls submitted back to back
   *  (lookahead) must give the frames of awaited ones. */
  async 'sampler.gpu'(ctx) {
    const V = 4098, L = V * NCB, gpu = ctx.gpu;
    const rng = new MT19937(1234);
    const unif = () => (rng.next() + 0.5) / 4294967296;
    const gauss = () => Math.sqrt(-2 * Math.log(unif())) * Math.cos(2 * Math.PI * unif());
    // ids < 300 lead (the repetition history draws from ids < 200), EOA at 4096 is a strong candidate
    const cond = new Float32Array(L), twin = new Float32Array(L);
    for (let i = 0; i < L; i++) {
      cond[i] = 1.5 * gauss() + (i % V < 300 ? 3 : 0) + (i % V === 4096 ? 6 : 0);
      twin[i] = cond[i] + 0.7 * gauss();
    }
    const history = Array.from({ length: 120 }, () => Array.from({ length: NCB }, () => Math.floor(unif() * 200)));
    const ids = gpu.buffer((NCB + 1) * 4, 'smp.ids');
    const gs = new GpuSampler(gpu, V, NCB, gpu.upload(cond, 'smp.cond'), gpu.upload(twin, 'smp.twin'), ids);
    const draw = async (s: SamplingSetup, row: number) => {
      gs.start(s);
      const p = gs.next(row);
      const out = await gpu.submitRead(enc => p.encode(enc), [{ buf: ids, offset: 0, bytes: (NCB + 1) * 4 }]);
      return new Int32Array(out[0].buffer);
    };
    const base: Sampling = { temperature: 1, top_k: 0, top_p: 1, min_p: 0, rep_penalty: 1, rep_window: 96, rep_codebooks: NCB, seed: 0, greedy: false };
    const greedy: Sampling = { ...base, temperature: 0, greedy: true };
    const greedyCases: [string, SamplingSetup, number][] = [
      ['plain', { sp: greedy, minFrames: 3, cfgScale: null }, 9],
      ['stop', { sp: greedy, minFrames: 0, cfgScale: null }, 9],
      ['penalty', { sp: { ...greedy, rep_penalty: 2.4, rep_codebooks: 6 }, minFrames: 3, cfgScale: null, history }, 9],
      ['cfg', { sp: { ...greedy, rep_penalty: 2.4 }, minFrames: 3, cfgScale: 1.5, history }, 9],
      ['coldStart', { sp: greedy, minFrames: 0, cfgScale: null }, 3],
    ];
    const greedyBad: string[] = [];
    for (const [name, s, row] of greedyCases) {
      const g = await draw(s, row), h = hostDraw(s, row, cond, twin);
      if (g.some((v, i) => v !== h[i])) greedyBad.push(`${name}: gpu ${Array.from(g)} host ${Array.from(h)}`);
    }
    const DRAWS = 2000;
    const settings: [string, Partial<Sampling>, boolean][] = [
      ['topk100', { top_k: 100 }, false], ['topp0.69', { top_p: 0.69 }, false], ['minp0.05', { min_p: 0.05 }, false],
      ['penalty2.4', { rep_penalty: 2.4 }, true], ['defaults+cfg', { top_k: 100, rep_penalty: 2.4, rep_codebooks: 6 }, true],
    ];
    const tv: Record<string, number> = {}, same: Record<string, number> = {};
    let maxTv = 0;
    for (const [name, o, hist] of settings) {
      const hg = new Uint32Array(L), hh = new Uint32Array(L);
      let agree = 0;
      for (let d = 0; d < DRAWS; d++) {
        const s: SamplingSetup = {
          sp: { ...base, ...o, seed: d }, minFrames: 0, cfgScale: name.endsWith('cfg') ? 1.5 : null, history: hist ? history : undefined,
        };
        const g = await draw(s, 9), h = hostDraw(s, 9, cond, twin);
        for (let cb = 0; cb < NCB; cb++) {
          hg[cb * V + g[cb]]++;
          hh[cb * V + h[cb]]++;
          if (g[cb] === h[cb]) agree++;
        }
      }
      let worst = 0;
      for (let cb = 0; cb < NCB; cb++) {
        let d = 0;
        for (let i = 0; i < V; i++) d += Math.abs(hg[cb * V + i] - hh[cb * V + i]);
        worst = Math.max(worst, d / (2 * DRAWS));
      }
      tv[name] = worst;
      same[name] = agree / (DRAWS * NCB);
      maxTv = Math.max(maxTv, worst);
    }

    // the backbone: prefill logits -> first frame, then 12 greedy steps awaited one by one vs submitted back to back
    const f = await load(ctx);
    const steps = 12;
    const run = async (lookahead: boolean) => {
      const pre = await f.bb.prefill(0, f.ids, f.cond);
      const s: SamplingSetup = { sp: { ...greedy, rep_penalty: 2.4, rep_codebooks: 6 }, minFrames: 0, cfgScale: null };
      f.bb.startSampling(s);
      const first = await f.bb.sampleFirst(9);
      const frames = [first];
      if (lookahead) {
        const pending = Array.from({ length: steps }, (_, k) => f.bb.stepSample(10 + k, null, null, null));
        for (const r of await Promise.all(pending)) frames.push(r.frame);
      } else {
        for (let k = 0; k < steps; k++) frames.push((await f.bb.stepSample(10 + k, null, null, null)).frame);
      }
      return { frames, host: hostDraw(s, 9, pre.logits, pre.logits) };
    };
    const serial = await run(false), ahead = await run(true);
    const firstOk = serial.frames[0].every((v, i) => v === serial.host[i]);
    const lookOk = serial.frames.every((fr, k) => fr.every((v, i) => v === ahead.frames[k][i]));
    return {
      ok: greedyBad.length === 0 && maxTv < 0.05 && firstOk && lookOk,
      metric: { greedyMismatch: greedyBad, tv, sameDraw: same, firstFrameMatchesHost: firstOk, lookaheadMatchesSerial: lookOk },
    };
  },
};

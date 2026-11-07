// Codec parity: ZacStream on goldens/tts.npy with the server's streaming block schedule vs zac-cli (zac_audio.npy);
// the same schedule with the host half in the codec worker (RemoteSpecSink) must match the local sink bit for bit.
import { HOP } from '../../src/generation';
import { ZacStream } from '../../src/zac';
import type { GemmPlan } from '../../src/gpu/h8';
import { loadModel } from '../../src/gpu/weights';
import { buildIdx, finalBlocks, readyBlocks, type Range } from '../../src/schedule';
import { RemoteSpecSink, type HostParams, type SpecSink } from '../../src/spec-sink';
import { cosine, maxAbs, type Check, type Ctx } from './util';

interface Run { got: Float32Array; blocks: number; ms: number; plan: GemmPlan }

/** The gained post-prefill audio of goldens/tts.npy, decoded block by block as the server streams it (gemm kernels
 *  timed on this device unless `plan` is given). */
async function run(ctx: Ctx, makeSink?: (p: HostParams) => SpecSink, plan?: GemmPlan): Promise<Run> {
  const codesNpy = await ctx.npy('tts.npy');
  const [eos, prefill] = Array.from((await ctx.npy('tts.npy.eos.npy')).data, Math.round);
  const H = codesNpy.shape[0];
  const codes = Int32Array.from(codesNpy.data, Math.round);
  const zac = await ZacStream.create(ctx.gpu, await loadModel(ctx.gpu, ctx.base, 'zac'), makeSink, plan);
  zac.reset();

  // Rows arrive one per decode step; EOS (codebook 0 at row `eos`) is known once that row exists.
  const blocks: Range[] = [];
  let emitted = 0;
  for (let h = 1; h <= H; h++) {
    const r = readyBlocks(h, emitted, h > eos ? eos : -1);
    blocks.push(...r.blocks);
    emitted = r.emitted;
  }
  blocks.push(...finalBlocks(H, emitted, eos));

  const parts: Float32Array[] = [];
  const t0 = performance.now();
  for (const [p0, p1] of blocks) parts.push(await zac.chunk(buildIdx(codes, H, p0, p1 - p0), p0, p1 - p0));
  parts.push(await zac.flush());
  const ms = performance.now() - t0;
  const all = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) { all.set(p, o); o += p.length; }
  return {
    got: all.slice(Math.min(all.length, prefill * HOP)).map(v => v * zac.gain), blocks: blocks.length, ms, plan: zac.gemmPlan,
  };
}

export const checks: Record<string, Check> = {
  async zac_stream(ctx) {
    const want = (await ctx.npy('zac_audio.npy')).data;
    const { got, blocks, ms, plan } = await run(ctx);
    const n = Math.min(got.length, want.length);
    const cos = cosine(got.subarray(0, n), want.subarray(0, n));
    return {
      ok: got.length === want.length && cos >= 0.999,
      metric: { cosine: cos, maxAbs: maxAbs(got.subarray(0, n), want.subarray(0, n)), samples: got.length, want: want.length,
        blocks, ms: Math.round(ms), gemm: `${plan.skinny}/${plan.large}` },
    };
  },

  async zac_stream_worker(ctx) {
    const local = await run(ctx);
    const worker = new Worker(new URL('../../src/codec-worker.ts', import.meta.url), { type: 'module' });
    try {
      const ch = new MessageChannel();
      worker.postMessage({ type: 'port' }, [ch.port2]);
      const remote = await run(ctx, p => new RemoteSpecSink(ch.port1, p), local.plan);
      const same = remote.got.length === local.got.length;
      const diff = same ? maxAbs(remote.got, local.got) : Infinity;
      return {
        ok: same && diff === 0,
        metric: { maxAbs: diff, samples: remote.got.length, localSamples: local.got.length, ms: Math.round(remote.ms),
          localMs: Math.round(local.ms) },
      };
    } finally {
      worker.terminate();
    }
  },
};

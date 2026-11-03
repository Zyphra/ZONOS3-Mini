// Codec parity: ZacStream on goldens/tts.npy with the server's streaming block schedule vs zac-cli (zac_audio.npy).
import { ZacStream, HOP } from '../../src/zac';
import { loadModel } from '../../src/gpu/weights';
import { buildIdx, finalBlocks, readyBlocks, type Range } from '../../src/schedule';
import { cosine, maxAbs, type Check } from './util';

export const checks: Record<string, Check> = {
  async zac_stream(ctx) {
    const codesNpy = await ctx.npy('tts.npy');
    const [eos, prefill] = Array.from((await ctx.npy('tts.npy.eos.npy')).data, Math.round);
    const want = (await ctx.npy('zac_audio.npy')).data;
    const H = codesNpy.shape[0];
    const codes = Int32Array.from(codesNpy.data, Math.round);
    const zac = new ZacStream(ctx.gpu, await loadModel(ctx.gpu, ctx.base, 'zac'));
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
    parts.push(zac.flush());
    const ms = performance.now() - t0;
    const all = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
    let o = 0;
    for (const p of parts) { all.set(p, o); o += p.length; }
    const got = all.slice(Math.min(all.length, prefill * HOP)).map(v => v * zac.gain);

    const n = Math.min(got.length, want.length);
    const cos = cosine(got.subarray(0, n), want.subarray(0, n));
    return {
      ok: got.length === want.length && cos >= 0.999,
      metric: { cosine: cos, maxAbs: maxAbs(got.subarray(0, n), want.subarray(0, n)), samples: got.length, want: want.length,
        blocks: blocks.length, ms: Math.round(ms) },
    };
  },
};

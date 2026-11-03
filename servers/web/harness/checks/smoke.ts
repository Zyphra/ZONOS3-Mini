// Harness smoke check: the goldens are reachable and WebGPU runs a dispatch.
import type { Check } from './util';

export const checks: Record<string, Check> = {
  async smoke(ctx) {
    const ids = await ctx.npy('ids.npy');
    const buf = ctx.gpu.upload(new Float32Array([1, 2, 3, 4]), 'smoke');
    const back = await ctx.gpu.read(buf, 0, 16);
    buf.destroy();
    return { ok: ids.shape[1] === 11 && back[3] === 4, metric: { idsShape: ids.shape, read: Array.from(back) } };
  },
};

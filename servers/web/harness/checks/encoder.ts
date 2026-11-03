// Reference-encoder parity: RefEncoder codes of default_voices/AmericanFemale.mp3 (the ffmpeg-decoded 44.1 kHz PCM the
// CLIs read, goldens/spk/pcm44k.npy) vs the shipped AmericanFemale.codes.npy (dac-encoder-cli, native BF16 recipe).
// The web encoder ports that BF16 recipe, but near-tie RVQ picks still follow GPU accumulation order, so codes differ
// slightly; the plan's bar is codebook-0 agreement >= 0.95.
import { RefEncoder } from '../../src/encoder';
import { loadReferenceCodes } from '../../src/npy';
import type { Check } from './util';

export const checks: Record<string, Check> = {
  async encoder_codes(ctx) {
    const pcm = (await ctx.npy('spk/pcm44k.npy')).data;
    const res = await fetch(new URL('voices/AmericanFemale.codes.npy', ctx.base).href);
    if (!res.ok) throw new Error(`voices/AmericanFemale.codes.npy: ${res.status}`);
    const want = loadReferenceCodes(await res.arrayBuffer());
    const enc = await RefEncoder.load(ctx.gpu, ctx.base);
    const t0 = performance.now();
    const got = await enc.encode(pcm);
    const ms = performance.now() - t0;
    const frames = Math.min(got.length, want.length) / 10;
    const agree = new Array<number>(10).fill(0);
    for (let t = 0; t < frames; t++) for (let c = 0; c < 10; c++) if (got[t * 10 + c] === want[t * 10 + c]) agree[c]++;
    const rate = agree.map(a => Math.round((a / frames) * 1000) / 1000);
    return {
      ok: got.length === want.length && rate[0] >= 0.95,
      metric: { frames: got.length / 10, want: want.length / 10, cb0: rate[0], perCodebook: rate, ms: Math.round(ms) },
    };
  },
};

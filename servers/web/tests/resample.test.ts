import { describe, expect, it } from 'vitest';
import { MT19937 } from '../src/mt19937';
import { Resampler } from '../src/resample';

function rand(rng: MT19937, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = rng.next() / 0x100000000 - 0.5;
  return out;
}

const concat = (parts: Float32Array[]) => {
  const out = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
};

describe('Resampler', () => {
  const rng = new MT19937(7);
  const kernel = rand(rng, 735 * 770);
  const x = rand(rng, 7 * 2944);

  it('is chunking-invariant and emits exactly 2940 samples per 2944 input', () => {
    const whole = new Resampler(kernel);
    const ref = concat([whole.push(x), whole.flush()]);
    expect(ref.length).toBe(7 * 2940);

    const rs = new Resampler(kernel);
    const parts: Float32Array[] = [];
    let o = 0;
    for (const n of [100, 2944, 5000, x.length - 8044]) { parts.push(rs.push(x.subarray(o, o + n))); o += n; }
    parts.push(rs.flush());
    expect(concat(parts)).toEqual(ref);
  });

  it('sums only the non-zero band, identical to the dense formula', () => {
    const KL = 770, NW = 735, O = 736, W = 17;
    const band = new Float32Array(NW * KL);
    for (let i = 0; i < NW; i++) band.set(rand(rng, 33), i * KL + i + 1);   // taps [i+1, i+34)
    const rs = new Resampler(band);
    const got = concat([rs.push(x), rs.flush()]);
    const xe = (j: number) => (j < 0 || j >= x.length ? 0 : x[j]);
    const ref = new Float32Array(7 * 2940);
    for (let n = 0; n * NW < ref.length; n++) for (let i = 0; i < NW && n * NW + i < ref.length; i++) {
      let acc = 0;
      for (let k = 0; k < KL; k++) acc += band[i * KL + k] * xe(n * O + k - W);
      ref[n * NW + i] = acc;
    }
    expect(got).toEqual(ref);
  });
});

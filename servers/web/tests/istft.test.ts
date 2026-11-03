// Mixed-radix irfft vs a naive inverse DFT, and IstftHead stream lengths.
import { describe, expect, it } from 'vitest';
import { IstftHead, Irfft } from '../src/istft';

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
}

function naive(hc: Float64Array, n: number, scale: number): Float64Array {
  const out = new Float64Array(n);
  const h = n / 2;
  for (let j = 0; j < n; j++) {
    let s = hc[0] + hc[n - 1] * (j % 2 ? -1 : 1);
    for (let k = 1; k < h; k++) {
      const a = (2 * Math.PI * ((j * k) % n)) / n;
      s += 2 * (hc[2 * k - 1] * Math.cos(a) - hc[2 * k] * Math.sin(a));
    }
    out[j] = s * scale;
  }
  return out;
}

describe('Irfft', () => {
  for (const n of [736, 2208]) {
    it(`matches a naive inverse DFT for n=${n}`, () => {
      const r = rng(n);
      const hc = new Float64Array(n).map(() => r());
      const got = new Float64Array(n);
      new Irfft(n).exec(hc, got, 1 / n);
      const want = naive(hc, n, 1 / n);
      let err = 0;
      for (let j = 0; j < n; j++) err = Math.max(err, Math.abs(got[j] - want[j]));
      expect(err).toBeLessThan(1e-9);
    });
  }
});

describe('IstftHead', () => {
  it('push + flush yields cols * hop samples', () => {
    const nFft: [number, number] = [736, 2208];
    const hann = (n: number) => new Float32Array(n).map((_, j) => 0.5 - 0.5 * Math.cos((2 * Math.PI * j) / n));
    const r = rng(7);
    const mixer = new Float32Array(2 * 127).map(() => r());
    const head = new IstftHead({ nFft, hop: 184, windows: [hann(736), hann(2208)], mixer, maxLogMag: Math.log(100) });
    const spec = (cols: number, n: number) => new Float32Array(cols * (n + 2)).map(() => r());
    let total = 0;
    for (const cols of [64, 256, 16, 1]) total += head.push(spec(cols, nFft[0]), spec(cols, nFft[1]), cols).length;
    total += head.flush().length;
    expect(total).toBe((64 + 256 + 16 + 1) * 184);
    expect(head.flush().length).toBe(0);
    head.reset();
    const a = head.push(spec(32, nFft[0]), spec(32, nFft[1]), 32).length;
    expect(a + head.flush().length).toBe(32 * 184);
  });
});

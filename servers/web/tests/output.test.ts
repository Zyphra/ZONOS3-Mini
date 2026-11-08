import { describe, expect, it } from 'vitest';
import { OUTPUT_LIMIT_DBFS } from '../src/generation';
import { outputPcm } from '../src/output';

const KNEE = Math.fround(10 ** (OUTPUT_LIMIT_DBFS / 20));

describe('outputPcm (gain + soft limiter)', () => {
  it('only gains samples at or below the knee', () => {
    const x = Float32Array.from({ length: 2001 }, (_, i) => ((i - 1000) / 1000) * 1.2);
    const gain = 0.7;
    const y = outputPcm(x, gain);
    for (let i = 0; i < x.length; i++) {
      const g = Math.fround(x[i] * gain);
      if (Math.abs(g) <= KNEE) expect(y[i]).toBe(g);
    }
  });

  it('bends peaks into full scale monotonically and symmetrically', () => {
    const x = Float32Array.from({ length: 2000 }, (_, i) => KNEE + (7 * (i + 1)) / 2000);
    const y = outputPcm(x, 1);
    const neg = outputPcm(x.map((v) => -v), 1);
    for (let i = 0; i < y.length; i++) {
      expect(y[i]).toBeGreaterThan(KNEE);
      expect(y[i]).toBeLessThanOrEqual(1);
      expect(y[i]).toBeLessThanOrEqual(x[i]);
      if (i) expect(y[i]).toBeGreaterThanOrEqual(y[i - 1]);
      expect(neg[i]).toBe(-y[i]);
    }
    expect(outputPcm(Float32Array.of(1.2), 1)[0]).toBeLessThan(1);
  });

  it('streams chunk-independently', () => {
    const x = Float32Array.from({ length: 5000 }, (_, i) => 1.4 * Math.sin(i / 7));
    const whole = outputPcm(x, 0.8);
    const parts: number[] = [];
    for (let i = 0; i < x.length; i += 333) parts.push(...outputPcm(x.subarray(i, i + 333), 0.8));
    expect(Float32Array.from(parts)).toEqual(whole);
  });
});

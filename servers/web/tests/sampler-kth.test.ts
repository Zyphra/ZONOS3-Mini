import { describe, expect, it } from 'vitest';
import { MT19937 } from '../src/mt19937';
import { kthLargest } from '../src/sampler';

describe('kthLargest', () => {
  it('equals the sorted-array k-th largest, ties and -Infinity included', () => {
    const rng = new MT19937(1234);
    const below = (n: number) => rng.next() % n;
    const pool = [-Infinity, -3.5, -1, 0, 0.25, 0.25, 2, 7.75, Infinity];
    const scratch = new Float32Array(4098);
    for (let t = 0; t < 200; t++) {
      const n = t < 5 ? t + 1 : 1 + below(4098);
      // a small value set so ties are common; some arrays are drawn from only two values
      const set = t % 7 === 0 ? [pool[below(pool.length)], pool[below(pool.length)]] : pool;
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = set[below(set.length)];
      const k = 1 + below(n);
      const before = a.slice();
      expect(kthLargest(a, k, scratch)).toBe(Float32Array.from(a).sort()[n - k]);
      expect(a).toEqual(before);
    }
  });
});

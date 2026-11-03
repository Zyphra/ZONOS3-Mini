import { describe, expect, it } from 'vitest';
import { MT19937, canonical53 } from '../src/mt19937';

describe('mt19937', () => {
  it('matches the reference sequence for seed 5489', () => {
    const r = new MT19937(5489);
    expect(r.next()).toBe(3499211612);
    for (let i = 2; i < 10000; i++) r.next();
    expect(r.next()).toBe(4123659995);
  });
  it('canonical53 combines two draws low word first', () => {
    const a = new MT19937(1), b = new MT19937(1);
    const r1 = a.next(), r2 = a.next();
    expect(canonical53(b)).toBe((r1 + r2 * 2 ** 32) / 2 ** 64);
  });
});

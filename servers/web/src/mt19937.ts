// MT19937 (32-bit) + libc++ generate_canonical<double,53> / uniform_real_distribution<double>.
export class MT19937 {
  private mt = new Uint32Array(624);
  private idx = 625;

  constructor(seed: number) {
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const prev = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      // 1812433253 * prev + i, mod 2^32, without losing precision
      this.mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
    this.idx = 624;
  }

  next(): number {
    const mt = this.mt;
    if (this.idx >= 624) {
      for (let i = 0; i < 624; i++) {
        const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
        let v = mt[(i + 397) % 624] ^ (y >>> 1);
        if (y & 1) v ^= 0x9908b0df;
        mt[i] = v >>> 0;
      }
      this.idx = 0;
    }
    let y = mt[this.idx++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
}

/** libc++ generate_canonical<double, 53>(mt19937): two draws, first is the low word. */
export function canonical53(rng: MT19937): number {
  const r1 = rng.next();
  const r2 = rng.next();
  return (r1 + r2 * 4294967296) / 18446744073709551616;
}

/** std::uniform_real_distribution<double>(0, s2)(rng) as libc++ computes it. */
export function uniform(rng: MT19937, s2: number): number {
  return s2 * canonical53(rng);
}

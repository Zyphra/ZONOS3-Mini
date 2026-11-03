// Streaming torchaudio sinc resampler 44160 -> 44100 Hz, a port of resample() in zac.cpp:
//   out[n*nw + i] = sum_k kernel[i][k] * x[n*orig + k - width],  x = 0 outside [0, inCount).
// Block n needs input up to n*orig + orig + width; push() waits for it, flush() zero-extends and
// truncates the total to ceil(nw * inCount / orig).
// Rows of the torchaudio kernel are a short contiguous band (32–33 of 770 taps); only the band is summed,
// which is exact because the skipped taps are 0.
export class Resampler {
  private buf = new Float32Array(0);
  private len = 0;          // valid samples in buf
  private base = 0;         // absolute index of buf[0]
  private inCount = 0;      // total samples pushed
  private nextBlock = 0;
  private readonly kl: number;
  private readonly lo: Int32Array;   // first non-zero tap per row
  private readonly hi: Int32Array;   // one past the last non-zero tap per row

  constructor(private kernel: Float32Array, private orig = 736, private nw = 735, private width = 17) {
    this.kl = 2 * width + orig;
    if (kernel.length !== nw * this.kl) throw new Error(`resampler: kernel has ${kernel.length} taps, expected ${nw * this.kl}`);
    this.lo = new Int32Array(nw);
    this.hi = new Int32Array(nw);
    for (let i = 0; i < nw; i++) {
      const ki = i * this.kl;
      let a = 0; while (a < this.kl && kernel[ki + a] === 0) a++;
      let z = this.kl; while (z > a && kernel[ki + z - 1] === 0) z--;
      this.lo[i] = a; this.hi[i] = z;
    }
  }

  push(x: Float32Array): Float32Array {
    if (this.len + x.length > this.buf.length) {
      const nb = new Float32Array(Math.max(this.len + x.length, 2 * this.buf.length));
      nb.set(this.buf.subarray(0, this.len));
      this.buf = nb;
    }
    this.buf.set(x, this.len);
    this.len += x.length;
    this.inCount += x.length;
    return this.run(false);
  }

  flush(): Float32Array { return this.run(true); }

  reset(): void {
    this.len = 0;
    this.base = 0;
    this.inCount = 0;
    this.nextBlock = 0;
  }

  private run(flush: boolean): Float32Array {
    const { orig: O, nw: N, width: W, kl: KL, kernel, lo, hi } = this;
    const total = flush ? Math.floor((N * this.inCount + O - 1) / O) : -1;
    const blocks: Float32Array[] = [];
    const xw = new Float32Array(KL);
    for (;; this.nextBlock++) {
      const n = this.nextBlock, x0 = n * O - W;
      if (flush ? n * N >= total : x0 + KL > this.inCount) break;
      for (let k = 0; k < KL; k++) {
        const j = x0 + k;
        xw[k] = j < 0 || j >= this.inCount ? 0 : this.buf[j - this.base];
      }
      const emit = flush ? Math.min(N, total - n * N) : N;
      const y = new Float32Array(emit);
      for (let i = 0; i < emit; i++) {
        const ki = i * KL;
        let acc = 0;
        for (let k = lo[i]; k < hi[i]; k++) acc += kernel[ki + k] * xw[k];
        y[i] = acc;
      }
      blocks.push(y);
    }
    // Drop samples no future block reads.
    const keepFrom = Math.min(this.inCount, Math.max(0, this.nextBlock * O - W));
    if (keepFrom > this.base) {
      const drop = keepFrom - this.base;
      this.buf.copyWithin(0, drop, this.len);
      this.len -= drop;
      this.base = keepFrom;
    }
    const out = new Float32Array(blocks.reduce((a, b) => a + b.length, 0));
    let o = 0;
    for (const b of blocks) { out.set(b, o); o += b.length; }
    return out;
  }
}

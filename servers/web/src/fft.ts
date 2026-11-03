// In-place radix-2 forward complex FFT of length 512 in Float64 (fbank frontends of the SSL teacher and speaker encoder).
const TWIDDLE_RE = new Float64Array(256);
const TWIDDLE_IM = new Float64Array(256);
for (let i = 0; i < 256; i++) {
  const angle = -2 * Math.acos(-1) * i / 512;
  TWIDDLE_RE[i] = Math.cos(angle);
  TWIDDLE_IM[i] = Math.sin(angle);
}

export function fft512(re: Float64Array, im: Float64Array): void {
  const n = 512;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const step = n / len, half = len / 2;
    for (let i = 0; i < n; i += len) {
      for (let j = 0; j < half; j++) {
        const wr = TWIDDLE_RE[j * step], wi = TWIDDLE_IM[j * step];
        const a = i + j, b = a + half;
        const vr = re[b] * wr - im[b] * wi, vi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr; im[a] += vi;
      }
    }
  }
}

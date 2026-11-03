// Host multi-resolution iSTFT head + FIR mixer at 44160 Hz: a port of head_init/head_push/ola_finalize/
// mix_emit/head_flush (servers/ggml/src/zac.cpp), with a Float64 mixed-radix real inverse DFT in place of pocketfft.

/** Real inverse DFT of even length n from FFTPACK halfcomplex input [r0, r1, i1, …, r(n/2-1), i(n/2-1), r(n/2)]:
 *  x[j] = scale * (r0 + r(n/2)(-1)^j + 2 Σ_k (rk cos(2πjk/n) - ik sin(2πjk/n))). Mixed-radix decimation in time over
 *  the Hermitian-extended spectrum: radix-2/3 butterflies, a direct DFT for any other prime factor (23 here). */
export class Irfft {
  private readonly factors: number[] = [];
  private readonly cos: Float64Array;   // cos/sin(2πk/n)
  private readonly sin: Float64Array;
  private readonly xr: Float64Array;    // Hermitian spectrum
  private readonly xi: Float64Array;
  private readonly yr: Float64Array;    // complex output
  private readonly yi: Float64Array;
  private readonly tr: Float64Array;    // butterfly scratch
  private readonly ti: Float64Array;

  constructor(readonly n: number) {
    if (!Number.isInteger(n) || n < 2 || n % 2) throw new Error(`irfft: n=${n} must be even`);
    let r = n;
    for (const p of [2, 3]) while (r % p === 0) { this.factors.push(p); r /= p; }
    for (let p = 5; r > 1; p += 2) while (r % p === 0) { this.factors.push(p); r /= p; }
    // The largest factor is the leaf: its direct DFT runs on gathered inputs without twiddles.
    this.factors.sort((a, b) => a - b);
    this.cos = new Float64Array(n);
    this.sin = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const a = (2 * Math.PI * k) / n;
      this.cos[k] = Math.cos(a);
      this.sin[k] = Math.sin(a);
    }
    this.xr = new Float64Array(n); this.xi = new Float64Array(n);
    this.yr = new Float64Array(n); this.yi = new Float64Array(n);
    const pmax = this.factors[this.factors.length - 1];
    this.tr = new Float64Array(pmax); this.ti = new Float64Array(pmax);
  }

  exec(hc: ArrayLike<number>, out: Float64Array | Float32Array, scale: number): void {
    const { n, xr, xi, yr } = this;
    const h = n / 2;
    xr[0] = hc[0]; xi[0] = 0;
    for (let k = 1; k < h; k++) {
      xr[k] = hc[2 * k - 1]; xi[k] = hc[2 * k];
      xr[n - k] = xr[k]; xi[n - k] = -xi[k];
    }
    xr[h] = hc[n - 1]; xi[h] = 0;
    this.rec(0, 1, 0, 0, n);
    for (let j = 0; j < n; j++) out[j] = yr[j] * scale;
  }

  /** y[o + j] = Σ_k x[i0 + k*is] e^{+2πi jk/nl}, j < nl, splitting by factors[level..]. */
  private rec(i0: number, is: number, o: number, level: number, nl: number): void {
    const { xr, xi, yr, yi, tr, ti, cos, sin, n } = this;
    const p = this.factors[level], m = nl / p;
    if (m === 1) {
      for (let r = 0; r < p; r++) { tr[r] = xr[i0 + r * is]; ti[r] = xi[i0 + r * is]; }
      this.dft(p, o, 1);
      return;
    }
    for (let r = 0; r < p; r++) this.rec(i0 + r * is, is * p, o + r * m, level + 1, m);
    const s = n / nl;   // w_nl^a = w_n^(a*s)
    for (let j0 = 0; j0 < m; j0++) {
      tr[0] = yr[o + j0]; ti[0] = yi[o + j0];
      for (let r = 1; r < p; r++) {
        const a = yr[o + r * m + j0], b = yi[o + r * m + j0];
        const k = (j0 * r * s) % n, c = cos[k], d = sin[k];
        tr[r] = a * c - b * d; ti[r] = a * d + b * c;
      }
      this.dft(p, o + j0, m);
    }
  }

  /** y[o + q*stride] = Σ_r t[r] w_p^(qr), q < p. */
  private dft(p: number, o: number, stride: number): void {
    const { yr, yi, tr, ti, cos, sin, n } = this;
    if (p === 2) {
      yr[o] = tr[0] + tr[1]; yi[o] = ti[0] + ti[1];
      yr[o + stride] = tr[0] - tr[1]; yi[o + stride] = ti[0] - ti[1];
      return;
    }
    if (p === 3) {
      const c = -0.5, d = Math.sqrt(3) / 2;   // w_3 = e^{+2πi/3}
      const sr = tr[1] + tr[2], si = ti[1] + ti[2];
      const dr = tr[1] - tr[2], di = ti[1] - ti[2];
      const ar = tr[0] + c * sr, ai = ti[0] + c * si;
      yr[o] = tr[0] + sr; yi[o] = ti[0] + si;
      yr[o + stride] = ar - d * di; yi[o + stride] = ai + d * dr;
      yr[o + 2 * stride] = ar + d * di; yi[o + 2 * stride] = ai - d * dr;
      return;
    }
    const s = n / p;
    for (let q = 0; q < p; q++) {
      let ar = 0, ai = 0;
      for (let r = 0; r < p; r++) {
        const k = ((q * r) % p) * s, c = cos[k], d = sin[k];
        ar += tr[r] * c - ti[r] * d;
        ai += tr[r] * d + ti[r] * c;
      }
      yr[o + q * stride] = ar; yi[o + q * stride] = ai;
    }
  }
}

/** Growable f32 vector with front erase (the std::vector<float> operations the C++ state uses). */
class Vec {
  data = new Float32Array(0);
  len = 0;

  /** std::vector::resize(n, 0.0f). */
  resize(n: number): void {
    if (n > this.data.length) {
      const d = new Float32Array(Math.max(n, 2 * this.data.length));
      d.set(this.data.subarray(0, this.len));
      this.data = d;
    } else if (n > this.len) this.data.fill(0, this.len, n);
    this.len = n;
  }

  erase(k: number): void {
    this.data.copyWithin(0, k, this.len);
    this.len -= k;
  }

  clear(): void { this.len = 0; }
}

export interface IstftParams {
  nFft: [number, number];                 // head.n_fft
  hop: number;                            // head.hop
  windows: [Float32Array, Float32Array];  // head.window.{0,1}
  mixer: Float32Array;                    // head.mixer.weight [2][K]
  maxLogMag: number;                      // head.max_log_magnitude
}

/** Streaming equivalent of torch.istft(center=True, length=frames*hop) per branch followed by the non-causal mixer
 *  Conv1d (2 -> 1, taps K, zero padding (K-1)/2). Frame t of branch i overlaps padded positions [t*hop, t*hop + n);
 *  branch sample m is padded position m + n/2. */
export class IstftHead {
  private readonly nFft: [number, number];
  private readonly hop: number;
  private readonly windows: [Float32Array, Float32Array];
  private readonly mixer: Float32Array;
  private readonly fir: number;
  private readonly maxLogMag: number;
  private readonly irfft: [Irfft, Irfft];
  private readonly hc: [Float64Array, Float64Array];
  private readonly buf: [Float32Array, Float32Array];

  private ola: [Vec, Vec] = [new Vec(), new Vec()];
  private env: [Vec, Vec] = [new Vec(), new Vec()];
  private olaBase = [0, 0];
  private mixIn: [Vec, Vec] = [new Vec(), new Vec()];
  private mixAvail = [0, 0];
  private mixBase = 0;
  private mixNext = 0;
  private headFrames = 0;
  private headDone = false;

  constructor(p: IstftParams) {
    this.nFft = p.nFft;
    this.hop = p.hop;
    this.fir = p.mixer.length / 2;
    if (!Number.isInteger(this.fir) || this.fir % 2 !== 1) throw new Error(`istft: mixer has ${p.mixer.length} taps, expected 2 x odd`);
    for (let i = 0; i < 2; i++)
      if (p.windows[i].length !== p.nFft[i]) throw new Error(`istft: window ${i} has ${p.windows[i].length} taps, expected ${p.nFft[i]}`);
    this.windows = p.windows;
    this.mixer = p.mixer;
    this.maxLogMag = p.maxLogMag;
    this.irfft = [new Irfft(p.nFft[0]), new Irfft(p.nFft[1])];
    this.hc = [new Float64Array(p.nFft[0]), new Float64Array(p.nFft[1])];
    this.buf = [new Float32Array(p.nFft[0]), new Float32Array(p.nFft[1])];
  }

  /** Head frames pushed so far. */
  get frames(): number { return this.headFrames; }

  reset(): void {
    for (let i = 0; i < 2; i++) {
      this.ola[i].clear(); this.env[i].clear(); this.mixIn[i].clear();
      this.olaBase[i] = 0; this.mixAvail[i] = 0;
    }
    this.mixBase = 0; this.mixNext = 0;
    this.headFrames = 0; this.headDone = false;
  }

  /** Adds `nCols` head frames: y0 [nCols][n0 + 2], y1 [nCols][n1 + 2] (logmag[0:bins], phase[bins:2*bins]);
   *  returns the 44160 Hz samples that are final so far. */
  push(y0: Float32Array, y1: Float32Array, nCols: number): Float32Array {
    if (this.headDone) throw new Error('istft: push after flush');
    if (this.headFrames === 0) this.init();
    const y = [y0, y1];
    const hop = this.hop;
    for (let i = 0; i < 2; i++) {
      const n = this.nFft[i], bins = n / 2 + 1;
      if (y[i].length < nCols * (n + 2)) throw new Error(`istft: branch ${i} has ${y[i].length} values, expected ${nCols * (n + 2)}`);
      const win = this.windows[i], hc = this.hc[i], buf = this.buf[i], yi = y[i], maxLog = this.maxLogMag;
      const ola = this.ola[i], env = this.env[i];
      for (let c = 0; c < nCols; c++) {
        const t = this.headFrames + c;
        const f = c * (n + 2), ph = f + bins;
        const mag = (k: number) => Math.exp(Math.min(yi[f + k], maxLog));
        hc[0] = mag(0) * Math.cos(yi[ph]);
        for (let k = 1; k < n / 2; k++) {
          const a = mag(k);
          hc[2 * k - 1] = a * Math.cos(yi[ph + k]);
          hc[2 * k] = a * Math.sin(yi[ph + k]);
        }
        hc[n - 1] = mag(n / 2) * Math.cos(yi[ph + n / 2]);
        this.irfft[i].exec(hc, buf, 1 / n);
        const need = t * hop + n - this.olaBase[i];
        if (ola.len < need) { ola.resize(need); env.resize(need); }
        const o = t * hop - this.olaBase[i], od = ola.data, ed = env.data;
        for (let j = 0; j < n; j++) { od[o + j] += win[j] * buf[j]; ed[o + j] += win[j] * win[j]; }
      }
    }
    this.headFrames += nCols;
    for (let i = 0; i < 2; i++) this.olaFinalize(i, this.headFrames * hop);
    return this.mixEmit();
  }

  /** End of stream: every sample m < L = frames*hop is final (the envelope only covers existing frames, like torch);
   *  the mixer's right zero padding is h zeros after the last branch sample. */
  flush(): Float32Array {
    if (this.headDone || this.headFrames === 0) return new Float32Array(0);
    this.headDone = true;
    const h = (this.fir - 1) / 2;
    for (let i = 0; i < 2; i++) {
      this.olaFinalize(i, this.olaBase[i] + this.ola[i].len);
      // Samples m >= L belong to the discarded tail padding; olaFinalize above produced them too.
      const L = this.headFrames * this.hop;
      const keep = h + L - this.mixBase;
      this.mixIn[i].resize(keep);
      this.mixIn[i].resize(keep + h);
      this.mixAvail[i] = this.mixBase + keep + h;
    }
    return this.mixEmit();
  }

  private init(): void {
    const h = (this.fir - 1) / 2;
    for (let i = 0; i < 2; i++) {
      this.mixIn[i].clear();
      this.mixIn[i].resize(h);
      this.mixAvail[i] = h;
    }
    this.mixBase = 0; this.mixNext = 0;
  }

  /** Finalizes overlap-add positions [olaBase, pf) of branch i into the mixer input (dropping the centre padding m < 0). */
  private olaFinalize(i: number, pf: number): void {
    const pad = this.nFft[i] / 2;
    const n = pf - this.olaBase[i];
    if (n <= 0) return;
    const ola = this.ola[i], env = this.env[i], mix = this.mixIn[i];
    const skip = Math.min(n, Math.max(0, pad - this.olaBase[i]));
    const m0 = mix.len;
    mix.resize(m0 + n - skip);
    const md = mix.data, od = ola.data, ed = env.data;
    for (let j = skip; j < n; j++) md[m0 + j - skip] = od[j] / ed[j];
    this.mixAvail[i] += n - skip;
    ola.erase(n);
    env.erase(n);
    this.olaBase[i] = pf;
  }

  /** Emits mixer outputs y[m] for every m whose taps are available: mixIn index j holds s[j - h];
   *  y[m] = Σ_c Σ_k w[c][k] * mixIn_c[m + k]. */
  private mixEmit(): Float32Array {
    const K = this.fir;
    const avail = Math.min(this.mixAvail[0], this.mixAvail[1]);
    const m1 = avail - K + 1;   // exclusive bound on m
    if (m1 <= this.mixNext) return new Float32Array(0);
    const n = m1 - this.mixNext;
    const y = new Float32Array(n);
    for (let c = 0; c < 2; c++) {
      const s = this.mixIn[c].data, s0 = this.mixNext - this.mixBase, w = c * K;
      for (let k = 0; k < K; k++) {
        const wk = this.mixer[w + k];
        for (let i = 0; i < n; i++) y[i] += wk * s[s0 + i + k];
      }
    }
    this.mixNext = m1;
    // Later outputs read from index mixNext on.
    for (let c = 0; c < 2; c++) if (this.mixNext > this.mixBase) this.mixIn[c].erase(this.mixNext - this.mixBase);
    this.mixBase = this.mixNext;
    return y;
  }
}

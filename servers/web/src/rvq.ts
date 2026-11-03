// Host-side reference-encoder pieces in F32 arithmetic (Math.fround at every native float operation):
// pool_ssl and the RVQ quantizer of zac-encoder.cpp.
const f = Math.fround;
export const LANE = 300;
const SSL_DIM = 1024;

/** Port of pool_ssl: windowed average of the teacher frames [T][1024] onto the 300 lane frames. */
export function poolSsl(ssl: Float32Array, T: number, windowRatio = 2): Float32Array {
  const C = SSL_DIM;
  const prefix = new Float32Array((T + 1) * C);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < C; c++) prefix[(t + 1) * C + c] = prefix[t * C + c] + ssl[t * C + c];
  const pooled = new Float32Array(LANE * C);
  const ratio = f(T / LANE), half = f(f(0.5 * f(windowRatio)) * ratio);
  for (let t = 0; t < LANE; t++) {
    const center = f(f(t + 0.5) * ratio);
    const left = Math.min(Math.max(f(center - half), 0), T), right = Math.min(Math.max(f(center + half), 0), T);
    const li = Math.floor(left), ri = Math.floor(right);
    const lf = f(left - li), rf = f(right - ri);
    const span = f(right - left);
    const lr = Math.min(li, T - 1) * C, rr = Math.min(ri, T - 1) * C;
    for (let c = 0; c < C; c++) {
      const a = f(prefix[li * C + c] + f(ssl[lr + c] * lf));
      const b = f(prefix[ri * C + c] + f(ssl[rr + c] * rf));
      pooled[t * C + c] = f(f(b - a) / span);
    }
  }
  return pooled;
}

/** One residual VQ level: in_proj -> l2-normalized nearest codebook row -> straight-through lookup -> out_proj. */
export class Rvq {
  readonly inputDim: number;
  private readonly codeDim = 8;
  private readonly size = 4096;
  private inW: Float32Array;
  private inB: Float32Array;
  private outW: Float32Array;
  private outB: Float32Array;
  private codebook: Float32Array;
  private normalized: Float32Array;
  private normSquared: Float32Array;

  constructor(host: Map<string, Float32Array>, c: number) {
    const get = (n: string) => {
      const v = host.get(`rvq.${c}.${n}`);
      if (!v) throw new Error(`encoder: missing rvq.${c}.${n}; re-export with npm run export`);
      return v;
    };
    this.inputDim = c === 0 ? 768 : 1024;
    this.inW = get('in_proj.weight'); this.inB = get('in_proj.bias');
    this.outW = get('out_proj.weight'); this.outB = get('out_proj.bias');
    this.codebook = get('codebook');
    const D = this.codeDim;
    this.normalized = new Float32Array(this.codebook.length);
    this.normSquared = new Float32Array(this.size);
    for (let k = 0; k < this.size; k++) {
      let norm = 0;
      for (let d = 0; d < D; d++) { const v = this.codebook[k * D + d]; norm = f(norm + f(v * v)); }
      norm = Math.max(f(Math.sqrt(norm)), f(1e-12));
      let sq = 0;
      for (let d = 0; d < D; d++) {
        const v = f(this.codebook[k * D + d] / norm);
        this.normalized[k * D + d] = v;
        sq = f(sq + f(v * v));
      }
      this.normSquared[k] = sq;
    }
  }

  /** Port of quantize(): input [T][inputDim] -> codes [T] and quantized [T][inputDim]. */
  quantize(input: Float32Array, T: number): { ids: Int32Array; quantized: Float32Array } {
    const C = this.inputDim, D = this.codeDim;
    const ids = new Int32Array(T);
    const quantized = new Float32Array(T * C);
    const proj = new Float32Array(D), nrm = new Float32Array(D), lookup = new Float32Array(D);
    for (let t = 0; t < T; t++) {
      let norm = 0;
      for (let d = 0; d < D; d++) {
        let v = 0;
        for (let c = 0; c < C; c++) v = f(v + f(input[t * C + c] * this.inW[d * C + c]));
        v = f(v + this.inB[d]);
        proj[d] = v;
        norm = f(norm + f(v * v));
      }
      norm = Math.max(f(Math.sqrt(norm)), f(1e-12));
      let enc2 = 0;
      for (let d = 0; d < D; d++) { nrm[d] = f(proj[d] / norm); enc2 = f(enc2 + f(nrm[d] * nrm[d])); }
      let best = Infinity, id = 0;
      for (let k = 0; k < this.size; k++) {
        let dot = 0;
        for (let d = 0; d < D; d++) dot = f(dot + f(nrm[d] * this.normalized[k * D + d]));
        const dist = f(f(enc2 - f(2 * dot)) + this.normSquared[k]);
        if (dist < best) { best = dist; id = k; }   // first index wins exact ties
      }
      ids[t] = id;
      for (let d = 0; d < D; d++) { const ze = proj[d]; lookup[d] = f(ze + f(this.codebook[id * D + d] - ze)); }
      for (let c = 0; c < C; c++) {
        let v = 0;
        for (let d = 0; d < D; d++) v = f(v + f(lookup[d] * this.outW[c * D + d]));
        quantized[t * C + c] = f(v + this.outB[c]);
      }
    }
    return { ids, quantized };
  }
}

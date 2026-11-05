// Split-K gemm_h8 for few rows (T <= 32; native/cl/kernels/gemm_h8.cl GEMM_H8_KS, f32 accumulation). S is a module
// constant injected by Gpu.kernel (name gemm_h8_ks<S>). Workgroup = 8 channel groups (64 outputs) x S K-slices over
// 4 rows: slice s covers input channels [s*IC/S, (s+1)*IC/S) of every tap (IC % (4*S) == 0); partials reduce in
// workgroup memory and the 32 (row, output) results of each channel group are finished by the S slice threads.
// Weights, params, bindings and epilogue as gemm_h8.wgsl.
struct P {
  T: u32, O: u32, IC: u32, K: u32, d: u32,
  x_stride: u32, x_off: u32, y_stride: u32, y_off: u32, r_stride: u32, r_off: u32,
  nb_bias: u32, flags: u32, act: u32, alpha: f32, cap: f32, rows_dyn: u32,
}
@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> scl: array<f32>;
@group(0) @binding(4) var<storage, read> shf: array<f32>;
@group(0) @binding(5) var<storage, read> res: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(14) var<uniform> dyn: Dyn;
@group(0) @binding(15) var<uniform> p: P;

/** [slice][channel group][row][half] partial sums of 8 outputs (two vec4f per row). */
var<workgroup> red: array<vec4f, 8u * S * 4u * 2u>;

fn lo8(w: vec4<u32>) -> vec4f { return vec4f(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn hi8(w: vec4<u32>) -> vec4f { return vec4f(unpack2x16float(w.z), unpack2x16float(w.w)); }

@compute @workgroup_size(8, S)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) l3: vec3u) {
  let lx = l3.x;
  let s = l3.y;
  let og = wid.x * 8u + lx;
  let t0 = wid.z * 4u;
  let OG = (p.O + 7u) / 8u;
  let OP8 = ((p.O + 31u) / 32u) * 4u;
  let nv = select(p.T, min(p.T, dyn.b * p.rows_dyn), p.rows_dyn != 0u);
  if (t0 >= nv) { return; }
  var acc: array<vec4f, 8>;
  for (var r = 0u; r < 8u; r++) { acc[r] = vec4f(0.0); }
  if (og < OG) {
    let len = p.IC / S;
    let lo = s * len;
    let hi = lo + len;
    var xb: array<u32, 4>;
    for (var k = 0u; k < p.K; k++) {
      let wb = k * p.IC * OP8 + og;
      for (var r = 0u; r < 4u; r++) { xb[r] = (p.x_off + (min(t0 + r, p.T - 1u) + k * p.d) * p.x_stride) >> 2u; }
      for (var i = lo; i < hi; i += 4u) {
        let wi = wb + i * OP8;
        let w0 = W[wi]; let w1 = W[wi + OP8]; let w2 = W[wi + 2u * OP8]; let w3 = W[wi + 3u * OP8];
        let a0 = lo8(w0); let a1 = lo8(w1); let a2 = lo8(w2); let a3 = lo8(w3);
        let b0 = hi8(w0); let b1 = hi8(w1); let b2 = hi8(w2); let b3 = hi8(w3);
        let xi = i >> 2u;
        for (var r = 0u; r < 4u; r++) {
          let xv = X[xb[r] + xi];
          acc[2u * r] += xv.x * a0 + xv.y * a1 + xv.z * a2 + xv.w * a3;
          acc[2u * r + 1u] += xv.x * b0 + xv.y * b1 + xv.z * b2 + xv.w * b3;
        }
      }
    }
  }
  let rb = (s * 8u + lx) * 8u;
  for (var r = 0u; r < 8u; r++) { red[rb + r] = acc[r]; }
  workgroupBarrier();
  if (og >= OG) { return; }
  for (var e = s; e < 32u; e += S) {
    let r = e >> 3u;
    let j = e & 7u;
    let t = t0 + r;
    let o = og * 8u + j;
    if (t >= p.T || o >= p.O) { continue; }
    var y = 0.0;
    for (var q = 0u; q < S; q++) { y += red[(q * 8u + lx) * 8u + 2u * r + (j >> 2u)][j & 3u]; }
    if ((p.flags & 1u) != 0u) { y += bias[o % p.nb_bias]; }
    y = act(y, p.act, p.cap);
    if ((p.flags & 2u) != 0u) { y *= scl[o]; }
    if ((p.flags & 4u) != 0u) { y += shf[o]; }
    y *= p.alpha;
    if ((p.flags & 8u) != 0u) { y += res[p.r_off + t * p.r_stride + o]; }
    Y[p.y_off + t * p.y_stride + o] = y;
  }
}

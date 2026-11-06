// Register-tiled implicit GEMM over k-major half weights (repack_h8 layout W[K][IC][Op], Op = ceil32(O); one
// vec4<u32> = 8 halves): adjacent threads (consecutive og) read adjacent half8 chunks; no workgroup memory
// (native/cl/kernels/gemm_h8.cl GEMM_H8, f32 accumulation). R, LX, LY are module constants injected by Gpu.kernel
// (name gemm_h8_r<R>_<LX>x<LY>). Thread (og, tg): rows tg*R .. +R-1 x outputs og*8 .. +7.
// acc[t][o] = sum_k sum_i X[x_off + (min(t,T-1) + k*d)*x_stride + i] * W[k][i][o]; IC % 16 == 0, x_off and x_stride
// multiples of 4. rows_dyn != 0: row groups at or past dyn.b * rows_dyn valid rows exit. Epilogue as gemm_q8.wgsl.
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

fn lo8(w: vec4<u32>) -> vec4f { return vec4f(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn hi8(w: vec4<u32>) -> vec4f { return vec4f(unpack2x16float(w.z), unpack2x16float(w.w)); }

fn store1(v: f32, t: u32, o: u32) {
  if (o >= p.O) { return; }
  var y = v;
  if ((p.flags & 1u) != 0u) { y += bias[o % p.nb_bias]; }
  y = act(y, p.act, p.cap);
  if ((p.flags & 2u) != 0u) { y *= scl[o]; }
  if ((p.flags & 4u) != 0u) { y += shf[o]; }
  y *= p.alpha;
  if ((p.flags & 8u) != 0u) { y += res[p.r_off + t * p.r_stride + o]; }
  Y[p.y_off + t * p.y_stride + o] = y;
}

fn store8(a: vec4f, b: vec4f, t: u32, o: u32) {
  store1(a.x, t, o); store1(a.y, t, o + 1u); store1(a.z, t, o + 2u); store1(a.w, t, o + 3u);
  store1(b.x, t, o + 4u); store1(b.y, t, o + 5u); store1(b.z, t, o + 6u); store1(b.w, t, o + 7u);
}

@compute @workgroup_size(LX, LY)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let og = gid.x;
  let t0 = gid.y * R;
  let OG = (p.O + 7u) / 8u;
  let OP8 = ((p.O + 31u) / 32u) * 4u;
  let nv = select(p.T, min(p.T, dyn.b * p.rows_dyn), p.rows_dyn != 0u);
  if (og >= OG || t0 >= nv) { return; }
  var acc: array<vec4f, 2u * R>;
  for (var r = 0u; r < 2u * R; r++) { acc[r] = vec4f(0.0); }
  var xb: array<u32, R>;
  for (var k = 0u; k < p.K; k++) {
    let wb = k * p.IC * OP8 + og;
    for (var r = 0u; r < R; r++) { xb[r] = (p.x_off + (min(t0 + r, p.T - 1u) + k * p.d) * p.x_stride) >> 2u; }
    for (var i = 0u; i < p.IC; i += 4u) {
      let wi = wb + i * OP8;
      let w0 = W[wi]; let w1 = W[wi + OP8]; let w2 = W[wi + 2u * OP8]; let w3 = W[wi + 3u * OP8];
      let a0 = lo8(w0); let a1 = lo8(w1); let a2 = lo8(w2); let a3 = lo8(w3);
      let b0 = hi8(w0); let b1 = hi8(w1); let b2 = hi8(w2); let b3 = hi8(w3);
      let xi = i >> 2u;
      for (var r = 0u; r < R; r++) {
        let xv = X[xb[r] + xi];
        acc[2u * r] += xv.x * a0 + xv.y * a1 + xv.z * a2 + xv.w * a3;
        acc[2u * r + 1u] += xv.x * b0 + xv.y * b1 + xv.z * b2 + xv.w * b3;
      }
    }
  }
  for (var r = 0u; r < R; r++) {
    let t = t0 + r;
    if (t < p.T) { store8(acc[2u * r], acc[2u * r + 1u], t, og * 8u); }
  }
}

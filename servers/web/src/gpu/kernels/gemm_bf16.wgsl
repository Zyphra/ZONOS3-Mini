// gemm.wgsl with bf16 weights (two per u32, element 2i in the low half) for the native BF16 autocast recipe
// (reference encoder + SSL teacher): the X operand is rounded to bf16 as it is loaded (ggml converts src1 to the
// BF16 vec_dot type), products accumulate in f32, and `rnd` rounds the epilogue to bf16 at the requested points:
// 1 after the accumulation, 2 after +bias, 4 after act, 8 after *alpha, 16 after +res.
// Same tile and params as gemm.wgsl (64x64 output tile, 16x16 threads x 4x4 outputs, K-step 16, IC % 16 == 0).
struct P {
  T: u32, O: u32, IC: u32, K: u32, d: u32,
  x_stride: u32, x_off: u32, y_stride: u32, y_off: u32, r_stride: u32, r_off: u32,
  nb_bias: u32, flags: u32, act: u32, alpha: f32, cap: f32, rows_dyn: u32, rnd: u32,
}
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> scl: array<f32>;
@group(0) @binding(4) var<storage, read> shf: array<f32>;
@group(0) @binding(5) var<storage, read> res: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(14) var<uniform> dyn: Dyn;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> Xs: array<array<f32, 64>, 16>;
var<workgroup> Ws: array<array<f32, 64>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) l3: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let o0 = wid.x * 64u;
  let t0 = wid.y * 64u;
  let nv = select(p.T, min(p.T, dyn.b * p.rows_dyn), p.rows_dyn != 0u);
  if (t0 >= nv) { return; }
  let tx = l3.x;
  let ty = l3.y;
  var acc: array<array<f32, 4>, 4>;
  for (var a = 0u; a < 4u; a++) { for (var b = 0u; b < 4u; b++) { acc[a][b] = 0.0; } }
  let RR = p.K * p.IC;
  let xt = lid >> 2u;
  let xr = (lid & 3u) * 4u;
  let wo = lid >> 3u;
  let wr = (lid & 7u) * 2u;
  for (var r0 = 0u; r0 < RR; r0 += 16u) {
    let k = r0 / p.IC;
    let i0 = r0 - k * p.IC;
    let t = t0 + xt;
    if (t < p.T) {
      let base = p.x_off + (t + k * p.d) * p.x_stride + i0 + xr;
      Xs[xr][xt] = bf16r(X[base]);
      Xs[xr + 1u][xt] = bf16r(X[base + 1u]);
      Xs[xr + 2u][xt] = bf16r(X[base + 2u]);
      Xs[xr + 3u][xt] = bf16r(X[base + 3u]);
    } else {
      Xs[xr][xt] = 0.0; Xs[xr + 1u][xt] = 0.0; Xs[xr + 2u][xt] = 0.0; Xs[xr + 3u][xt] = 0.0;
    }
    for (var h = 0u; h < 2u; h++) {
      let oo = wo + h * 32u;
      let o = o0 + oo;
      var v = vec2f(0.0);
      if (o < p.O) {
        let w = Wt[((k * p.O + o) * p.IC + i0 + wr) >> 1u];
        v = vec2f(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u));
      }
      Ws[wr][oo] = v.x;
      Ws[wr + 1u][oo] = v.y;
    }
    workgroupBarrier();
    for (var r = 0u; r < 16u; r++) {
      let x0 = Xs[r][ty]; let x1 = Xs[r][ty + 16u]; let x2 = Xs[r][ty + 32u]; let x3 = Xs[r][ty + 48u];
      let w0 = Ws[r][tx]; let w1 = Ws[r][tx + 16u]; let w2 = Ws[r][tx + 32u]; let w3 = Ws[r][tx + 48u];
      acc[0][0] += x0 * w0; acc[0][1] += x0 * w1; acc[0][2] += x0 * w2; acc[0][3] += x0 * w3;
      acc[1][0] += x1 * w0; acc[1][1] += x1 * w1; acc[1][2] += x1 * w2; acc[1][3] += x1 * w3;
      acc[2][0] += x2 * w0; acc[2][1] += x2 * w1; acc[2][2] += x2 * w2; acc[2][3] += x2 * w3;
      acc[3][0] += x3 * w0; acc[3][1] += x3 * w1; acc[3][2] += x3 * w2; acc[3][3] += x3 * w3;
    }
    workgroupBarrier();
  }
  for (var a = 0u; a < 4u; a++) {
    let t = t0 + ty + 16u * a;
    if (t >= p.T) { continue; }
    for (var b = 0u; b < 4u; b++) {
      let o = o0 + tx + 16u * b;
      if (o >= p.O) { continue; }
      var y = acc[a][b];
      if ((p.rnd & 1u) != 0u) { y = bf16r(y); }
      if ((p.flags & 1u) != 0u) { y += bias[o % p.nb_bias]; }
      if ((p.rnd & 2u) != 0u) { y = bf16r(y); }
      y = act(y, p.act, p.cap);
      if ((p.rnd & 4u) != 0u) { y = bf16r(y); }
      if ((p.flags & 2u) != 0u) { y *= scl[o]; }
      if ((p.flags & 4u) != 0u) { y += shf[o]; }
      y *= p.alpha;
      if ((p.rnd & 8u) != 0u) { y = bf16r(y); }
      if ((p.flags & 8u) != 0u) { y += res[p.r_off + t * p.r_stride + o]; }
      if ((p.rnd & 16u) != 0u) { y = bf16r(y); }
      Y[p.y_off + t * p.y_stride + o] = y;
    }
  }
}

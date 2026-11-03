// Skinny implicit-GEMM causal conv / linear for small T: 16(T) x 64(O) output tile, 256 threads x 4 outputs,
// K-step 32. Same math, params and bindings as gemm.wgsl; weights are read 8 f16 per vec4<u32> (IC % 32 == 0).
struct P {
  T: u32, O: u32, IC: u32, K: u32, d: u32,
  x_stride: u32, x_off: u32, y_stride: u32, y_off: u32, r_stride: u32, r_off: u32,
  nb_bias: u32, flags: u32, act: u32, alpha: f32, cap: f32, rows_dyn: u32,
}
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> scl: array<f32>;
@group(0) @binding(4) var<storage, read> shf: array<f32>;
@group(0) @binding(5) var<storage, read> res: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(14) var<uniform> dyn: Dyn;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> Xs: array<array<f32, 16>, 32>;   // [r][t]
var<workgroup> Ws: array<array<f32, 65>, 32>;   // [r][o], padded

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let o0 = wid.x * 64u;
  let t0 = wid.y * 16u;
  let nv = select(p.T, min(p.T, dyn.b * p.rows_dyn), p.rows_dyn != 0u);
  if (t0 >= nv) { return; }
  let tx = lid & 15u;
  let ty = lid >> 4u;
  var acc: array<f32, 4>;
  for (var b = 0u; b < 4u; b++) { acc[b] = 0.0; }
  let RR = p.K * p.IC;
  for (var r0 = 0u; r0 < RR; r0 += 32u) {
    let k = r0 / p.IC;
    let i0 = r0 - k * p.IC;
    let t = t0 + ty;
    let xr = tx * 2u;
    if (t < p.T) {
      let base = p.x_off + (t + k * p.d) * p.x_stride + i0 + xr;
      Xs[xr][ty] = X[base];
      Xs[xr + 1u][ty] = X[base + 1u];
    } else {
      Xs[xr][ty] = 0.0;
      Xs[xr + 1u][ty] = 0.0;
    }
    let col = lid >> 2u;
    let part = (lid & 3u) * 8u;
    let o = o0 + col;
    var w = vec4<u32>(0u);
    if (o < p.O) { w = Wt[((k * p.O + o) * p.IC + i0 + part) >> 3u]; }
    let a = unpack2x16float(w.x); let b2 = unpack2x16float(w.y);
    let c = unpack2x16float(w.z); let d = unpack2x16float(w.w);
    Ws[part][col] = a.x; Ws[part + 1u][col] = a.y; Ws[part + 2u][col] = b2.x; Ws[part + 3u][col] = b2.y;
    Ws[part + 4u][col] = c.x; Ws[part + 5u][col] = c.y; Ws[part + 6u][col] = d.x; Ws[part + 7u][col] = d.y;
    workgroupBarrier();
    for (var r = 0u; r < 32u; r++) {
      let x = Xs[r][ty];
      acc[0] += x * Ws[r][tx]; acc[1] += x * Ws[r][tx + 16u];
      acc[2] += x * Ws[r][tx + 32u]; acc[3] += x * Ws[r][tx + 48u];
    }
    workgroupBarrier();
  }
  let t = t0 + ty;
  if (t >= p.T) { return; }
  for (var b = 0u; b < 4u; b++) {
    let o = o0 + tx + 16u * b;
    if (o >= p.O) { continue; }
    var y = acc[b];
    if ((p.flags & 1u) != 0u) { y += bias[o % p.nb_bias]; }
    y = act(y, p.act, p.cap);
    if ((p.flags & 2u) != 0u) { y *= scl[o]; }
    if ((p.flags & 4u) != 0u) { y += shf[o]; }
    y *= p.alpha;
    if ((p.flags & 8u) != 0u) { y += res[p.r_off + t * p.r_stride + o]; }
    Y[p.y_off + t * p.y_stride + o] = y;
  }
}

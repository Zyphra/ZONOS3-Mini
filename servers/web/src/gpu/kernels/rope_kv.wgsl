// Split a fused qkv row into heads; optional per-head RMS norm (+ temp on q); interleaved RoPE from a
// [pos][hd] (cos, sin) table on q and k; q -> q buffer, k/v -> KV store row (pos % ring).
// One 64-thread workgroup per (row, head slot); slots = nh q heads, nkv k heads, nkv v heads.
// hd <= 64: threads d >= hd read 0, join every barrier and never write.
// Dynamic mode: pos = dyn.a * pos_mul + i, valid rows = dyn.b * pos_mul. rnd != 0: roped q/k rounded to bf16.
struct P {
  n: u32, nh: u32, nkv: u32, in_stride: u32, qk_norm: u32, has_temp: u32, eps: f32, ring: u32,
  pos_dyn: u32, pos0: u32, n_dyn: u32, hd: u32, pos_mul: u32, rnd: u32,
}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> temp: array<f32>;
@group(0) @binding(2) var<storage, read> rope: array<f32>;
@group(0) @binding(3) var<storage, read_write> qo: array<f32>;
@group(0) @binding(4) var<storage, read_write> kc: array<f32>;
@group(0) @binding(5) var<storage, read_write> vc: array<f32>;
@group(0) @binding(14) var<uniform> dyn: Dyn;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> xs: array<f32, 64>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) d: u32) {
  let slots = p.nh + 2u * p.nkv;
  let id = wid.x + wid.y * nwg.x;
  if (id >= p.n * slots) { return; }
  let i = id / slots;
  let sl = id - i * slots;
  let nvalid = select(p.n, dyn.b * p.pos_mul, p.n_dyn != 0u);
  if (i >= nvalid) { return; }
  let hd = p.hd;
  let live = d < hd;
  let pos = select(p.pos0, dyn.a * p.pos_mul, p.pos_dyn != 0u) + i;
  let row = pos % p.ring;
  let base = i * p.in_stride + sl * hd;
  var x = select(0.0, qkv[base + min(d, hd - 1u)], live);
  if (sl >= p.nh + p.nkv) {
    if (live) { vc[(row * p.nkv + (sl - p.nh - p.nkv)) * hd + d] = x; }
    return;
  }
  if (p.qk_norm != 0u) {
    red[d] = x * x;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st >>= 1u) {
      if (d < st) { red[d] += red[d + st]; }
      workgroupBarrier();
    }
    x = x * inverseSqrt(red[0] / f32(hd) + p.eps);
  }
  if (sl < p.nh && p.has_temp != 0u) { x *= temp[sl]; }
  xs[d] = x;
  workgroupBarrier();
  if (!live) { return; }
  let pr = d >> 1u;
  let c = rope[pos * hd + 2u * pr];
  let sn = rope[pos * hd + 2u * pr + 1u];
  let x0 = xs[2u * pr];
  let x1 = xs[2u * pr + 1u];
  var o = select(x1 * c + x0 * sn, x0 * c - x1 * sn, (d & 1u) == 0u);
  if (p.rnd != 0u) { o = bf16r(o); }
  if (sl < p.nh) {
    qo[(i * p.nh + sl) * hd + d] = o;
  } else {
    kc[(row * p.nkv + (sl - p.nh)) * hd + d] = o;
  }
}

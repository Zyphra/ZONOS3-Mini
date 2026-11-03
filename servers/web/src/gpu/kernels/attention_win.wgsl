// Windowed causal self-attention for the codec stages (head dim 64, nkv == nh, no gate) over a ring KV
// store. One workgroup per (16 query rows, head): K/V tiles of 16 keys are staged in workgroup memory and
// each row keeps an online softmax. pos = dyn.a * pos_mul + i; tiles starting at or past the
// dyn.b * pos_mul valid rows exit (padded rows never feed valid rows).
struct P { n: u32, nh: u32, window: u32, ring: u32, scale: f32, pos_mul: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(14) var<uniform> dyn: Dyn;
@group(0) @binding(15) var<uniform> p: P;

const BQ = 16u;
const BK = 16u;
const HD = 64u;
var<workgroup> Qs: array<f32, 1024>;   // [BQ][HD]
var<workgroup> Ks: array<f32, 1040>;   // [BK][HD + 1], padded against bank conflicts
var<workgroup> Vs: array<f32, 1024>;   // [BK][HD]
var<workgroup> Ss: array<f32, 272>;    // [BQ][BK + 1]

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let h = wid.y;
  let i0 = wid.x * BQ;
  if (i0 >= dyn.b * p.pos_mul) { return; }
  let pos0 = dyn.a * p.pos_mul;
  for (var e = lid; e < BQ * HD; e += 256u) {
    let i = i0 + e / HD;
    Qs[e] = select(0.0, q[(i * p.nh + h) * HD + e % HD], i < p.n);
  }
  let r = lid >> 4u;      // query row within the tile (scores and outputs)
  let c = lid & 15u;      // key column within the tile (scores)
  let od = c * 4u;        // output dims od..od+3
  let qpos = pos0 + i0 + r;
  let first = pos0 + i0;
  let last = first + BQ - 1u;
  var jlo = 0u;
  if (first + 1u > p.window) { jlo = first + 1u - p.window; }
  var o = vec4f(0.0);
  var m = -3.0e38;
  var l = 0.0;
  workgroupBarrier();
  for (var j0 = jlo; j0 <= last; j0 += BK) {
    for (var e = lid; e < BK * HD; e += 256u) {
      let kk = e / HD;
      let d = e % HD;
      let slot = (((j0 + kk) % p.ring) * p.nh + h) * HD + d;
      Ks[kk * (HD + 1u) + d] = kc[slot];
      Vs[e] = vc[slot];
    }
    workgroupBarrier();
    let j = j0 + c;
    var s = -3.0e38;
    if (j <= qpos && qpos - j < p.window) {
      var acc = 0.0;
      for (var d = 0u; d < HD; d++) { acc += Qs[r * HD + d] * Ks[c * (HD + 1u) + d]; }
      s = acc * p.scale;
    }
    Ss[r * (BK + 1u) + c] = s;
    workgroupBarrier();
    var mt = m;
    for (var k = 0u; k < BK; k++) { mt = max(mt, Ss[r * (BK + 1u) + k]); }
    let cr = exp(m - mt);
    var ls = 0.0;
    var acc4 = vec4f(0.0);
    for (var k = 0u; k < BK; k++) {
      let sv = Ss[r * (BK + 1u) + k];
      let pe = select(0.0, exp(sv - mt), sv > -1.0e38);
      ls += pe;
      acc4 += pe * vec4f(Vs[k * HD + od], Vs[k * HD + od + 1u], Vs[k * HD + od + 2u], Vs[k * HD + od + 3u]);
    }
    o = o * cr + acc4;
    l = l * cr + ls;
    m = mt;
    workgroupBarrier();
  }
  let i = i0 + r;
  if (i < p.n) {
    let ob = (i * p.nh + h) * HD + od;
    let inv = 1.0 / l;
    outp[ob] = o.x * inv; outp[ob + 1u] = o.y * inv; outp[ob + 2u] = o.z * inv; outp[ob + 3u] = o.w * inv;
  }
}

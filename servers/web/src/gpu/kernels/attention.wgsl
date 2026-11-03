// Causal/windowed attention over a (ring) KV store; 32 threads per (row, head), online softmax + merge.
// Head dim hd <= 64. Dynamic mode: pos = dyn.a + i.
struct P {
  n: u32, nh: u32, nkv: u32, pos_dyn: u32, pos0: u32, window: u32, ring: u32, scale: f32,
  has_gate: u32, gate_stride: u32, gate_off: u32, hd: u32,
}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(14) var<uniform> dyn: Dyn;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> qs: array<f32, 64>;
var<workgroup> ms: array<f32, 32>;
var<workgroup> ls: array<f32, 32>;
var<workgroup> accs: array<f32, 2048>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) s: u32) {
  let id = wid.x + wid.y * nwg.x;
  if (id >= p.n * p.nh) { return; }
  let hd = p.hd;
  let i = id / p.nh;
  let h = id - i * p.nh;
  let pos = select(p.pos0, dyn.a, p.pos_dyn != 0u) + i;
  let kvh = h / (p.nh / p.nkv);
  let qb = (i * p.nh + h) * hd;
  for (var d = s; d < hd; d += 32u) { qs[d] = q[qb + d]; }
  workgroupBarrier();
  var jlo = 0u;
  if (pos + 1u > p.window) { jlo = pos + 1u - p.window; }
  var m = -3.0e38;
  var l = 0.0;
  var acc: array<f32, 64>;
  for (var d = 0u; d < hd; d++) { acc[d] = 0.0; }
  for (var j = jlo + s; j <= pos; j += 32u) {
    let base = ((j % p.ring) * p.nkv + kvh) * hd;
    var qk = 0.0;
    for (var d = 0u; d < hd; d++) { qk += qs[d] * kc[base + d]; }
    let sc = qk * p.scale;
    let mn = max(m, sc);
    let corr = exp(m - mn);
    let pe = exp(sc - mn);
    l = l * corr + pe;
    for (var d = 0u; d < hd; d++) { acc[d] = acc[d] * corr + pe * vc[base + d]; }
    m = mn;
  }
  ms[s] = m;
  ls[s] = l;
  for (var d = 0u; d < hd; d++) { accs[s * 64u + d] = acc[d]; }
  workgroupBarrier();
  var M = -3.0e38;
  for (var t = 0u; t < 32u; t++) { M = max(M, ms[t]); }
  var L = 0.0;
  for (var t = 0u; t < 32u; t++) { L += ls[t] * exp(ms[t] - M); }
  var g = 1.0;
  if (p.has_gate != 0u) { g = sigmoid(gate[p.gate_off + i * p.gate_stride + h]); }
  for (var hh = 0u; hh < 2u; hh++) {
    let d = s + hh * 32u;
    if (d < hd) {
      var o = 0.0;
      for (var t = 0u; t < 32u; t++) { o += accs[t * 64u + d] * exp(ms[t] - M); }
      outp[qb + d] = o / L * g;
    }
  }
}

// Masked probe heads (zonos3-graph.cpp build_probe): decode-row attention of head h over keys 0..dyn.a with the text
// keys (cache rows text_start + j, j < text_len) outside the head's window [lo, hi] blocked; overwrites the head's row of
// outp with o * sigmoid(gate). Only heads with flag != 0 of probe layer `layer` (pin layout: probe.wgsl) write.
// 32 threads per head, online softmax + merge (attention.wgsl).
struct P { layer: u32, nh: u32, nkv: u32, hd: u32, ring: u32, scale: f32, gate_off: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read> pin: array<u32>;
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;
@group(0) @binding(14) var<uniform> dyn: Dyn;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> qs: array<f32, 64>;
var<workgroup> ms: array<f32, 32>;
var<workgroup> ls: array<f32, 32>;
var<workgroup> accs: array<f32, 2048>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) s: u32) {
  let h = wid.x;
  let hd = p.hd;
  let e = 4u + (p.layer * p.nh + h) * 4u;
  let ts = i32(pin[0]);
  let tl = i32(pin[1]);
  let on = tl != 0 && pin[e + 3u] != 0u;
  let lo = i32(pin[e + 1u]);
  let hi = i32(pin[e + 2u]);
  let nkeys = select(0u, dyn.a + 1u, on);
  let kvh = h / (p.nh / p.nkv);
  let qb = h * hd;
  for (var d = s; d < hd; d += 32u) { qs[d] = q[qb + d]; }
  workgroupBarrier();
  var m = -3.0e38;
  var l = 0.0;
  var acc: array<f32, 64>;
  for (var d = 0u; d < hd; d++) { acc[d] = 0.0; }
  for (var j = s; j < nkeys; j += 32u) {
    let t = i32(j) - ts;
    if (t < 0 || t >= tl || (t >= lo && t <= hi)) {
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
  }
  ms[s] = m;
  ls[s] = l;
  for (var d = 0u; d < hd; d++) { accs[s * 64u + d] = acc[d]; }
  workgroupBarrier();
  if (!on) { return; }
  var M = -3.0e38;
  for (var t = 0u; t < 32u; t++) { M = max(M, ms[t]); }
  var L = 0.0;
  for (var t = 0u; t < 32u; t++) { L += ls[t] * exp(ms[t] - M); }
  let g = sigmoid(gate[p.gate_off + h]);
  for (var hh = 0u; hh < 2u; hh++) {
    let d = s + hh * 32u;
    if (d < hd) {
      var o = 0.0;
      for (var t = 0u; t < 32u; t++) { o += accs[t * 64u + d] * exp(ms[t] - M); }
      outp[qb + d] = o / L * g;
    }
  }
}

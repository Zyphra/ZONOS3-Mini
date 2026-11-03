// Decode-step alignment-probe readout (zonos3-graph.cpp build_probe). pin (u32 words) = [text_start, text_len, 0, 0],
// then per probe layer l and head h at 4 + (l*nh + h)*4: weight f32, lo, hi, flag (window clipped into [0, text_len)).
// text_len 0 = probe off.
// mode 0, one workgroup per head of probe layer `layer`:
//   sc[layer][h][j] = softmax_{j in [lo, hi]}(q_h . kc[text_start + j][h / (nh/nkv)] * scale), heads with weight != 0.
// mode 1, one thread per text index j < lmax: acc[j] = sum_l (sum_h w * sc[l][h][j]) over weighted heads with
//   j in [lo, hi] (0 elsewhere).
struct P { mode: u32, layer: u32, nl: u32, nh: u32, nkv: u32, hd: u32, ring: u32, scale: f32, lmax: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> pin: array<u32>;
@group(0) @binding(3) var<storage, read_write> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> acc: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> qs: array<f32, 64>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let tl = pin[1];
  if (p.mode == 1u) {
    let j = wid.x * 256u + lid;
    if (tl != 0u && j < p.lmax) {
      var total = 0.0;
      for (var l = 0u; l < p.nl; l++) {
        var s = 0.0;
        for (var h = 0u; h < p.nh; h++) {
          let e = 4u + (l * p.nh + h) * 4u;
          let w = bitcast<f32>(pin[e]);
          if (w != 0.0 && j >= pin[e + 1u] && j <= pin[e + 2u]) { s += sc[(l * p.nh + h) * p.lmax + j] * w; }
        }
        total += s;
      }
      acc[j] = total;
    }
    return;
  }
  let h = wid.x;
  let e = 4u + (p.layer * p.nh + h) * 4u;
  let on = tl != 0u && bitcast<f32>(pin[e]) != 0.0;
  let lo = select(1u, pin[e + 1u], on);
  let hi = select(0u, pin[e + 2u], on);
  let kvh = h / (p.nh / p.nkv);
  if (lid < p.hd) { qs[lid] = q[h * p.hd + lid]; }
  workgroupBarrier();
  let ts = pin[0];
  let base = (p.layer * p.nh + h) * p.lmax;
  var m = -3.0e38;
  for (var j = lo + lid; j <= hi; j += 256u) {
    let kb = (((ts + j) % p.ring) * p.nkv + kvh) * p.hd;
    var s = 0.0;
    for (var d = 0u; d < p.hd; d++) { s += qs[d] * kc[kb + d]; }
    let v = s * p.scale;
    sc[base + j] = v;
    m = max(m, v);
  }
  red[lid] = m;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) { red[lid] = max(red[lid], red[lid + st]); }
    workgroupBarrier();
  }
  let M = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = lo + lid; j <= hi; j += 256u) {
    let ev = exp(sc[base + j] - M);
    sc[base + j] = ev;
    sum += ev;
  }
  red[lid] = sum;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) { red[lid] += red[lid + st]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  for (var j = lo + lid; j <= hi; j += 256u) { sc[base + j] *= inv; }
}

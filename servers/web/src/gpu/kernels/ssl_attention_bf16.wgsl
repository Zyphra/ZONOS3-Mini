// SSL teacher self-attention (16 heads x 64, Shaw relative bias) at the native BF16 autocast boundaries
// (ssl-encoder.cpp build_graph): content = bf16(q.k), score = bf16(bf16(content) * scale), rel = bf16(bf16(dist.q) *
// scale), score = bf16(score + rel[clamp(j - i, -64, 8) + 64]); F32 softmax over all T keys; the probabilities are
// rounded to bf16 as the BF16 value matmul reads them and out = bf16(sum_j p_j * v_j). q, k, v and dist hold bf16
// values. One 64-thread workgroup per (query row, head); T <= 1024.
struct P { T: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> qb: array<f32>;
@group(0) @binding(1) var<storage, read> kb: array<f32>;
@group(0) @binding(2) var<storage, read> vb: array<f32>;
@group(0) @binding(3) var<storage, read> dist: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> qs: array<f32, 64>;
var<workgroup> rb: array<f32, 73>;
var<workgroup> sc: array<f32, 1024>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) s: u32) {
  let id = wid.x + wid.y * nwg.x;
  if (id >= p.T * 16u) { return; }
  let qi = id / 16u;
  let h = id - qi * 16u;
  let qo = qi * 1024u + h * 64u;
  qs[s] = qb[qo + s];
  workgroupBarrier();
  for (var r = s; r < 73u; r += 64u) {
    var d = 0.0;
    for (var c = 0u; c < 64u; c++) { d += dist[r * 64u + c] * qs[c]; }
    rb[r] = bf16r(bf16r(d) * p.scale);
  }
  workgroupBarrier();
  var mx = -3.0e38;
  for (var j = s; j < p.T; j += 64u) {
    let base = j * 1024u + h * 64u;
    var qk = 0.0;
    for (var d = 0u; d < 64u; d++) { qk += qs[d] * kb[base + d]; }
    let ri = u32(clamp(i32(j) - i32(qi), -64, 8) + 64);
    let v = bf16r(bf16r(bf16r(qk) * p.scale) + rb[ri]);
    sc[j] = v;
    mx = max(mx, v);
  }
  red[s] = mx;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (s < st) { red[s] = max(red[s], red[s + st]); }
    workgroupBarrier();
  }
  let M = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = s; j < p.T; j += 64u) {
    let e = exp(sc[j] - M);
    sc[j] = e;
    sum += e;
  }
  red[s] = sum;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (s < st) { red[s] += red[s + st]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  for (var j = s; j < p.T; j += 64u) { sc[j] = bf16r(sc[j] * inv); }
  workgroupBarrier();
  var o = 0.0;
  for (var j = 0u; j < p.T; j++) { o += sc[j] * vb[j * 1024u + h * 64u + s]; }
  outp[qo + s] = bf16r(o);
}

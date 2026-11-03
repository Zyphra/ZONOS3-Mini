// Full AttnRes route fused with the following weighted RMSNorm. One 256-thread workgroup per output
// row i (source row r = row0 + i), D <= 1024. Sources s < ns are bank[s] ([ns][n][D]).
// Logit l_s = q[qi] . rmsnorm(src_s); mix = softmax(l) . src; out[i] = rmsnorm(mix) * w.
struct P { D: u32, n: u32, ns: u32, qi: u32, row0: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> bank: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> red: array<vec2f, 256>;
var<workgroup> lg: array<f32, 32>;

fn reduce2(lid: u32, v: vec2f) -> vec2f {
  red[lid] = v;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) { red[lid] += red[lid + st]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

fn src_at(s: u32, r: u32, c: u32) -> f32 { return bank[(s * p.n + r) * p.D + c]; }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let i = wid.x;
  let r = p.row0 + i;
  let D = p.D;
  for (var s = 0u; s < p.ns; s++) {
    var acc = vec2f(0.0);
    for (var k = 0u; k < 4u; k++) {
      let c = lid + 256u * k;
      if (c < D) {
        let v = src_at(s, r, c);
        let qp = unpack2x16float(q[(p.qi * D + c) >> 1u]);
        let qv = select(qp.y, qp.x, (c & 1u) == 0u);
        acc += vec2f(v * v, qv * v);
      }
    }
    let t = reduce2(lid, acc);
    if (lid == 0u) { lg[s] = t.y * inverseSqrt(t.x / f32(D) + p.eps); }
  }
  workgroupBarrier();
  var mx = -3.0e38;
  for (var s = 0u; s < p.ns; s++) { mx = max(mx, lg[s]); }
  var sum = 0.0;
  for (var s = 0u; s < p.ns; s++) { sum += exp(lg[s] - mx); }
  var mixv: array<f32, 4>;
  for (var k = 0u; k < 4u; k++) { mixv[k] = 0.0; }
  for (var s = 0u; s < p.ns; s++) {
    let wt = exp(lg[s] - mx) / sum;
    for (var k = 0u; k < 4u; k++) {
      let c = lid + 256u * k;
      if (c < D) { mixv[k] += wt * src_at(s, r, c); }
    }
  }
  var ss = 0.0;
  for (var k = 0u; k < 4u; k++) { ss += mixv[k] * mixv[k]; }
  let scale = inverseSqrt(reduce2(lid, vec2f(ss, 0.0)).x / f32(D) + p.eps);
  for (var k = 0u; k < 4u; k++) {
    let c = lid + 256u * k;
    if (c < D) { outp[i * D + c] = mixv[k] * scale * w[c]; }
  }
}

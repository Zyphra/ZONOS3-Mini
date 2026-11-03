// Per row i (input row r = row0 + i): if has_x, res[r] += x[r] (in place);
// out[i] = res[r] * rsqrt(mean(res[r]^2) + eps) * (has_w ? w : 1).
struct P { N: u32, eps: f32, has_x: u32, has_w: u32, row0: u32 }
@group(0) @binding(0) var<storage, read_write> res: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let i = wid.x;
  let rb = (p.row0 + i) * p.N;
  var ss = 0.0;
  for (var c = lid; c < p.N; c += 256u) {
    var v = res[rb + c];
    if (p.has_x != 0u) { v += x[rb + c]; res[rb + c] = v; }
    ss += v * v;
  }
  red[lid] = ss;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) { red[lid] += red[lid + st]; }
    workgroupBarrier();
  }
  let scale = inverseSqrt(red[0] / f32(p.N) + p.eps);
  let ob = i * p.N;
  for (var c = lid; c < p.N; c += 256u) {
    var v = res[rb + c] * scale;
    if (p.has_w != 0u) { v *= w[c]; }
    outp[ob + c] = v;
  }
}

// Attentive statistics pooling over time, one workgroup per column k < N: alpha = softmax_t(att[:, k]),
// pooled[k] = mean = sum_t alpha*x, pooled[N + k] = sqrt(clamp(sum_t alpha*x*x - mean^2, var_floor, 3e38)).
struct P { T: u32, N: u32, var_floor: f32 }
@group(0) @binding(0) var<storage, read> att: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> pooled: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> red: array<f32, 256>;
var<workgroup> red2: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let k = wid.x + wid.y * nwg.x;
  if (k >= p.N) { return; }
  var m = -3.0e38;
  for (var t = lid; t < p.T; t += 256u) { m = max(m, att[t * p.N + k]); }
  red[lid] = m;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) { red[lid] = max(red[lid], red[lid + st]); }
    workgroupBarrier();
  }
  let M = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var t = lid; t < p.T; t += 256u) { s += exp(att[t * p.N + k] - M); }
  red[lid] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) { red[lid] += red[lid + st]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  workgroupBarrier();
  var s1 = 0.0;
  var s2 = 0.0;
  for (var t = lid; t < p.T; t += 256u) {
    let a = exp(att[t * p.N + k] - M) * inv;
    let v = x[t * p.N + k];
    s1 += a * v;
    s2 += a * (v * v);
  }
  red[lid] = s1;
  red2[lid] = s2;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) { red[lid] += red[lid + st]; red2[lid] += red2[lid + st]; }
    workgroupBarrier();
  }
  if (lid == 0u) {
    let mean = red[0];
    pooled[k] = mean;
    pooled[p.N + k] = sqrt(clamp(red2[0] - mean * mean, p.var_floor, 3.0e38));
  }
}

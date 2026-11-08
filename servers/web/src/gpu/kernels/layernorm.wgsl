// y = act((x - mean) / sqrt(var + eps) * w + b), biased variance (ggml_norm), one row per workgroup.
// rows_dyn != 0: rows at or past dyn.b * rows_dyn valid rows exit.
struct P { N: u32, eps: f32, act: u32, rows_dyn: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(14) var<uniform> dyn: Dyn;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> red: array<f32, 256>;

fn reduce(v: f32, lid: u32) -> f32 {
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

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let row = wid.x + wid.y * nwg.x;
  if (p.rows_dyn != 0u && row >= dyn.b * p.rows_dyn) { return; }
  let rb = row * p.N;
  var s = 0.0;
  for (var c = lid; c < p.N; c += 256u) { s += x[rb + c]; }
  let mean = reduce(s, lid) / f32(p.N);
  var s2 = 0.0;
  for (var c = lid; c < p.N; c += 256u) { let d = x[rb + c] - mean; s2 += d * d; }
  let scale = inverseSqrt(reduce(s2, lid) / f32(p.N) + p.eps);
  for (var c = lid; c < p.N; c += 256u) {
    y[rb + c] = act((x[rb + c] - mean) * scale * w[c] + b[c], p.act, 0.0);
  }
}

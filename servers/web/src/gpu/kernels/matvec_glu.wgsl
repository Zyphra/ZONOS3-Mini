// y[o] = (Wu[o] . x) * silu(Wg[o] . x)
struct P { O: u32, I: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wu: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> Wg: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> ru: array<f32, 256>;
var<workgroup> rg: array<f32, 256>;

fn dot8(w: vec4<u32>, xb: u32) -> f32 {
  let a = unpack2x16float(w.x); let b = unpack2x16float(w.y);
  let c = unpack2x16float(w.z); let d = unpack2x16float(w.w);
  return a.x * x[xb] + a.y * x[xb + 1u] + b.x * x[xb + 2u] + b.y * x[xb + 3u]
       + c.x * x[xb + 4u] + c.y * x[xb + 5u] + d.x * x[xb + 6u] + d.y * x[xb + 7u];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let lane = lid & 31u;
  let o = (wid.x + wid.y * nwg.x) * 8u + (lid >> 5u);
  var su = 0.0;
  var sg = 0.0;
  if (o < p.O) {
    let n8 = p.I / 8u;
    for (var j = lane; j < n8; j += 32u) {
      su += dot8(Wu[o * n8 + j], j * 8u);
      sg += dot8(Wg[o * n8 + j], j * 8u);
    }
  }
  ru[lid] = su;
  rg[lid] = sg;
  workgroupBarrier();
  for (var st = 16u; st > 0u; st >>= 1u) {
    if (lane < st) { ru[lid] += ru[lid + st]; rg[lid] += rg[lid + st]; }
    workgroupBarrier();
  }
  if (lane == 0u && o < p.O) { y[o] = ru[lid] * silu(rg[lid]); }
}

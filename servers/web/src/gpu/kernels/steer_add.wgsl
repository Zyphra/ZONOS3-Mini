// Hidden-space steering (zonos3-graph.cpp route): x[x_off + i] += d[d_off + i], i < n, when dyn.b != 0 (per-step gate).
struct P { n: u32, x_off: u32, d_off: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> d: array<f32>;
@group(0) @binding(14) var<uniform> dyn: Dyn;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let i = wid.x * 256u + lid;
  if (dyn.b == 0u || i >= p.n) { return; }
  x[p.x_off + i] += d[p.d_off + i];
}

// Repeat-upsample along time and add: y[t][k] = e[t / s][k] + xf[t][k], n = T * N.
struct P { n: u32, N: u32, s: u32 }
@group(0) @binding(0) var<storage, read> e: array<f32>;
@group(0) @binding(1) var<storage, read> xf: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let i = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (i >= p.n) { return; }
  let t = i / p.N;
  y[i] = e[(t / p.s) * p.N + (i - t * p.N)] + xf[i];
}

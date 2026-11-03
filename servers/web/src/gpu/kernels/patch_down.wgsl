// Patch down (time-major rows): y[t][ch*h + j] = x[t*h + j][ch]; n = T*C*h outputs, C input channels.
struct P { n: u32, C: u32, h: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let i = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (i >= p.n) { return; }
  let row = p.C * p.h;
  let t = i / row;
  let k = i - t * row;
  let ch = k / p.h;
  let j = k - ch * p.h;
  y[i] = x[(t * p.h + j) * p.C + ch];
}

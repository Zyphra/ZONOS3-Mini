// Patch up (time-major rows): y[t*h + j][ch] = x[t][ch*h + j]; n = T*h*C outputs, C output channels.
struct P { n: u32, C: u32, h: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let i = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (i >= p.n) { return; }
  let r = i / p.C;
  let ch = i % p.C;
  let t = r / p.h;
  let j = r % p.h;
  y[i] = x[t * p.C * p.h + ch * p.h + j];
}

// Time subsampling of [T][row] rows: y[t][c] = x[t*s][c], n = T_out * row.
struct P { n: u32, row: u32, s: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let i = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (i >= p.n) { return; }
  let t = i / p.row;
  y[i] = x[t * p.s * p.row + (i - t * p.row)];
}

// Dense stage fusion: acc[t][k] = (S == 0 ? 0 : acc[t][k]) + feat[t][k] * w[S*N + k], n = T * N.
struct P { n: u32, N: u32, S: u32 }
@group(0) @binding(0) var<storage, read> feat: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let i = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (i >= p.n) { return; }
  let v = feat[i] * w[p.S * p.N + i % p.N];
  acc[i] = select(acc[i] + v, v, p.S == 0u);
}

// GDN output gate: outp[t][h][d] = rmsnorm_h(o)[d] * nw[d] * silu(z[t][h][d]), eps 1e-5, z = mixed[1536..2048).
// One 128-thread workgroup per (t, h).
struct P { n: u32 }
@group(0) @binding(0) var<storage, read> o: array<f32>;
@group(0) @binding(1) var<storage, read> mixed: array<f32>;
@group(0) @binding(2) var<storage, read> nw: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) d: u32) {
  let id = wid.x + wid.y * nwg.x;
  if (id >= p.n * 4u) { return; }
  let t = id / 4u;
  let h = id % 4u;
  let idx = t * 512u + h * 128u + d;
  let v = o[idx];
  red[d] = v * v;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st >>= 1u) {
    if (d < st) { red[d] += red[d + st]; }
    workgroupBarrier();
  }
  let rs = inverseSqrt(red[0] / 128.0 + 1e-5);
  outp[idx] = v * rs * nw[d] * silu(mixed[t * 2056u + 1536u + h * 128u + d]);
}

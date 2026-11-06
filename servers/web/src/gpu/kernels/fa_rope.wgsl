// FuseAlign / Dia RoPE, in place on the q and k parts of a fused qkv buffer (rows of `stride` floats; q at q_off, k at
// k_off, each [nh][hd]). rotate_half: (x1, x2) -> (x1 c - x2 s, x2 c + x1 s) with per-row tables cs/sn [n][hd / 2]
// (cos/sin of position * inv_freq; both halves share them).
struct P { n: u32, nh: u32, hd: u32, stride: u32, q_off: u32, k_off: u32 }
@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> cs: array<f32>;
@group(0) @binding(2) var<storage, read> sn: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let hh = p.hd / 2u;
  let per_row = p.nh * hh;
  let id = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (id >= p.n * per_row) { return; }
  let row = id / per_row;
  let rem = id - row * per_row;
  let h = rem / hh;
  let i = rem - h * hh;
  let c = cs[row * hh + i];
  let s = sn[row * hh + i];
  for (var part = 0u; part < 2u; part++) {
    let base = row * p.stride + select(p.q_off, p.k_off, part == 1u) + h * p.hd + i;
    let x1 = qkv[base];
    let x2 = qkv[base + hh];
    qkv[base] = x1 * c - x2 * s;
    qkv[base + hh] = x2 * c + x1 * s;
  }
}

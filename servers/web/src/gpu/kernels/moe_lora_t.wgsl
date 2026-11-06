// LoRA down-projection of the routed experts: tout[t_off + (t*2 + j)*R + o] = D[e][o] . x[t*x_t + j*x_j ..+I),
// e = sel[t][j], o < R. D f16 [64][R][I] (R <= 32, I % 2 == 0, I <= 1536). One workgroup per (token, pick): 8 lanes
// per output row.
struct P { R: u32, I: u32, x_t: u32, x_j: u32, t_off: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<storage, read> D: array<u32>;
@group(0) @binding(3) var<storage, read_write> tout: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> xs: array<f32, 1536>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let t = wid.x;
  let j = wid.y;
  let x0 = t * p.x_t + j * p.x_j;
  for (var i = lid; i < p.I; i += 256u) { xs[i] = x[x0 + i]; }
  workgroupBarrier();
  let o = lid >> 3u;
  let lane = lid & 7u;
  let e = sel[t * 2u + j];
  var s = 0.0;
  if (o < p.R) {
    let i2 = p.I / 2u;
    let row = (e * p.R + o) * i2;
    for (var i = lane; i < i2; i += 8u) {
      let w = unpack2x16float(D[row + i]);
      s += w.x * xs[2u * i] + w.y * xs[2u * i + 1u];
    }
  }
  red[lid] = s;
  workgroupBarrier();
  for (var st = 4u; st > 0u; st >>= 1u) {
    if (lane < st) { red[lid] += red[lid + st]; }
    workgroupBarrier();
  }
  if (lane == 0u && o < p.R) { tout[p.t_off + (t * 2u + j) * p.R + o] = red[lid]; }
}

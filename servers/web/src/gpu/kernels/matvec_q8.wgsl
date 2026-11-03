// matvec.wgsl with q8_0 weights: int8 W[o][i] (2 vec4<u32> per 32-weight block), f32 scale Wsc[o*I/32 + j]; I % 32 == 0.
// y[y_off + o] = act(W[o] . x[x_off:] + bias[o % nb_bias]) + res[o]; 8 rows x 32 lanes per workgroup.
struct P { O: u32, I: u32, x_off: u32, y_off: u32, nb_bias: u32, flags: u32, act: u32, cap: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> res: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(7) var<storage, read> Wsc: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let lane = lid & 31u;
  let o = (wid.x + wid.y * nwg.x) * 8u + (lid >> 5u);
  var s = 0.0;
  if (o < p.O) {
    let nb = p.I / 32u;
    for (var j = lane; j < nb; j += 32u) {
      let blk = o * nb + j;
      let wa = Wq[blk * 2u]; let wb = Wq[blk * 2u + 1u];
      let xb = p.x_off + j * 32u;
      var d = 0.0;
      d += dot(unpack_i8(wa.x), vec4f(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]));
      d += dot(unpack_i8(wa.y), vec4f(x[xb + 4u], x[xb + 5u], x[xb + 6u], x[xb + 7u]));
      d += dot(unpack_i8(wa.z), vec4f(x[xb + 8u], x[xb + 9u], x[xb + 10u], x[xb + 11u]));
      d += dot(unpack_i8(wa.w), vec4f(x[xb + 12u], x[xb + 13u], x[xb + 14u], x[xb + 15u]));
      d += dot(unpack_i8(wb.x), vec4f(x[xb + 16u], x[xb + 17u], x[xb + 18u], x[xb + 19u]));
      d += dot(unpack_i8(wb.y), vec4f(x[xb + 20u], x[xb + 21u], x[xb + 22u], x[xb + 23u]));
      d += dot(unpack_i8(wb.z), vec4f(x[xb + 24u], x[xb + 25u], x[xb + 26u], x[xb + 27u]));
      d += dot(unpack_i8(wb.w), vec4f(x[xb + 28u], x[xb + 29u], x[xb + 30u], x[xb + 31u]));
      s += d * Wsc[blk];
    }
  }
  red[lid] = s;
  workgroupBarrier();
  for (var st = 16u; st > 0u; st >>= 1u) {
    if (lane < st) { red[lid] += red[lid + st]; }
    workgroupBarrier();
  }
  if (lane == 0u && o < p.O) {
    var v = red[lid];
    if ((p.flags & 1u) != 0u) { v += bias[o % p.nb_bias]; }
    v = act(v, p.act, p.cap);
    if ((p.flags & 8u) != 0u) { v += res[o]; }
    y[p.y_off + o] = v;
  }
}

// matvec.wgsl with q8_0 weights: int8 W[o][i] (2 vec4<u32> per 32-weight block), f32 scale Wsc[o*I/32 + j];
// I % 32 == 0, x_off % 4 == 0. y[y_off + o] = act(W[o] . x[x_off:] + bias[o % nb_bias]) + res[o].
// 16 rows per 64-thread workgroup: 4 row-groups x 16 lanes, each row-group computing 4 rows that share every x load;
// lane = quant block (native/cl/kernels/matvec_q8.cl).
struct P { O: u32, I: u32, x_off: u32, y_off: u32, nb_bias: u32, flags: u32, act: u32, cap: f32 }
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> res: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(7) var<storage, read> Wsc: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> red: array<vec4f, 64>;

/** One 32-weight block of row r (quants qa | qb) against x0..x7. */
fn blk32(qa: vec4<u32>, qb: vec4<u32>, x0: vec4f, x1: vec4f, x2: vec4f, x3: vec4f,
         x4: vec4f, x5: vec4f, x6: vec4f, x7: vec4f) -> f32 {
  return dot(unpack_i8(qa.x), x0) + dot(unpack_i8(qa.y), x1) + dot(unpack_i8(qa.z), x2) + dot(unpack_i8(qa.w), x3)
       + dot(unpack_i8(qb.x), x4) + dot(unpack_i8(qb.y), x5) + dot(unpack_i8(qb.z), x6) + dot(unpack_i8(qb.w), x7);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let lane = lid & 15u;
  let rg = lid >> 4u;
  let o0 = (wid.x + wid.y * nwg.x) * 16u + rg * 4u;
  let nb = p.I / 32u;
  let last = p.O - 1u;
  let r0 = min(o0, last) * nb;
  let r1 = min(o0 + 1u, last) * nb;
  let r2 = min(o0 + 2u, last) * nb;
  let r3 = min(o0 + 3u, last) * nb;
  let xb = p.x_off >> 2u;
  var s = vec4f(0.0);
  for (var j = lane; j < nb; j += 16u) {
    let xo = xb + j * 8u;
    let x0 = x[xo]; let x1 = x[xo + 1u]; let x2 = x[xo + 2u]; let x3 = x[xo + 3u];
    let x4 = x[xo + 4u]; let x5 = x[xo + 5u]; let x6 = x[xo + 6u]; let x7 = x[xo + 7u];
    s.x += Wsc[r0 + j] * blk32(Wq[(r0 + j) * 2u], Wq[(r0 + j) * 2u + 1u], x0, x1, x2, x3, x4, x5, x6, x7);
    s.y += Wsc[r1 + j] * blk32(Wq[(r1 + j) * 2u], Wq[(r1 + j) * 2u + 1u], x0, x1, x2, x3, x4, x5, x6, x7);
    s.z += Wsc[r2 + j] * blk32(Wq[(r2 + j) * 2u], Wq[(r2 + j) * 2u + 1u], x0, x1, x2, x3, x4, x5, x6, x7);
    s.w += Wsc[r3 + j] * blk32(Wq[(r3 + j) * 2u], Wq[(r3 + j) * 2u + 1u], x0, x1, x2, x3, x4, x5, x6, x7);
  }
  red[lid] = s;
  workgroupBarrier();
  for (var st = 8u; st > 0u; st >>= 1u) {
    if (lane < st) { red[lid] += red[lid + st]; }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let r = red[lid];
    var v = array<f32, 4>(r.x, r.y, r.z, r.w);
    for (var k = 0u; k < 4u; k++) {
      let o = o0 + k;
      if (o >= p.O) { break; }
      var t = v[k];
      if ((p.flags & 1u) != 0u) { t += bias[o % p.nb_bias]; }
      t = act(t, p.act, p.cap);
      if ((p.flags & 8u) != 0u) { t += res[o]; }
      y[p.y_off + o] = t;
    }
  }
}

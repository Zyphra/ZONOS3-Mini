// y[y_off + o] = act(W[o] . x[x_off:] + bias[o % nb_bias]) + res[o]; W f16 [O][I] (I % 8 == 0), x_off % 4 == 0.
// 16 rows per 64-thread workgroup: 4 row-groups x 16 lanes, each row-group computing 4 rows that share every x load
// (native/cl/kernels/matvec.cl).
struct P { O: u32, I: u32, x_off: u32, y_off: u32, nb_bias: u32, flags: u32, act: u32, cap: f32 }
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Wt: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> res: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> red: array<vec4f, 64>;

fn dot8(w: vec4<u32>, xa: vec4f, xb: vec4f) -> f32 {
  let a = unpack2x16float(w.x); let b = unpack2x16float(w.y);
  let c = unpack2x16float(w.z); let d = unpack2x16float(w.w);
  return dot(vec4f(a, b), xa) + dot(vec4f(c, d), xb);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let lane = lid & 15u;
  let rg = lid >> 4u;
  let o0 = (wid.x + wid.y * nwg.x) * 16u + rg * 4u;
  let n8 = p.I / 8u;
  let last = p.O - 1u;
  let w0 = min(o0, last) * n8;
  let w1 = min(o0 + 1u, last) * n8;
  let w2 = min(o0 + 2u, last) * n8;
  let w3 = min(o0 + 3u, last) * n8;
  let xb = p.x_off >> 2u;
  var s = vec4f(0.0);
  for (var j = lane; j < n8; j += 16u) {
    let xa = x[xb + 2u * j];
    let xc = x[xb + 2u * j + 1u];
    s.x += dot8(Wt[w0 + j], xa, xc);
    s.y += dot8(Wt[w1 + j], xa, xc);
    s.z += dot8(Wt[w2 + j], xa, xc);
    s.w += dot8(Wt[w3 + j], xa, xc);
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

// y[y_off + o] = act(W[o] . x[x_off:] + bias[o % nb_bias]) + res[o]; 8 rows x 32 lanes per workgroup.
struct P { O: u32, I: u32, x_off: u32, y_off: u32, nb_bias: u32, flags: u32, act: u32, cap: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> res: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let lane = lid & 31u;
  let o = (wid.x + wid.y * nwg.x) * 8u + (lid >> 5u);
  var s = 0.0;
  if (o < p.O) {
    let n8 = p.I / 8u;
    let wb = o * n8;
    for (var j = lane; j < n8; j += 32u) {
      let w = Wt[wb + j];
      let xb = p.x_off + j * 8u;
      let a = unpack2x16float(w.x); let b = unpack2x16float(w.y);
      let c = unpack2x16float(w.z); let d = unpack2x16float(w.w);
      s += a.x * x[xb] + a.y * x[xb + 1u] + b.x * x[xb + 2u] + b.y * x[xb + 3u]
         + c.x * x[xb + 4u] + c.y * x[xb + 5u] + d.x * x[xb + 6u] + d.y * x[xb + 7u];
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

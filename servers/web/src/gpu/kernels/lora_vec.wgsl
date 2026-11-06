// Decode-row LoRA side path in one dispatch: y[y_off + o] += U[o] . (D . x), D f16 [K][I], U f16 [O][K].
// Every workgroup recomputes t = D . x (K x I MACs, 256 / K lanes per rank row) before its 256 outputs.
// K a power of two in 8..256; I even, I <= 2048.
struct P { O: u32, I: u32, K: u32, y_off: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> D: array<u32>;
@group(0) @binding(2) var<storage, read> U: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> xs: array<f32, 2048>;
var<workgroup> red: array<f32, 256>;
var<workgroup> ts: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  for (var i = lid; i < p.I; i += 256u) { xs[i] = x[i]; }
  workgroupBarrier();
  let lanes = 256u / p.K;
  let k = lid / lanes;
  let lane = lid - k * lanes;
  let i2 = p.I / 2u;
  var s = 0.0;
  for (var i = lane; i < i2; i += lanes) {
    let w = unpack2x16float(D[k * i2 + i]);
    s += w.x * xs[2u * i] + w.y * xs[2u * i + 1u];
  }
  red[lid] = s;
  workgroupBarrier();
  for (var st = lanes >> 1u; st > 0u; st >>= 1u) {
    if (lane < st) { red[lid] += red[lid + st]; }
    workgroupBarrier();
  }
  if (lane == 0u) { ts[k] = red[lid]; }
  workgroupBarrier();
  let o = wid.x * 256u + lid;
  if (o >= p.O) { return; }
  let k8 = p.K / 8u;
  var acc = 0.0;
  for (var j = 0u; j < k8; j++) {
    let w = U[o * k8 + j];
    let b = j * 8u;
    acc += dot(vec4f(unpack2x16float(w.x), unpack2x16float(w.y)), vec4f(ts[b], ts[b + 1u], ts[b + 2u], ts[b + 3u])) +
           dot(vec4f(unpack2x16float(w.z), unpack2x16float(w.w)), vec4f(ts[b + 4u], ts[b + 5u], ts[b + 6u], ts[b + 7u]));
  }
  y[p.y_off + o] += acc;
}

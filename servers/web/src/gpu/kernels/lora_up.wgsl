// LoRA up-projection added into a linear's output: Y[y_off + r*O + o] += U[o] . t[r*K .. r*K + K), r < n.
// U f16 [O][K] (K % 8 == 0, K <= 256); t [n][K] = down . x (the adapter's down-projection, gemm / matvec).
struct P { O: u32, K: u32, y_off: u32 }
@group(0) @binding(0) var<storage, read> t: array<f32>;
@group(0) @binding(1) var<storage, read> U: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> ts: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let r = wid.y;
  if (lid < p.K) { ts[lid] = t[r * p.K + lid]; }
  workgroupBarrier();
  let o = wid.x * 256u + lid;
  if (o >= p.O) { return; }
  let k8 = p.K / 8u;
  var s = 0.0;
  for (var j = 0u; j < k8; j++) {
    let w = U[o * k8 + j];
    let b = j * 8u;
    s += dot(vec4f(unpack2x16float(w.x), unpack2x16float(w.y)), vec4f(ts[b], ts[b + 1u], ts[b + 2u], ts[b + 3u])) +
         dot(vec4f(unpack2x16float(w.z), unpack2x16float(w.w)), vec4f(ts[b + 4u], ts[b + 5u], ts[b + 6u], ts[b + 7u]));
  }
  Y[p.y_off + r * p.O + o] += s;
}

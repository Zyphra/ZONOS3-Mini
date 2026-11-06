// Logits head of the speech-edit vocabulary over the release head: per codebook block cb of VE = V + AX rows,
// row j < V is the release head's row cb*V + j (base) and row j >= V the grown row cb*AX + j - V (extra); the adapter
// adds U[r] . t (U f16 [NCB*VE][K], t = down . h), then the softcap: logits[r] = cap * tanh((head + U[r] . t) / cap).
struct P { n: u32, V: u32, AX: u32, K: u32, cap: f32 }
@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> extra: array<f32>;
@group(0) @binding(2) var<storage, read> t: array<f32>;
@group(0) @binding(3) var<storage, read> U: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> ts: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  if (lid < p.K) { ts[lid] = t[lid]; }
  workgroupBarrier();
  let r = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (r >= p.n) { return; }
  let ve = p.V + p.AX;
  let cb = r / ve;
  let j = r - cb * ve;
  var s = 0.0;
  if (j < p.V) { s = base[cb * p.V + j]; } else { s = extra[cb * p.AX + j - p.V]; }
  let k8 = p.K / 8u;
  for (var i = 0u; i < k8; i++) {
    let w = U[r * k8 + i];
    let b = i * 8u;
    s += dot(vec4f(unpack2x16float(w.x), unpack2x16float(w.y)), vec4f(ts[b], ts[b + 1u], ts[b + 2u], ts[b + 3u])) +
         dot(vec4f(unpack2x16float(w.z), unpack2x16float(w.w)), vec4f(ts[b + 4u], ts[b + 5u], ts[b + 6u], ts[b + 7u]));
  }
  outp[r] = act(s, 5u, p.cap);
}

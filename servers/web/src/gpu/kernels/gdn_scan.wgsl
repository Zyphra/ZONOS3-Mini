// Gated delta rule recurrence (ggml_gated_delta_net, scalar gate). State M[h][j][i] (row j = v index)
// lives in state[(h*128 + j)*128 + i]; one thread owns one row. Per token:
//   M *= decay;  dl[j] = (v[j] - M[j].k) * beta;  M[j] += dl[j] * k;  o[j] = (M[j].q) / sqrt(128).
// 64-thread workgroups never straddle a head, so k and q for the token are staged in workgroup memory.
struct P { n: u32 }
@group(0) @binding(0) var<storage, read> qn: array<f32>;
@group(0) @binding(1) var<storage, read> kn: array<f32>;
@group(0) @binding(2) var<storage, read> conv: array<f32>;
@group(0) @binding(3) var<storage, read> decay: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> state: array<f32>;
@group(0) @binding(6) var<storage, read_write> outp: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> ks: array<f32, 128>;
var<workgroup> qs: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let g = wid.x * 64u + lid;
  let h = g / 128u;
  let j = g % 128u;
  let sb = (h * 128u + j) * 128u;
  var row: array<f32, 128>;
  for (var i = 0u; i < 128u; i++) { row[i] = state[sb + i]; }
  for (var t = 0u; t < p.n; t++) {
    let hb = t * 512u + h * 128u;
    workgroupBarrier();
    ks[lid] = kn[hb + lid];
    ks[lid + 64u] = kn[hb + lid + 64u];
    qs[lid] = qn[hb + lid];
    qs[lid + 64u] = qn[hb + lid + 64u];
    workgroupBarrier();
    let dc = decay[t * 4u + h];
    var sk = 0.0;
    for (var i = 0u; i < 128u; i++) { row[i] *= dc; sk += row[i] * ks[i]; }
    let dl = (conv[t * 1536u + 1024u + h * 128u + j] - sk) * beta[t * 4u + h];
    var o = 0.0;
    for (var i = 0u; i < 128u; i++) { row[i] += dl * ks[i]; o += row[i] * qs[i]; }
    outp[hb + j] = o * 0.08838834764831845;
  }
  for (var i = 0u; i < 128u; i++) { state[sb + i] = row[i]; }
}

// embed.wgsl over the speech-edit vocabulary: the release table E (ten audio tables of V rows, then TV text rows at
// row 10*V) plus the grown rows X (AX per audio table, then the text marker rows at row 10*AX). Audio id j >= V reads
// X[k*AX + j - V]; text id j >= TV reads X[10*AX + j - TV].
struct P { n: u32, V: u32, AX: u32, TV: u32 }
@group(0) @binding(0) var<storage, read> ids: array<i32>;
@group(0) @binding(1) var<storage, read> E: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<u32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let t = wid.y;
  if (t >= p.n) { return; }
  let pr = wid.x * 256u + lid;
  var s = vec2f(0.0);
  for (var k = 0u; k < 11u; k++) {
    let id = u32(ids[t * 11u + k]);
    let lim = select(p.V, p.TV, k == 10u);
    if (id < lim) {
      s += unpack2x16float(E[(k * p.V + id) * 512u + pr]);
    } else {
      s += unpack2x16float(X[(k * p.AX + id - lim) * 512u + pr]);
    }
  }
  outp[t * 1024u + 2u * pr] = s.x;
  outp[t * 1024u + 2u * pr + 1u] = s.y;
}

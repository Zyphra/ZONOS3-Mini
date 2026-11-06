// Per row w of lg [rows][F]: out[out_off + w] = the first index of the row's maximum (torch.argmax), as f32.
// One workgroup per row.
struct P { F: u32, out_off: u32 }
@group(0) @binding(0) var<storage, read> lg: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let w = wid.x;
  var best = -3.4e38;
  var idx = 0xffffffffu;
  for (var f = lid; f < p.F; f += 256u) {
    let v = lg[w * p.F + f];
    if (v > best || idx == 0xffffffffu) { best = v; idx = f; }
  }
  bv[lid] = best;
  bi[lid] = idx;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) {
      let ov = bv[lid + st];
      let oi = bi[lid + st];
      if (oi != 0xffffffffu && (bi[lid] == 0xffffffffu || ov > bv[lid] || (ov == bv[lid] && oi < bi[lid]))) {
        bv[lid] = ov;
        bi[lid] = oi;
      }
    }
    workgroupBarrier();
  }
  if (lid == 0u) { outp[p.out_off + w] = f32(bi[0]); }
}

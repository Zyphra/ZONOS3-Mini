// moe_down.wgsl with q8_0 experts (the merged speech-edit backbone):
// out[t][o] = sum_j wts[t][j] * (Wd[sel[t][j]][o] . hbuf[t][j]); int8 Wd [64][256][1536] (384 u32 per row), f32 scale
// per 32-weight block (48 per row).
@group(0) @binding(0) var<storage, read> hbuf: array<f32>;
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<storage, read> wts: array<f32>;
@group(0) @binding(3) var<storage, read> dq: array<u32>;
@group(0) @binding(4) var<storage, read> ds: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;

var<workgroup> hs: array<f32, 3072>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let t = wid.z;
  for (var m = 0u; m < 12u; m++) { hs[lid + 256u * m] = hbuf[t * 3072u + lid + 256u * m]; }
  workgroupBarrier();
  let r = wid.x * 8u + (lid >> 5u);
  let lane = lid & 31u;
  var total = 0.0;
  for (var j = 0u; j < 2u; j++) {
    let row = sel[t * 2u + j] * 256u + r;
    var s = 0.0;
    for (var u = lane; u < 384u; u += 32u) {
      let hb = j * 1536u + u * 4u;
      let hv = vec4f(hs[hb], hs[hb + 1u], hs[hb + 2u], hs[hb + 3u]);
      s += ds[row * 48u + (u >> 3u)] * dot(unpack_i8(dq[row * 384u + u]), hv);
    }
    red[lid] = s;
    workgroupBarrier();
    for (var st = 16u; st > 0u; st >>= 1u) {
      if (lane < st) { red[lid] += red[lid + st]; }
      workgroupBarrier();
    }
    if (lane == 0u) { total += red[lid] * wts[t * 2u + j]; }
    workgroupBarrier();
  }
  if (lane == 0u) { outp[t * 256u + r] = total; }
}

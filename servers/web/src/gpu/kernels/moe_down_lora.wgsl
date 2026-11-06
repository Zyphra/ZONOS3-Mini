// moe_down.wgsl plus the speech-edit adapter's expert output shift:
// out[t][o] = sum_j wts[t][j] * (Wd[e][o] . hbuf[t][j] + Ud[e][o] . td[t][j]), e = sel[t][j]; td [n][2][16] the
// down-projections of hbuf (moe_lora_t, R = 16), Ud f16 [64][256][16]. Lanes 0..7 of a row add f16 pair `lane`.
@group(0) @binding(0) var<storage, read> hbuf: array<f32>;
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<storage, read> wts: array<f32>;
@group(0) @binding(3) var<storage, read> dq: array<u32>;
@group(0) @binding(4) var<storage, read> ds: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;
@group(0) @binding(6) var<storage, read> td: array<f32>;
@group(0) @binding(7) var<storage, read> U: array<u32>;

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
    let e = sel[t * 2u + j];
    let row = e * 256u + r;
    var s = 0.0;
    for (var u = lane; u < 192u; u += 32u) {
      let hb = j * 1536u + (u >> 2u) * 32u + (u & 3u) * 4u;
      let lo = vec4f(hs[hb], hs[hb + 1u], hs[hb + 2u], hs[hb + 3u]);
      let hi = vec4f(hs[hb + 16u], hs[hb + 17u], hs[hb + 18u], hs[hb + 19u]);
      let q = dq[row * 192u + u];
      s += ds[row * 48u + (u >> 2u)] * (dot(unpack_q4_lo(q), lo) + dot(unpack_q4_hi(q), hi));
    }
    if (lane < 8u) {
      let f = unpack2x16float(U[row * 8u + lane]);
      let tb = (t * 2u + j) * 16u + 2u * lane;
      s += f.x * td[tb] + f.y * td[tb + 1u];
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

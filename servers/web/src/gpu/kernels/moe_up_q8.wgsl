// moe_up.wgsl with q8_0 experts (the merged speech-edit backbone): hbuf[t][j][f] = silu(Wg[e] . z_t) * (Wu[e] . z_t),
// e = sel[t][j]; int8 weights [64][1536][256] (8 u32 per 32-weight block), f32 scale per block.
// Workgroup = 32 rows x 8 lanes; lane l handles quant block l of the row.
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<storage, read> gq: array<u32>;
@group(0) @binding(3) var<storage, read> gs: array<f32>;
@group(0) @binding(4) var<storage, read> uq: array<u32>;
@group(0) @binding(5) var<storage, read> us: array<f32>;
@group(0) @binding(6) var<storage, read_write> hbuf: array<f32>;

var<workgroup> xs: array<f32, 256>;
var<workgroup> rg: array<f32, 256>;
var<workgroup> ru: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let t = wid.z;
  let j = wid.y;
  xs[lid] = x[t * 256u + lid];
  workgroupBarrier();
  let e = sel[t * 2u + j];
  let r = wid.x * 32u + (lid >> 3u);
  let lane = lid & 7u;
  let row = e * 1536u + r;
  let qb = row * 64u + lane * 8u;
  var sg = 0.0;
  var su = 0.0;
  for (var w = 0u; w < 8u; w++) {
    let xl = lane * 32u + w * 4u;
    let xv = vec4f(xs[xl], xs[xl + 1u], xs[xl + 2u], xs[xl + 3u]);
    sg += dot(unpack_i8(gq[qb + w]), xv);
    su += dot(unpack_i8(uq[qb + w]), xv);
  }
  rg[lid] = sg * gs[row * 8u + lane];
  ru[lid] = su * us[row * 8u + lane];
  workgroupBarrier();
  for (var st = 4u; st > 0u; st >>= 1u) {
    if (lane < st) { rg[lid] += rg[lid + st]; ru[lid] += ru[lid + st]; }
    workgroupBarrier();
  }
  if (lane == 0u) { hbuf[(t * 2u + j) * 1536u + r] = ru[lid] * silu(rg[lid]); }
}

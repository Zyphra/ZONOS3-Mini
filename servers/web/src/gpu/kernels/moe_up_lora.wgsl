// moe_up.wgsl plus the speech-edit adapter's expert shifts on the gate/up pre-activations:
// hbuf[t][j][f] = silu(Wg[e] . z_t + Ug[e][f] . tl[t][j][0..16)) * (Wu[e] . z_t + Uu[e][f] . tl[t][j][16..32)),
// e = sel[t][j]; tl = the stacked down-projections (moe_lora_t, R = 32: gate rows then up rows), read from hbuf at
// float tl_off (past the n x 2 x 1536 outputs: eight storage bindings at most); U f16 [64][2][1536][16] (gate, up).
// Lane l of a row adds f16 pair l of each factor row.
struct P { tl_off: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<storage, read> gq: array<u32>;
@group(0) @binding(3) var<storage, read> gs: array<f32>;
@group(0) @binding(4) var<storage, read> uq: array<u32>;
@group(0) @binding(5) var<storage, read> us: array<f32>;
@group(0) @binding(6) var<storage, read_write> hbuf: array<f32>;
@group(0) @binding(7) var<storage, read> U: array<u32>;
@group(0) @binding(15) var<uniform> p: P;

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
  let qb = row * 32u + lane * 4u;
  var sg = 0.0;
  var su = 0.0;
  for (var w = 0u; w < 4u; w++) {
    let xl = lane * 32u + w * 4u;
    let xlo = vec4f(xs[xl], xs[xl + 1u], xs[xl + 2u], xs[xl + 3u]);
    let xhi = vec4f(xs[xl + 16u], xs[xl + 17u], xs[xl + 18u], xs[xl + 19u]);
    let g = gq[qb + w]; let u = uq[qb + w];
    sg += dot(unpack_q4_lo(g), xlo) + dot(unpack_q4_hi(g), xhi);
    su += dot(unpack_q4_lo(u), xlo) + dot(unpack_q4_hi(u), xhi);
  }
  let tb = (t * 2u + j) * 32u + 2u * lane;
  let fg = unpack2x16float(U[((e * 2u) * 1536u + r) * 8u + lane]);
  let fu = unpack2x16float(U[((e * 2u + 1u) * 1536u + r) * 8u + lane]);
  let tg = hbuf[p.tl_off + tb]; let tg1 = hbuf[p.tl_off + tb + 1u];
  let tu = hbuf[p.tl_off + tb + 16u]; let tu1 = hbuf[p.tl_off + tb + 17u];
  rg[lid] = sg * gs[row * 8u + lane] + fg.x * tg + fg.y * tg1;
  ru[lid] = su * us[row * 8u + lane] + fu.x * tu + fu.y * tu1;
  workgroupBarrier();
  for (var st = 4u; st > 0u; st >>= 1u) {
    if (lane < st) { rg[lid] += rg[lid + st]; ru[lid] += ru[lid + st]; }
    workgroupBarrier();
  }
  if (lane == 0u) { hbuf[(t * 2u + j) * 1536u + r] = ru[lid] * silu(rg[lid]); }
}

// Full (non-causal) softmax attention for FuseAlign (head dim 64) and Dia (head dim 128) over n rows of a fused qkv
// buffer: q, k, v at q_off, k_off, v_off of each `stride`-float row, each [nh][hd]; out rows of out_stride floats.
// One workgroup per (bq query rows, head). Each row is split over cpr = hd / 16 threads holding 16 query dims and 16
// output dims in registers; tiles of bk = 512 / hd keys/values are staged in workgroup memory, the score partials
// reduced through it, and every row keeps an online softmax (F32 throughout).
struct P { n: u32, nh: u32, hd: u32, stride: u32, q_off: u32, k_off: u32, v_off: u32, out_stride: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

const DPT = 16u;
var<workgroup> Ks: array<f32, 512>;    // [bk][hd]
var<workgroup> Vs: array<f32, 512>;    // [bk][hd]
var<workgroup> Ps: array<f32, 2048>;   // [bq][cpr][bk] score partials
var<workgroup> Ss: array<f32, 512>;    // [bq][bk] scores

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let cpr = p.hd / DPT;
  let bq = 256u / cpr;
  let bk = 512u / p.hd;
  let h = wid.y;
  let r = lid / cpr;
  let c = lid - r * cpr;
  let i = wid.x * bq + r;
  let valid = i < p.n;
  var qr: array<f32, 16>;
  var o: array<f32, 16>;
  for (var d = 0u; d < DPT; d++) { qr[d] = 0.0; o[d] = 0.0; }
  if (valid) {
    let qb = i * p.stride + p.q_off + h * p.hd + c * DPT;
    for (var d = 0u; d < DPT; d++) { qr[d] = qkv[qb + d]; }
  }
  var m = -3.0e38;
  var l = 0.0;
  for (var j0 = 0u; j0 < p.n; j0 += bk) {
    for (var e = lid; e < 512u; e += 256u) {
      let jj = e / p.hd;
      let d = e - jj * p.hd;
      let j = j0 + jj;
      var kv = 0.0;
      var vv = 0.0;
      if (j < p.n) {
        let base = j * p.stride + h * p.hd + d;
        kv = qkv[base + p.k_off];
        vv = qkv[base + p.v_off];
      }
      Ks[e] = kv;
      Vs[e] = vv;
    }
    workgroupBarrier();
    for (var jj = 0u; jj < bk; jj++) {
      let kb = jj * p.hd + c * DPT;
      var acc = 0.0;
      for (var d = 0u; d < DPT; d++) { acc += qr[d] * Ks[kb + d]; }
      Ps[(r * cpr + c) * bk + jj] = acc;
    }
    workgroupBarrier();
    for (var jj = c; jj < bk; jj += cpr) {
      var s = 0.0;
      for (var cc = 0u; cc < cpr; cc++) { s += Ps[(r * cpr + cc) * bk + jj]; }
      Ss[r * bk + jj] = select(-3.0e38, s * p.scale, j0 + jj < p.n);
    }
    workgroupBarrier();
    var mt = m;
    for (var jj = 0u; jj < bk; jj++) { mt = max(mt, Ss[r * bk + jj]); }
    let corr = exp(m - mt);
    l *= corr;
    for (var d = 0u; d < DPT; d++) { o[d] *= corr; }
    for (var jj = 0u; jj < bk; jj++) {
      let sv = Ss[r * bk + jj];
      if (sv > -1.0e38) {
        let pe = exp(sv - mt);
        l += pe;
        let vb = jj * p.hd + c * DPT;
        for (var d = 0u; d < DPT; d++) { o[d] += pe * Vs[vb + d]; }
      }
    }
    m = mt;
    workgroupBarrier();
  }
  if (valid) {
    let inv = 1.0 / l;
    let ob = i * p.out_stride + h * p.hd + c * DPT;
    for (var d = 0u; d < DPT; d++) { outp[ob + d] = o[d] * inv; }
  }
}

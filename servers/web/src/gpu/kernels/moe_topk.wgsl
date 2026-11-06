// Linear-router top-2 of 64 experts, one 64-thread workgroup per token. Selection on lg - rbias (strict >,
// lowest index wins ties): each thread ranks its expert against all 64 in parallel; ranks 0..1 are picked
// (native/cl/kernels/moe_topk.cl). Weights = softmax over the two selected raw logits.
@group(0) @binding(0) var<storage, read> lg: array<f32>;
@group(0) @binding(1) var<storage, read> rbias: array<f32>;
@group(0) @binding(2) var<storage, read_write> sel: array<u32>;
@group(0) @binding(3) var<storage, read_write> wts: array<f32>;

var<workgroup> lgs: array<f32, 64>;
var<workgroup> scs: array<f32, 64>;
var<workgroup> picked: array<u32, 2>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) j: u32) {
  let t = wid.x;
  let v = lg[t * 64u + j];
  lgs[j] = v;
  scs[j] = v - rbias[j];
  workgroupBarrier();
  let sj = scs[j];
  var rank = 0u;
  for (var e = 0u; e < 64u; e++) {
    let se = scs[e];
    rank += select(0u, 1u, se > sj || (se == sj && e < j));
  }
  if (rank < 2u) { picked[rank] = j; }
  workgroupBarrier();
  if (j == 0u) {
    let p0 = picked[0];
    let p1 = picked[1];
    let l0 = lgs[p0];
    let l1 = lgs[p1];
    let mx = max(l0, l1);
    let e0 = exp(l0 - mx);
    let e1 = exp(l1 - mx);
    sel[t * 2u] = p0;
    sel[t * 2u + 1u] = p1;
    wts[t * 2u] = e0 / (e0 + e1);
    wts[t * 2u + 1u] = e1 / (e0 + e1);
  }
}

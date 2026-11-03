// Linear-router top-2 of 64 experts, one 64-thread workgroup per token. Selection on lg - rbias (strict >,
// lowest index wins ties); weights = softmax over the selected raw logits.
@group(0) @binding(0) var<storage, read> lg: array<f32>;
@group(0) @binding(1) var<storage, read> rbias: array<f32>;
@group(0) @binding(2) var<storage, read_write> sel: array<u32>;
@group(0) @binding(3) var<storage, read_write> wts: array<f32>;

var<workgroup> lgs: array<f32, 64>;
var<workgroup> scs: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) j: u32) {
  let t = wid.x;
  let v = lg[t * 64u + j];
  lgs[j] = v;
  scs[j] = v - rbias[j];
  workgroupBarrier();
  if (j == 0u) {
    var picked: array<u32, 2>;
    for (var k = 0u; k < 2u; k++) {
      var best = 0u;
      var bv = -3.0e38;
      var found = false;
      for (var e = 0u; e < 64u; e++) {
        let taken = k > 0u && picked[0] == e;
        if (!taken && (!found || scs[e] > bv)) { bv = scs[e]; best = e; found = true; }
      }
      picked[k] = best;
    }
    let l0 = lgs[picked[0]];
    let l1 = lgs[picked[1]];
    let mx = max(l0, l1);
    let e0 = exp(l0 - mx);
    let e1 = exp(l1 - mx);
    sel[t * 2u] = picked[0];
    sel[t * 2u + 1u] = picked[1];
    wts[t * 2u] = e0 / (e0 + e1);
    wts[t * 2u + 1u] = e1 / (e0 + e1);
  }
}

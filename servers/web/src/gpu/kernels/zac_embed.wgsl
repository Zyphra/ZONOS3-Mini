// Codec RVQ lookup; idx i32 [10][F], tables f16 pairs (row stride C/2), t < n.
// mode 0 (semantic): outp[y_off + t*C + c] = tables[idx[0][t]][c] + bias[c]
// mode 1 (residual, in place): outp[y_off + t*C + c] += sum_{cb=1..9} tables[(cb-1)*4096 + idx[cb][t]][c] + bias[c]
//   (bias = the sum of the 9 out_proj biases). A missing residual code (idx -1, early partial-frame decode) leaves its
//   codebook's term out: no table row, and its own bias biases[(cb-1)*C + c] comes off the sum. Complete codes take
//   the same operations as without the branch (bit-identical).
struct P { n: u32, C: u32, F: u32, y_off: u32, mode: u32 }
@group(0) @binding(0) var<storage, read> idx: array<i32>;
@group(0) @binding(1) var<storage, read> tables: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<storage, read> biases: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let t = wid.y;
  let pr = wid.x * 256u + lid;
  let hw = p.C >> 1u;
  if (t >= p.n || pr >= hw) { return; }
  let o = p.y_off + t * p.C + 2u * pr;
  var s = vec2f(bias[2u * pr], bias[2u * pr + 1u]);
  if (p.mode == 0u) {
    s += unpack2x16float(tables[u32(idx[t]) * hw + pr]);
  } else {
    var acc = vec2f(0.0);
    for (var cb = 1u; cb < 10u; cb++) {
      let code = idx[cb * p.F + t];
      if (code >= 0) {
        let row = (cb - 1u) * 4096u + u32(code);
        acc += unpack2x16float(tables[row * hw + pr]);
      } else {
        let b = (cb - 1u) * p.C + 2u * pr;
        s -= vec2f(biases[b], biases[b + 1u]);
      }
    }
    s += acc + vec2f(outp[o], outp[o + 1u]);
  }
  outp[o] = s.x;
  outp[o + 1u] = s.y;
}

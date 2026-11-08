// Causal depthwise conv over a history-prefixed input: y[t][c] = act(sum_k w[k][c] * Xf[t + k][c] (+ b[c])).
// rnd bits round to bf16: 1 each input as loaded, 2 the tap sum (before bias), 4 the output.
// rows_dyn != 0: rows at or past dyn.b * rows_dyn valid rows exit.
struct P { T: u32, C: u32, K: u32, has_bias: u32, act: u32, rnd: u32, rows_dyn: u32 }
@group(0) @binding(0) var<storage, read> Xf: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(14) var<uniform> dyn: Dyn;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let i = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (i >= p.T * p.C) { return; }
  let t = i / p.C;
  if (p.rows_dyn != 0u && t >= dyn.b * p.rows_dyn) { return; }
  let ch = i - t * p.C;
  var s = 0.0;
  let rin = (p.rnd & 1u) != 0u;
  for (var k = 0u; k < p.K; k++) {
    let xv = Xf[(t + k) * p.C + ch];
    s += w[k * p.C + ch] * select(xv, bf16r(xv), rin);
  }
  if ((p.rnd & 2u) != 0u) { s = bf16r(s); }
  if (p.has_bias != 0u) { s += b[ch]; }
  var o = act(s, p.act, 0.0);
  if ((p.rnd & 4u) != 0u) { o = bf16r(o); }
  y[i] = o;
}

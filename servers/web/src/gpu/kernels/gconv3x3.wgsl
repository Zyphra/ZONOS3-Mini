// Grouped 3x3 conv, pad 1, channel-last x [T_in][F_in][IC] -> y [T_out][F_out][OC]; w [OC][a][3][3] (kh along F, kw along T):
// y[t][f][oc] = sum_j (sum_kh sum_kw w[oc][j][kh][kw] * x[t*s_t+kw-1][f*s_f+kh-1][(oc/bg)*a + j]), zero outside.
// flags: 1 +bias, 2 *scl, 4 +shf, then act.
struct P {
  T_in: u32, F_in: u32, IC: u32, OC: u32, a: u32, bg: u32, s_f: u32, s_t: u32,
  T_out: u32, F_out: u32, flags: u32, act: u32,
}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> scl: array<f32>;
@group(0) @binding(4) var<storage, read> shf: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let i = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (i >= p.T_out * p.F_out * p.OC) { return; }
  let oc = i % p.OC;
  let r = i / p.OC;
  let f = r % p.F_out;
  let t = r / p.F_out;
  let c0 = (oc / p.bg) * p.a;
  var s = 0.0;
  for (var j = 0u; j < p.a; j++) {
    var term = 0.0;
    for (var kh = 0u; kh < 3u; kh++) {
      let fi = i32(f * p.s_f + kh) - 1;
      if (fi < 0 || fi >= i32(p.F_in)) { continue; }
      for (var kw = 0u; kw < 3u; kw++) {
        let ti = i32(t * p.s_t + kw) - 1;
        if (ti < 0 || ti >= i32(p.T_in)) { continue; }
        term += w[((oc * p.a + j) * 3u + kh) * 3u + kw] * x[(u32(ti) * p.F_in + u32(fi)) * p.IC + c0 + j];
      }
    }
    s += term;
  }
  if ((p.flags & 1u) != 0u) { s += bias[oc]; }
  if ((p.flags & 2u) != 0u) { s *= scl[oc]; }
  if ((p.flags & 4u) != 0u) { s += shf[oc]; }
  y[i] = act(s, p.act, 0.0);
}

// out[t][c] = sum_{k<11} embd_all[k*4098 + ids[t*11+k]][c]   (text table starts at row 40980 = 10*4098)
struct P { n: u32 }
@group(0) @binding(0) var<storage, read> ids: array<i32>;
@group(0) @binding(1) var<storage, read> E: array<u32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let t = wid.y;
  if (t >= p.n) { return; }
  let pr = wid.x * 256u + lid;
  var s = vec2f(0.0);
  for (var k = 0u; k < 11u; k++) {
    let row = k * 4098u + u32(ids[t * 11u + k]);
    s += unpack2x16float(E[row * 512u + pr]);
  }
  outp[t * 1024u + 2u * pr] = s.x;
  outp[t * 1024u + 2u * pr + 1u] = s.y;
}

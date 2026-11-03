// Dense 4-expert MoE mix (speaker encoder): y[t][k] = newgelu(sum_i softmax(lg[t])[i] * e[t][64i + k]), k < 64.
struct P { T: u32 }
@group(0) @binding(0) var<storage, read> lg: array<f32>;
@group(0) @binding(1) var<storage, read> e: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let i = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (i >= p.T * 64u) { return; }
  let t = i / 64u;
  let k = i - t * 64u;
  let l = vec4f(lg[t * 4u], lg[t * 4u + 1u], lg[t * 4u + 2u], lg[t * 4u + 3u]);
  let m = max(max(l.x, l.y), max(l.z, l.w));
  let ex = exp(l - vec4f(m));
  let r = ex * (1.0 / (ex.x + ex.y + ex.z + ex.w));
  let eb = t * 256u + k;
  var v = e[eb] * r.x;
  v += e[eb + 64u] * r.y;
  v += e[eb + 128u] * r.z;
  v += e[eb + 192u] * r.w;
  y[i] = act(v, 6u, 0.0);
}

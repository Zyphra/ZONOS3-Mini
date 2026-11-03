// GDN per-(token, head) prep: q/k l2-normalized as x / sqrt(sum x^2 + 1e-6) (FLA convention);
// decay = exp(-softplus(a + dt_bias) * exp(A_log)), beta = sigmoid(b). One 128-thread workgroup per (t, h).
struct P { n: u32 }
@group(0) @binding(0) var<storage, read> conv: array<f32>;
@group(0) @binding(1) var<storage, read> mixed: array<f32>;
@group(0) @binding(2) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(3) var<storage, read> a_log: array<f32>;
@group(0) @binding(4) var<storage, read_write> qn: array<f32>;
@group(0) @binding(5) var<storage, read_write> kn: array<f32>;
@group(0) @binding(6) var<storage, read_write> decay: array<f32>;
@group(0) @binding(7) var<storage, read_write> beta: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

var<workgroup> red: array<vec2f, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) d: u32) {
  let id = wid.x + wid.y * nwg.x;
  if (id >= p.n * 4u) { return; }
  let t = id / 4u;
  let h = id % 4u;
  let q = conv[t * 1536u + h * 128u + d];
  let k = conv[t * 1536u + 512u + h * 128u + d];
  red[d] = vec2f(q * q, k * k);
  workgroupBarrier();
  for (var st = 64u; st > 0u; st >>= 1u) {
    if (d < st) { red[d] += red[d + st]; }
    workgroupBarrier();
  }
  let s = red[0];
  qn[t * 512u + h * 128u + d] = q / sqrt(s.x + 1e-6);
  kn[t * 512u + h * 128u + d] = k / sqrt(s.y + 1e-6);
  if (d == 0u) {
    let a = mixed[t * 2056u + 2048u + h] + dt_bias[h];
    let b = mixed[t * 2056u + 2052u + h];
    let sp = select(log(1.0 + exp(a)), a, a > 20.0);
    decay[t * 4u + h] = exp(-sp * exp(a_log[h]));
    beta[t * 4u + h] = sigmoid(b);
  }
}

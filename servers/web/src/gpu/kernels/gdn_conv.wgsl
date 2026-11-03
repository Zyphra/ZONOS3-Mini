// GDN causal depthwise conv (kernel 4) over the raw q|k|v channels of `mixed` with a 3-row history:
// outp[t][c] = silu(h0*w[c][0] + h1*w[c][1] + h2*w[c][2] + x_t*w[c][3]), tap 0 oldest. One thread per
// channel walks the rows; hist[c*3 + r] (r = 0 oldest) ends as the last 3 raw rows (exact for n < 3).
struct P { n: u32, stride: u32 }
@group(0) @binding(0) var<storage, read> mixed: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

const C = 1536u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let c = wid.x * 256u + lid;
  if (c >= C) { return; }
  var h0 = hist[c * 3u];
  var h1 = hist[c * 3u + 1u];
  var h2 = hist[c * 3u + 2u];
  let w0 = w[c * 4u]; let w1 = w[c * 4u + 1u]; let w2 = w[c * 4u + 2u]; let w3 = w[c * 4u + 3u];
  for (var t = 0u; t < p.n; t++) {
    let x = mixed[t * p.stride + c];
    outp[t * C + c] = silu(h0 * w0 + h1 * w1 + h2 * w2 + x * w3);
    h0 = h1; h1 = h2; h2 = x;
  }
  hist[c * 3u] = h0;
  hist[c * 3u + 1u] = h1;
  hist[c * 3u + 2u] = h2;
}

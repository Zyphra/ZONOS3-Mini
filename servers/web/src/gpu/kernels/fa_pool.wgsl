// FuseAlign word mean pooling: y[w][c] = sum over the word's byte rows j in [spans[2w], spans[2w+1]) of
// x[(row0 + j) * dim + c] / count (each term scaled, as the pooling matmul does). One workgroup per word.
struct P { dim: u32, row0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> spans: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let w = wid.x;
  let a = spans[2u * w];
  let b = spans[2u * w + 1u];
  let inv = 1.0 / f32(max(b - a, 1u));
  for (var c = lid; c < p.dim; c += 256u) {
    var s = 0.0;
    for (var j = a; j < b; j++) { s += x[(p.row0 + j) * p.dim + c] * inv; }
    y[w * p.dim + c] = s;
  }
}

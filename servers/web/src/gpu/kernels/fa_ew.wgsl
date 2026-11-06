// FuseAlign elementwise ops over n outputs. op 0 swiglu: y[r][c] = silu(x[r][c]) * x[r][C + c] (x rows of 2C: gate,
// then up); op 1 bf16: y = x rounded to bf16 (the Dia encoder's bf16 output).
struct P { op: u32, n: u32, C: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let i = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (i >= p.n) { return; }
  if (p.op == 0u) {
    let r = i / p.C;
    let c = i - r * p.C;
    let base = r * 2u * p.C + c;
    y[i] = silu(x[base]) * x[base + p.C];
  } else {
    y[i] = bf16r(x[i]);
  }
}

// Repack a q8_0 [K][O][IC] matrix (int8 quants packed 4 per u32, f32 scale per 32-weight block) into the gemm_h8
// layout: f16 [K][IC][Op], Op = ceil32(O), output channels contiguous per input channel, columns o >= O zero
// (native/cl/kernels/repack_h8.cl). One thread per output u32 (two adjacent outputs).
struct P { O: u32, IC: u32, K: u32, words: u32 }
@group(0) @binding(0) var<storage, read> Wq: array<u32>;
@group(0) @binding(1) var<storage, read> Wsc: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<u32>;
@group(0) @binding(15) var<uniform> p: P;

fn q8(e: u32) -> f32 {
  return f32(bitcast<i32>(Wq[e >> 2u] << (24u - 8u * (e & 3u))) >> 24u) * Wsc[e >> 5u];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let i = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (i >= p.words) { return; }
  let half_op = ((p.O + 31u) / 32u) * 16u;
  let o = (i % half_op) * 2u;
  let r = i / half_op;
  let ic = r % p.IC;
  let k = r / p.IC;
  var v0 = 0.0;
  var v1 = 0.0;
  if (o < p.O) { v0 = q8((k * p.O + o) * p.IC + ic); }
  if (o + 1u < p.O) { v1 = q8((k * p.O + o + 1u) * p.IC + ic); }
  outp[i] = pack2x16float(vec2f(v0, v1));
}

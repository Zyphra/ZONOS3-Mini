// op 0 silu_mul: y = a * silu(b); 1 glu_split: y[r][c] = a[r][c] * sigmoid(a[r][C + c]);
// op 2 add: y = a + b; 3 snake: y = a + c[ch] * sin2(b[ch] * a)  (b = alpha, c = inv_alpha);
// op 4 glu_split_bf16: glu_split rounded to bf16 (nn.GLU under BF16 autocast).
struct P { op: u32, n: u32, C: u32, a_off: u32, y_off: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(15) var<uniform> p: P;

// Port of sin2_f32 (dac.cpp): Cody-Waite reduction + Cephes sinf/cosf polynomials.
fn sin2(v: f32) -> f32 {
  let q = round(v * 0.63661977236758134);
  var r = v - q * 1.5703125;
  r -= q * 4.837512969970703125e-4;
  r -= q * 7.54978995489188216e-8;
  let r2 = r * r;
  let s = r + r * r2 * ((-1.9515295891e-4 * r2 + 8.3321608736e-3) * r2 - 1.6666654611e-1);
  let cc = 1.0 - 0.5 * r2 + r2 * r2 * ((2.443315711809948e-5 * r2 - 1.388731625493765e-3) * r2 + 4.166664568298827e-2);
  let t = select(s, cc, (i32(q) & 1) != 0);
  return t * t;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let i = (wid.x + wid.y * nwg.x) * 256u + lid;
  if (i >= p.n) { return; }
  var v = 0.0;
  switch p.op {
    case 0u: { v = a[p.a_off + i] * silu(b[i]); }
    case 1u, 4u: {
      let r = i / p.C; let ch = i - r * p.C;
      let base = p.a_off + r * 2u * p.C;
      v = a[base + ch] * sigmoid(a[base + p.C + ch]);
      if (p.op == 4u) { v = bf16r(v); }
    }
    case 2u: { v = a[p.a_off + i] + b[i]; }
    default: {
      let ch = i % p.C;
      let xv = a[p.a_off + i];
      v = xv + sin2(b[ch] * xv) * c[ch];
    }
  }
  y[p.y_off + i] = v;
}

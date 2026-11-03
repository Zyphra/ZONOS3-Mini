// Shared helpers, prepended to every kernel.
struct Dyn { a: u32, b: u32, c: u32, d: u32 }

fn erf_as(x: f32) -> f32 {
  // Abramowitz & Stegun 7.1.26
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * exp(-a * a);
  return select(y, -y, x < 0.0);
}

fn tanh_c(x: f32) -> f32 {
  let c = clamp(x, -15.0, 15.0);
  let e = exp(-2.0 * abs(c));
  let t = (1.0 - e) / (1.0 + e);
  return select(t, -t, c < 0.0);
}

fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn gelu_erf(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_as(x * 0.70710678118654752)); }
fn new_gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + tanh_c(0.7978845608028654 * (x + 0.044715 * (x * x * x)))); }

fn act(x: f32, code: u32, cap: f32) -> f32 {
  switch code {
    case 1u: { return silu(x); }
    case 2u: { return gelu_erf(x); }
    case 3u: { return max(x, 0.0); }
    case 4u: { return tanh_c(x); }
    case 5u: { return cap * tanh_c(x / cap); }
    case 6u: { return new_gelu(x); }
    default: { return x; }
  }
}

// Round to the nearest BF16 value (ties to even), returned as f32: ggml_compute_fp32_to_bf16 for finite x.
fn bf16r(x: f32) -> f32 {
  let u = bitcast<u32>(x);
  return bitcast<f32>((u + 0x7fffu + ((u >> 16u) & 1u)) & 0xffff0000u);
}

fn unpack_i8(w: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(bitcast<i32>(w << 24u) >> 24u),
    f32(bitcast<i32>(w << 16u) >> 24u),
    f32(bitcast<i32>(w << 8u) >> 24u),
    f32(bitcast<i32>(w) >> 24u));
}

// q4_0: u32 word w of a 16-byte block holds elements 4w..4w+3 (low nibbles) and 16+4w..16+4w+3 (high nibbles).
fn unpack_q4_lo(w: u32) -> vec4<f32> {
  return vec4<f32>(f32(w & 15u), f32((w >> 8u) & 15u), f32((w >> 16u) & 15u), f32((w >> 24u) & 15u)) - vec4<f32>(8.0);
}
fn unpack_q4_hi(w: u32) -> vec4<f32> {
  return vec4<f32>(f32((w >> 4u) & 15u), f32((w >> 12u) & 15u), f32((w >> 20u) & 15u), f32(w >> 28u)) - vec4<f32>(8.0);
}

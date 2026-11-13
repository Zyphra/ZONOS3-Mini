// Sampler.sample (src/sampler.ts) plus a frame's cold start on the GPU (src/gpu/sampler.ts GpuSampler): one 256-thread
// workgroup per codebook cb samples logits[cb*V .. +V) (CFG: twin + cfg_scale * (logits - twin)). Order as the host
// sampler: the audio pad and the special ids other than stop_id masked (stop_id until generated >= min_frames + cb), the
// stop id's logit raised by stop_bias (LACI's stop boost),
// repetition penalty on every id among the ring's last rep_window entries, greedy argmax (lowest index on ties),
// temperature, top-k (ties at the threshold kept), softmax, top-p on the mass before each token (equal probabilities in
// index order), min-p, then the first kept id in index order whose inclusive prefix mass reaches u[cb] * kept mass
// (no kept mass: the most probable candidate). Codebooks past the stream row get the audio pad (cold start). Writes
// ids[cb] (and ids[ncb] = text_pad: the next decode row) and appends the id to the codebook's ring hist[cb][W]
// (cnt[cb] ids so far). Thread t holds ids [t*C, t*C + C) in registers, C = ceil(V / 256) <= MAXC. Sums run in f32 in
// GPU order, so draws near a boundary may differ from the host sampler's.
struct Smp {
  V: u32, codebook_size: u32, pad_id: u32, stop_id: u32,
  min_frames: u32, generated: u32, row: u32, text_pad: u32,
  greedy: u32, temperature: f32, top_k: u32, top_p: f32,
  min_p: f32, rep_penalty: f32, rep_window: u32, rep_codebooks: u32,
  cfg: u32, cfg_scale: f32, W: u32, neg_inf: u32,
  u: array<vec4<f32>, 3>,
  stop_bias: f32,
}
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> twin: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<u32>;
@group(0) @binding(3) var<storage, read_write> cnt: array<u32>;
@group(0) @binding(4) var<storage, read_write> ids: array<i32>;
@group(0) @binding(5) var<uniform> smp: Smp;

const MAXC = 20u;
const NONE = 0xffffffffu;

var<workgroup> redf: array<f32, 256>;
var<workgroup> redu: array<u32, 256>;
var<workgroup> seen: array<atomic<u32>, 160>;   // repetition penalty bitmap, MAXC * 256 / 32 words
var<workgroup> wg_pick: atomic<u32>;
var<workgroup> wg_last: atomic<u32>;

struct Best { v: f32, i: u32 }

// Order-preserving u32 key of a float (NaN-free).
fn okey(x: f32) -> u32 {
  let b = bitcast<u32>(x);
  return select(b | 0x80000000u, ~b, (b >> 31u) == 1u);
}
fn ounkey(k: u32) -> f32 { return bitcast<f32>(select(~k, k & 0x7fffffffu, (k >> 31u) == 1u)); }

fn sum_f(lid: u32, v: f32) -> f32 {
  redf[lid] = v;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) { redf[lid] += redf[lid + st]; }
    workgroupBarrier();
  }
  let r = redf[0];
  workgroupBarrier();
  return r;
}

fn max_f(lid: u32, v: f32) -> f32 {
  redf[lid] = v;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) { redf[lid] = max(redf[lid], redf[lid + st]); }
    workgroupBarrier();
  }
  let r = redf[0];
  workgroupBarrier();
  return r;
}

fn sum_u(lid: u32, v: u32) -> u32 {
  redu[lid] = v;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) { redu[lid] += redu[lid + st]; }
    workgroupBarrier();
  }
  let r = redu[0];
  workgroupBarrier();
  return r;
}

fn min_u(lid: u32, v: u32) -> u32 {
  redu[lid] = v;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) { redu[lid] = min(redu[lid], redu[lid + st]); }
    workgroupBarrier();
  }
  let r = redu[0];
  workgroupBarrier();
  return r;
}

// Largest v, lowest i on ties.
fn best(lid: u32, v: f32, i: u32) -> Best {
  redf[lid] = v;
  redu[lid] = i;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (lid < st) {
      let ov = redf[lid + st];
      let oi = redu[lid + st];
      if (ov > redf[lid] || (ov == redf[lid] && oi < redu[lid])) {
        redf[lid] = ov;
        redu[lid] = oi;
      }
    }
    workgroupBarrier();
  }
  let r = Best(redf[0], redu[0]);
  workgroupBarrier();
  return r;
}

// Exclusive prefix sum over threads.
fn scan_u(lid: u32, v: u32) -> u32 {
  redu[lid] = v;
  workgroupBarrier();
  for (var st = 1u; st < 256u; st <<= 1u) {
    var t = 0u;
    if (lid >= st) { t = redu[lid - st]; }
    workgroupBarrier();
    redu[lid] += t;
    workgroupBarrier();
  }
  var ex = 0u;
  if (lid > 0u) { ex = redu[lid - 1u]; }
  workgroupBarrier();
  return ex;
}

// Exclusive prefix sum over threads and the total.
fn scan_f(lid: u32, v: f32) -> vec2<f32> {
  redf[lid] = v;
  workgroupBarrier();
  for (var st = 1u; st < 256u; st <<= 1u) {
    var t = 0.0;
    if (lid >= st) { t = redf[lid - st]; }
    workgroupBarrier();
    redf[lid] += t;
    workgroupBarrier();
  }
  var ex = 0.0;
  if (lid > 0u) { ex = redf[lid - 1u]; }
  let total = redf[255];
  workgroupBarrier();
  return vec2<f32>(ex, total);
}

// Cold start, the next decode row and the repetition ring (one thread per codebook).
fn emit(cb: u32, ncb: u32, sampled: u32) {
  var id = sampled;
  if (cb > smp.row) { id = smp.pad_id; }
  ids[cb] = i32(id);
  if (cb == 0u) { ids[ncb] = i32(smp.text_pad); }
  let c = cnt[cb];
  hist[cb * smp.W + c % smp.W] = id;
  cnt[cb] = c + 1u;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let cb = wid.x;
  let V = smp.V;
  let C = (V + 255u) / 256u;
  let base = lid * C;
  var n = 0u;
  if (base < V) { n = min(C, V - base); }
  let NEG = bitcast<f32>(smp.neg_inf);
  let off = cb * V;
  let gate = smp.generated < smp.min_frames + cb;
  var l: array<f32, MAXC>;
  for (var j = 0u; j < n; j++) {
    let i = base + j;
    var v = logits[off + i];
    if (smp.cfg != 0u) {
      let t = twin[off + i];
      v = t + smp.cfg_scale * (v - t);
    }
    if (i >= smp.codebook_size && (i == smp.pad_id || i != smp.stop_id || gate)) { v = NEG; }
    if (i == smp.stop_id && smp.stop_bias != 0.0 && !gate) { v = v + smp.stop_bias; }
    l[j] = v;
  }

  // repetition penalty, once per id among the ring's last rep_window entries
  let words = (V + 31u) / 32u;
  for (var w = lid; w < words; w += 256u) { atomicStore(&seen[w], 0u); }
  if (lid == 0u) {
    atomicStore(&wg_pick, NONE);
    atomicStore(&wg_last, 0u);
  }
  workgroupBarrier();
  let penal = smp.rep_penalty != 1.0 && cb < smp.rep_codebooks;
  if (penal) {
    let c = cnt[cb];
    let m = min(c, smp.rep_window);
    for (var k = lid; k < m; k += 256u) {
      let t = hist[cb * smp.W + (c - m + k) % smp.W];
      if (t < smp.codebook_size) { atomicOr(&seen[t >> 5u], 1u << (t & 31u)); }
    }
  }
  workgroupBarrier();
  if (penal) {
    for (var j = 0u; j < n; j++) {
      let i = base + j;
      if (((atomicLoad(&seen[i >> 5u]) >> (i & 31u)) & 1u) != 0u) {
        l[j] = select(l[j] * smp.rep_penalty, l[j] / smp.rep_penalty, l[j] > 0.0);
      }
    }
  }

  if (smp.greedy != 0u) {
    var bv = NEG;
    var bi = NONE;
    for (var j = 0u; j < n; j++) {
      if (bi == NONE || l[j] > bv) { bv = l[j]; bi = base + j; }
    }
    let b = best(lid, bv, bi);
    if (lid == 0u) { emit(cb, nwg.x, b.i); }
    return;
  }

  if (smp.temperature > 0.0) {
    for (var j = 0u; j < n; j++) { l[j] = l[j] / smp.temperature; }
  }

  // top-k: the k-th largest value (largest key with at least k keys >= it), built bit by bit
  var kth = NEG;
  if (smp.top_k > 0u && smp.top_k < V) {
    var T = 0u;
    for (var b = 0u; b < 32u; b++) {
      let probe = T | (0x80000000u >> b);
      var c = 0u;
      for (var j = 0u; j < n; j++) { c += select(0u, 1u, okey(l[j]) >= probe); }
      if (sum_u(lid, c) >= smp.top_k) { T = probe; }
    }
    kth = ounkey(T);
  }
  var cand = 0u;
  var mx = NEG;
  for (var j = 0u; j < n; j++) {
    if (l[j] >= kth) {
      cand |= 1u << j;
      mx = max(mx, l[j]);
    }
  }

  // softmax over the candidates; l becomes p (0 outside them)
  mx = max_f(lid, mx);
  var e = 0.0;
  for (var j = 0u; j < n; j++) {
    var v = 0.0;
    if (((cand >> j) & 1u) != 0u) { v = exp(l[j] - mx); }
    l[j] = v;
    e += v;
  }
  let total = sum_f(lid, e);
  var bv = -1.0;
  var bi = NONE;
  for (var j = 0u; j < n; j++) {
    l[j] = l[j] / total;
    if (((cand >> j) & 1u) != 0u && l[j] > bv) { bv = l[j]; bi = base + j; }
  }
  let top = best(lid, bv, bi);

  var keep = cand;
  // top-p (a token is dropped when the mass ranked before it exceeds top_p): T = the largest key with mass(p > T) >
  // top_p; every candidate with p > T is kept except, at the smallest such probability v, the ties (in index order)
  // ranked past mass(p > v) + rank * v > top_p. Larger levels keep all their ties (their mass is part of mass(p > v)).
  if (smp.top_p > 0.0 && smp.top_p < 1.0) {
    let tp = smp.top_p;
    var T = 0u;
    for (var b = 0u; b < 32u; b++) {
      let probe = T | (0x80000000u >> b);
      var s = 0.0;
      for (var j = 0u; j < n; j++) { if (bitcast<u32>(l[j]) > probe) { s += l[j]; } }
      if (sum_f(lid, s) > tp) { T = probe; }
    }
    var vmin = NONE;
    for (var j = 0u; j < n; j++) {
      let k = bitcast<u32>(l[j]);
      if (k > T && ((cand >> j) & 1u) != 0u) { vmin = min(vmin, k); }
    }
    let vk = min_u(lid, vmin);
    let vs = bitcast<f32>(vk);
    var above = 0.0;
    var ties = 0u;
    for (var j = 0u; j < n; j++) {
      let k = bitcast<u32>(l[j]);
      if (k > vk) { above += l[j]; } else if (k == vk && ((cand >> j) & 1u) != 0u) { ties++; }
    }
    let mv = sum_f(lid, above);
    var rank = scan_u(lid, ties);
    for (var j = 0u; j < n; j++) {
      if (((cand >> j) & 1u) == 0u) { continue; }
      let k = bitcast<u32>(l[j]);
      if (k == vk) {
        if (mv + f32(rank) * vs > tp) { keep &= ~(1u << j); }
        rank++;
      } else if (k < vk) {
        keep &= ~(1u << j);
      }
    }
  }
  if (smp.min_p > 0.0) {
    let thr = smp.min_p * top.v;
    for (var j = 0u; j < n; j++) { if (l[j] < thr) { keep &= ~(1u << j); } }
  }

  // multinomial: the first kept id whose inclusive prefix mass reaches u * (kept mass)
  var own = 0.0;
  var lastk = NONE;
  for (var j = 0u; j < n; j++) {
    if (((keep >> j) & 1u) != 0u) { own += l[j]; lastk = base + j; }
  }
  if (lastk != NONE) { atomicMax(&wg_last, lastk); }
  let sc = scan_f(lid, own);
  let r = sc.y * smp.u[cb / 4u][cb % 4u];
  var acc = sc.x;
  for (var j = 0u; j < n; j++) {
    if (((keep >> j) & 1u) == 0u) { continue; }
    acc += l[j];
    if (acc >= r) {
      atomicMin(&wg_pick, base + j);
      break;
    }
  }
  workgroupBarrier();
  if (lid == 0u) {
    var id = top.i;
    if (sc.y > 0.0) {
      id = atomicLoad(&wg_pick);
      if (id == NONE) { id = atomicLoad(&wg_last); }
    }
    emit(cb, nwg.x, id);
  }
}

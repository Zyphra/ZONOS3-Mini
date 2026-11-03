// Conditioning rows (zonos3-graph.cpp build_graph): the summed embedding of the row at absolute position dyn.a + t is
// replaced by the projected speaker row (pos == speaker_row) or cond_embd.i[bucket_i] (pos == condition_row + i).
// cond (u32 words): speaker_row i32, condition_row i32, n_cond, bucket[3], 0, 0, then the projected speaker row f32[1024].
// Tables are f16 [count_i + 1][1024] packed in u32 (the JS side checks bucket_i <= count_i). One thread per column, grid [4, n].
struct P { n: u32 }
@group(0) @binding(0) var<storage, read_write> emb: array<f32>;
@group(0) @binding(1) var<storage, read> cond: array<u32>;
@group(0) @binding(2) var<storage, read> t0: array<u32>;
@group(0) @binding(3) var<storage, read> t1: array<u32>;
@group(0) @binding(4) var<storage, read> t2: array<u32>;
@group(0) @binding(14) var<uniform> dyn: Dyn;
@group(0) @binding(15) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let t = wid.y;
  if (t >= p.n) { return; }
  let c = wid.x * 256u + lid;
  let pos = i32(dyn.a + t);
  let sr = bitcast<i32>(cond[0]);
  let cr = bitcast<i32>(cond[1]);
  let nc = i32(cond[2]);
  if (sr >= 0 && pos == sr) {
    emb[t * 1024u + c] = bitcast<f32>(cond[8u + c]);
    return;
  }
  if (cr < 0 || pos < cr || pos >= cr + nc) { return; }
  let i = u32(pos - cr);
  let word = cond[3u + i] * 512u + c / 2u;
  var v: vec2f;
  if (i == 0u) { v = unpack2x16float(t0[word]); }
  else if (i == 1u) { v = unpack2x16float(t1[word]); }
  else { v = unpack2x16float(t2[word]); }
  emb[t * 1024u + c] = select(v.x, v.y, (c & 1u) != 0u);
}

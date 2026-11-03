// Alignment-probe host helpers: readout reduce, text-span lookup and window clipping (zonos3-graph.cpp, zonos3.tts.probe).

/** Argmax (first max) and centroid sum(j*p)/max(sum p, 1e-30) in f64 over dist[0..textLen) (zonos3_probe_reduce). */
export function probeReduce(acc: ArrayLike<number>, textLen: number): { argmax: number; centroid: number } {
  if (textLen <= 0) return { argmax: -1, centroid: 0 };
  let argmax = 0, total = 0, moment = 0;
  for (let j = 0; j < textLen; j++) {
    if (acc[j] > acc[argmax]) argmax = j;
    total += acc[j];
    moment += acc[j] * j;
  }
  return { argmax, centroid: Math.fround(moment / Math.max(total, 1e-30)) };
}

/** The BOS..EOS rows (inclusive) of the text column of ids [n][ncb+1] (zonos3_locate_text_span); null when absent
 *  or when a special id (< 4) lies strictly inside the span. */
export function locateTextSpan(ids: Int32Array, ncb = 10, bos = 2, eos = 3): { start: number; len: number } | null {
  const W = ncb + 1, col = W - 1, n = Math.floor(ids.length / W);
  let b = -1;
  for (let r = 0; r < n && b < 0; r++) if (ids[r * W + col] === bos) b = r;
  if (b < 0) return null;
  for (let r = b + 1; r < n; r++) {
    const t = ids[r * W + col];
    if (t === eos) return { start: b, len: r - b + 1 };
    if (t < 4) return null;
  }
  return null;
}

/** An inclusive text window inside [0, textLen) (clip_window). */
export function clipWindow(lo: number, hi: number, textLen: number): [number, number] {
  lo = Math.max(0, Math.min(lo, textLen - 1));
  return [lo, Math.min(textLen - 1, Math.max(hi, lo))];
}

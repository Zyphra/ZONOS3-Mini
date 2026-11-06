// Codec frames of a speech edit (zonos3.tts.edit): the edit prompt's delay-stacked layout, the generated fill of one
// region and the splice of every fill into the clip's codes. Codes are row-major flat Int32Arrays: ordinary frames
// [T][K], prompt rows [n][K + 1] (K audio columns, then the text column).
import type { Span } from './words';

/** The checkpoint ids the layout uses (zonos3.tts.prompt TTSPromptConfig). */
export interface EditLayout { nCodebooks: number; audioPadId: number; textVocab: number }

/** validate_edit_spans: between one and maxSpans ascending, separated spans inside the audio. */
export function validateEditSpans(spans: Span[], totalFrames: number, maxSpans: number): void {
  if (!spans.length || spans.length > maxSpans) throw new Error('Edit requires between one and max_spans regions');
  let previousEnd = -1;
  for (const [start, length] of spans) {
    if (start < 0 || length < 1 || start + length > totalFrames || start <= previousEnd)
      throw new Error('Edit spans must be ascending, separated and within the audio');
    previousEnd = start + length;
  }
}

/** Frame rows of a flat [T][K] buffer. */
function frameCount(frames: Int32Array, k: number): number {
  if (frames.length % k) throw new Error('Invalid codec frame shape');
  return frames.length / k;
}

/** sheared_block: the delay-stacked frames (plus one all-`endId` row when given) followed by K-1 pad rows, text
 *  column = the text pad. */
export function shearedBlock(frames: Int32Array, l: EditLayout, endId: number | null): Int32Array {
  const k = l.nCodebooks, w = k + 1, n = frameCount(frames, k);
  const rows = n + (endId === null ? 0 : 1) + k - 1;
  const out = new Int32Array(rows * w).fill(l.audioPadId);
  for (let r = 0; r < rows; r++) {
    out[r * w + k] = l.textVocab;
    for (let c = 0; c < k; c++) {
      const f = r - c;
      if (f >= 0 && f < n) out[r * w + c] = frames[f * k + c];
      else if (f === n && endId !== null) out[r * w + c] = endId;
    }
  }
  return out;
}

function concat(parts: Int32Array[]): Int32Array {
  const out = new Int32Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/** edit_prompt_tokens: the training edit layout up to the next infill marker: `textRows`, the unmasked context (each
 *  span replaced by its marker, the context ended by EOA), every already generated fill after its marker (ended by
 *  span-end), then marker `fills.length`. */
export function editPromptTokens(textRows: Int32Array, codes: Int32Array, spans: Span[], fills: Int32Array[],
                                 l: EditLayout, spanEndId: number): Int32Array {
  const k = l.nCodebooks;
  if (spanEndId !== l.audioPadId + 1) throw new Error('span_end_id must equal audio_pad_id + 1');
  validateEditSpans(spans, frameCount(codes, k), spans.length);
  if (fills.length >= spans.length) throw new Error('Prompt needs an ungenerated span');
  const marker = (j: number) => {
    const row = new Int32Array(k + 1).fill(l.audioPadId);
    row[k] = l.textVocab + 1 + j;
    return row;
  };
  const parts = [textRows];
  let cursor = 0;
  spans.forEach(([start, length], j) => {
    if (start > cursor) parts.push(shearedBlock(codes.subarray(cursor * k, start * k), l, null));
    parts.push(marker(j));
    cursor = start + length;
  });
  parts.push(shearedBlock(codes.subarray(cursor * k), l, l.audioPadId - 1));
  fills.forEach((fill, j) => parts.push(marker(j), shearedBlock(fill, l, spanEndId)));
  parts.push(marker(fills.length));
  return concat(parts);
}

/** extract_fill: the codec frames of one generated span (`delayedRows`: the generated rows without the finishing
 *  row; `eosFrame`: the aligned stop frame or null) and whether it ended on span-end. Without a stop the incomplete
 *  K-1 delay tail is dropped; the fill is cut before the first frame holding a non-codec token. */
export function extractFill(delayedRows: Int32Array, eosFrame: number | null, audioPadId: number,
                            codebookSizes: number[]): { fill: Int32Array; ended: boolean } {
  const k = codebookSizes.length, h = frameCount(delayedRows, k);
  let n = Math.min(h, Math.max(0, eosFrame ?? h - (k - 1)));
  let truncated = false;
  for (let f = 0; f < n && !truncated; f++) {
    for (let c = 0; c < k; c++) {
      const v = f + c < h ? delayedRows[(f + c) * k + c] : audioPadId;
      if (v < 0 || v >= codebookSizes[c]) {
        truncated = true;
        n = f;
        break;
      }
    }
  }
  const fill = new Int32Array(n * k);
  for (let f = 0; f < n; f++) for (let c = 0; c < k; c++) fill[f * k + c] = delayedRows[(f + c) * k + c];
  return { fill, ended: eosFrame !== null && !truncated };
}

/** splice_edit: the clip's codes with each span replaced by its fill. */
export function spliceEdit(codes: Int32Array, spans: Span[], fills: Int32Array[], nCodebooks: number): Int32Array {
  const k = nCodebooks;
  validateEditSpans(spans, frameCount(codes, k), spans.length);
  if (fills.length !== spans.length) throw new Error('One fill is required per span');
  const parts: Int32Array[] = [];
  let cursor = 0;
  spans.forEach(([start, length], j) => {
    if (fills[j].length % k) throw new Error('Invalid fill shape');
    parts.push(codes.subarray(cursor * k, start * k), fills[j]);
    cursor = start + length;
  });
  parts.push(codes.subarray(cursor * k));
  return concat(parts);
}

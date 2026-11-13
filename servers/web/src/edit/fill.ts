// The model side of a speech edit: one region's span-end-terminated, codes-only generation on the speech-edit backbone
// (servers/ggml/src/server/pyhost.cpp generate_fill + the scheduler's codes_only job) and the whole-clip decode of the
// spliced codes (decode_codes). Ports zonos3.tts.edit_runtime's EditEngine.fill / decode for the in-browser engine.
import type { BackboneApi, CodecApi } from '../api/engine';
import { buildPrompt, CODEBOOK_SIZE, NCB, type ModelHp, type Prompt } from '../prompt';
import type { Steering } from '../request/steer';
import { vocabIds, type Sampling } from '../sampler';
import { buildIdx, BLOCK } from '../schedule';
import { acceptFrame, newSlot, slotRow } from '../slot';
import type { Tokenizer } from '../tokenizer';
import { outputPcm } from '../output';
import { editPromptTokens, extractFill } from './frames';
import type { Span } from './words';

/** One region's generation (zonos3.tts.edit_runtime.FillJob, resolved): the edit prompt of `windowCodes` with `spans`
 *  masked and `fills` already generated, ended by span-end (masked until `minFrames` rows). */
export interface FillJob {
  text: string;                        // the window's target text, tokenized as sent
  sampling: Sampling;                  // seed included
  maxFrames: number;                   // the frame budget (fit to the slot by the caller)
  minFrames: number;
  buckets: number[];                   // resolved condition buckets
  speakerEmbedding: Float32Array | null;
  steering: Steering | null;           // the region's hidden steering (frames count from the fill's first one)
  windowCodes: Int32Array;             // ordinary [frames][K]
  spans: Span[];                       // window-relative
  fills: Int32Array[];                 // earlier regions of the window, ordinary [n][K] each
}

/** The speech-edit prompt of a fill job: [speaker row][condition rows] BOS text EOS (no silence prefill), then the
 *  edit layout up to the next infill marker. Throws the prompt builder's errors. */
export function fillPrompt(tok: Tokenizer, hp: ModelHp, job: FillJob): Prompt {
  if (hp.spanEndId === null) throw new Error('the backbone has no span-end token');
  const base = buildPrompt(tok, job.text, {
    speakerEmbedding: job.speakerEmbedding ?? undefined, buckets: job.buckets, silencePrefill: 0,
  }, hp);
  const ids = editPromptTokens(base.ids, job.windowCodes, job.spans, job.fills,
    { nCodebooks: hp.nCodebooks, audioPadId: hp.audioPad, textVocab: hp.textVocab }, hp.spanEndId);
  return { ...base, ids, nRows: ids.length / (hp.nCodebooks + 1) };
}

/** Runs a fill: prefills `ids` on slot 0, samples with span-end as the stop (EOA never sampled) until the stop's
 *  delay tail or `maxFrames`, and returns the region's codec frames and whether it ended on span-end. The decode that
 *  yields the fill's frame k is steered with the region's delta at frame k (steer then hold counts from frame 0). */
export async function runFill(bb: BackboneApi, hp: ModelHp, prompt: Prompt, job: FillJob,
                              aborted: () => boolean): Promise<{ fill: Int32Array; ended: boolean }> {
  const stopId = vocabIds(hp.audioVocab, { spanEndStop: true }).stopId;
  const slot = newSlot(0);
  await bb.prefill(0, prompt.ids, prompt.cond);
  bb.startSampling({ sp: job.sampling, vocab: { spanEndStop: true }, minFrames: job.minFrames, cfgScale: null });
  let frame = await bb.sampleFirst(slotRow(slot));
  for (;;) {
    if (aborted()) throw new Error('aborted');
    acceptFrame(slot, frame, stopId, job.maxFrames);
    if (slot.done) break;
    frame = (await bb.stepSample(slotRow(slot), job.steering?.at(slot.step) ?? null, null, null)).frame;
  }
  // the finishing row is not part of the fill's rows
  const rows = Int32Array.from(slot.codes.slice(0, Math.max(0, slot.codes.length - NCB)));
  return extractFill(rows, slot.eosFrame >= 0 ? slot.eosFrame : null, hp.audioPad, new Array<number>(NCB).fill(CODEBOOK_SIZE));
}

/** The 44.1 kHz audio (with the codec's output gain) of ordinary codes [frames][K], decoded as one utterance. */
export async function decodeCodes(codec: CodecApi, codes: Int32Array, padId: number): Promise<Float32Array> {
  const n = codes.length / NCB, h = n + NCB - 1;
  const rows = new Int32Array(h * NCB).fill(padId);
  for (let f = 0; f < n; f++) for (let c = 0; c < NCB; c++) rows[(f + c) * NCB + c] = codes[f * NCB + c];
  codec.reset();
  const parts: Float32Array[] = [];
  for (let p0 = 0; p0 < n; p0 += BLOCK) {
    const k = Math.min(BLOCK, n - p0);
    parts.push(await codec.chunk(buildIdx(rows, h, p0, k), p0, k));
  }
  parts.push(await codec.flush());
  const out = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return outputPcm(out, codec.gain);
}

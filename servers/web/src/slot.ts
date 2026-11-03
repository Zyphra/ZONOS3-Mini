// Port of zonos3_slot_sample: sample one frame, track EOS countdown, build the next input row.
import { AUDIO_PAD, NCB, TEXT_PAD } from './prompt';
import type { Sampler } from './sampler';

export interface SlotState {
  prefill: number;          // silence rows already in the audio stream before step 0
  step: number;
  eosFrame: number;         // stream row (prefill included) of the stop, -1 until seen
  countdown: number;
  done: boolean;
  codes: number[];          // flat [H][10]
  nextIds: Int32Array;      // [11]
}

export function newSlot(prefill = 0): SlotState {
  return { prefill, step: 0, eosFrame: -1, countdown: -1, done: false, codes: [], nextIds: new Int32Array(NCB + 1) };
}

export function sampleFrame(s: SlotState, smp: Sampler, logits: Float32Array, maxFrames: number): number[] {
  const frame: number[] = [];
  for (let cb = 0; cb < NCB; cb++) frame.push(smp.sample(logits, cb));
  // cold start: codebooks whose delayed stream has not started are forced to audio pad; r is the stream row
  const r = s.prefill + s.step;
  for (let cb = r + 1; cb < NCB; cb++) frame[cb] = AUDIO_PAD;
  smp.accept(frame);
  for (const c of frame) s.codes.push(c);
  let maxEoa = -1;
  for (let cb = 0; cb < NCB; cb++) if (frame[cb] === smp.stopId) maxEoa = cb;
  if (s.eosFrame < 0 && maxEoa >= 0) { s.eosFrame = Math.max(0, r - maxEoa); s.countdown = NCB + 1; }
  const finished = s.countdown > 0 && --s.countdown === 0;
  for (let cb = 0; cb < NCB; cb++) s.nextIds[cb] = frame[cb];
  s.nextIds[NCB] = TEXT_PAD;
  s.step += 1;
  if (finished || s.step >= maxFrames) s.done = true;
  return frame;
}

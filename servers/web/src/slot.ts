// Port of zonos3_slot_sample's bookkeeping: take one sampled frame (cold-start pads already applied by the sampler),
// track the EOS countdown, build the next input row.
import { NCB, TEXT_PAD } from './prompt';

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

/** The stream row the slot's next frame lands on (the sampler's cold start pads the codebooks past it). */
export function slotRow(s: SlotState): number {
  return s.prefill + s.step;
}

/** Accepts the frame sampled at stream row slotRow(s) (`frame` [>= NCB] ids; codebooks past the row already carry the
 *  audio pad). Returns its first NCB ids. */
export function acceptFrame(s: SlotState, frame: ArrayLike<number>, stopId: number, maxFrames: number): number[] {
  const r = slotRow(s);
  const out = Array.from({ length: NCB }, (_, cb) => frame[cb]);
  for (const c of out) s.codes.push(c);
  let maxEoa = -1;
  for (let cb = 0; cb < NCB; cb++) if (out[cb] === stopId) maxEoa = cb;
  if (s.eosFrame < 0 && maxEoa >= 0) { s.eosFrame = Math.max(0, r - maxEoa); s.countdown = NCB + 1; }
  const finished = s.countdown > 0 && --s.countdown === 0;
  for (let cb = 0; cb < NCB; cb++) s.nextIds[cb] = out[cb];
  s.nextIds[NCB] = TEXT_PAD;
  s.step += 1;
  if (finished || s.step >= maxFrames) s.done = true;
  return out;
}

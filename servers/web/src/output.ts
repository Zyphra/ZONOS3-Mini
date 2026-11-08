// The served PCM's output stage (zonos3.tts.audio.output_pcm, ggml zonos3_soft_limit): the decoder's output gain, then
// the soft limiter whose knee is the generation registry's output_limit_dbfs.
import { OUTPUT_LIMIT_DBFS } from './generation';

const KNEE = Math.fround(10 ** (OUTPUT_LIMIT_DBFS / 20));
const WIDTH = Math.fround(1 - KNEE);

/** `pcm` times `gain`, then the soft limiter: samples at or below the knee k pass unchanged, louder ones become
 *  sign(x) * (k + (1 - k) * tanh((|x| - k) / (1 - k))): slope 1 at the knee, bounded by full scale. Per sample, so
 *  streamed chunks equal the whole audio. A new array. */
export function outputPcm(pcm: Float32Array, gain: number): Float32Array {
  const out = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    const x = gain === 1 ? pcm[i] : Math.fround(pcm[i] * gain);
    const a = Math.abs(x);
    out[i] = a > KNEE ? Math.sign(x) * Math.fround(KNEE + WIDTH * Math.tanh((a - KNEE) / WIDTH)) : x;
  }
  return out;
}

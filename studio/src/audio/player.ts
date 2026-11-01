/**
 * One AudioContext for the whole studio. Plays float32 PCM (whole buffers or live streams), exposes what is
 * playing and where the playhead is as signals, and decodes encoded clips (speaker previews, recordings).
 */
import { signal } from '@preact/signals';

export interface NowPlaying {
  id: string;
  /** AudioContext time at which sample 0 of the clip would have played. */
  t0: number;
  duration: number;
  streaming: boolean;
}

export const nowPlaying = signal<NowPlaying | null>(null);
/** Seconds into the current clip, updated every animation frame while playing. */
export const playhead = signal(0);

let ctx: AudioContext | null = null;
let master: GainNode | null = null;
let sources: AudioBufferSourceNode[] = [];
let raf = 0;
let endTimer = 0;
let onEndCb: (() => void) | null = null;

function audio(): { ctx: AudioContext; out: GainNode } {
  if (!ctx) {
    ctx = new AudioContext({ latencyHint: 'interactive' });
    master = ctx.createGain();
    master.gain.value = 0.9;
    master.connect(ctx.destination);
  }
  if (ctx.state === 'suspended') void ctx.resume();
  return { ctx, out: master! };
}

function tick() {
  const np = nowPlaying.value;
  if (!np || !ctx) return;
  playhead.value = Math.max(0, Math.min(np.duration, ctx.currentTime - np.t0));
  raf = requestAnimationFrame(tick);
}

function begin(np: NowPlaying, onEnd?: () => void) {
  nowPlaying.value = np;
  onEndCb = onEnd ?? null;
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame(tick);
}

function scheduleEnd(at: number) {
  clearTimeout(endTimer);
  const ms = Math.max(0, (at - ctx!.currentTime) * 1000) + 30;
  endTimer = window.setTimeout(() => finish(), ms);
}

function finish() {
  clearTimeout(endTimer);
  cancelAnimationFrame(raf);
  const cb = onEndCb;
  onEndCb = null;
  nowPlaying.value = null;
  playhead.value = 0;
  cb?.();
}

export function stop() {
  for (const s of sources) {
    try {
      s.stop();
    } catch {
      /* already stopped */
    }
  }
  sources = [];
  if (nowPlaying.value) finish();
}

function bufferOf(pcm: Float32Array, sr: number): AudioBuffer {
  const { ctx } = audio();
  const buf = ctx.createBuffer(1, Math.max(1, pcm.length), sr);
  buf.copyToChannel(pcm as Float32Array<ArrayBuffer>, 0);
  return buf;
}

/** Play a clip from `offset` seconds, to its end or to `until` seconds. */
export function play(id: string, pcm: Float32Array, sr: number, offset = 0, onEnd?: () => void, until?: number) {
  stop();
  const { ctx, out } = audio();
  const src = ctx.createBufferSource();
  src.buffer = bufferOf(pcm, sr);
  src.connect(out);
  const start = ctx.currentTime + 0.02;
  const duration = pcm.length / sr;
  const off = Math.max(0, Math.min(offset, duration));
  const end = Math.max(off, Math.min(until ?? duration, duration));
  src.start(start, off, end - off);
  sources = [src];
  begin({ id, t0: start - off, duration, streaming: false }, onEnd);
  scheduleEnd(start + end - off);
}

export function isPlaying(id: string) {
  return nowPlaying.value?.id === id;
}

export function toggle(id: string, pcm: Float32Array | null | undefined, sr: number, onEnd?: () => void) {
  if (isPlaying(id)) stop();
  else if (pcm) play(id, pcm, sr, 0, onEnd);
}

export interface LiveStream {
  push(chunk: Float32Array): void;
  end(): void;
  readonly id: string;
}

/** Start playback of audio that is still arriving; chunks play back to back as they come. */
export function stream(id: string, sr: number): LiveStream {
  stop();
  const { ctx, out } = audio();
  let next = ctx.currentTime + 0.08;
  const t0 = next;
  let ended = false;
  begin({ id, t0, duration: 0, streaming: true });
  const mine = () => nowPlaying.value?.id === id && nowPlaying.value.t0 === t0;
  return {
    id,
    push(chunk) {
      if (!mine() || !chunk.length) return;
      const src = ctx.createBufferSource();
      src.buffer = bufferOf(chunk, sr);
      src.connect(out);
      const at = Math.max(next, ctx.currentTime + 0.01);
      src.start(at);
      next = at + chunk.length / sr;
      sources.push(src);
      nowPlaying.value = { ...nowPlaying.value!, duration: next - t0 };
      if (ended) scheduleEnd(next);
    },
    end() {
      ended = true;
      if (!mine()) return;
      nowPlaying.value = { ...nowPlaying.value!, streaming: false };
      scheduleEnd(next);
    },
  };
}

/** Decode an encoded clip (wav / mp3 / webm …) to mono float32. */
export async function decode(data: ArrayBuffer): Promise<{ pcm: Float32Array; sr: number }> {
  const { ctx } = audio();
  const buf = await ctx.decodeAudioData(data.slice(0));
  const pcm = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < ch.length; i++) pcm[i] += ch[i] / buf.numberOfChannels;
  }
  return { pcm, sr: buf.sampleRate };
}

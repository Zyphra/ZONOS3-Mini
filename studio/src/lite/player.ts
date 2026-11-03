/** Gapless WebAudio playback of float32 PCM as it streams in, with the playhead of what is playing. */
export class StreamPlayer {
  private ctx: AudioContext | null = null;
  private out: GainNode | null = null;
  private next = 0;
  /** Context time of the clip's sample 0 (the playhead is `currentTime - t0`); NaN until the first block is scheduled. */
  private t0 = NaN;
  private pending: Float32Array[] = [];
  private pendingLength = 0;
  private sources = new Set<AudioBufferSourceNode>();

  constructor(private readonly sampleRate: number) {}

  /** Starts a new stream (call from the user's click, so the browser lets the context run). */
  async begin(): Promise<void> {
    this.stop();
    this.ctx ??= new AudioContext();
    if (this.ctx.state !== 'running') await this.ctx.resume();
    this.out = this.ctx.createGain();
    this.out.connect(this.ctx.destination);
    this.next = 0;
    this.t0 = NaN;
  }

  /** Queues a chunk; scheduled in blocks of ≥ 0.1 s so tiny network chunks don't fragment playback. */
  push(pcm: Float32Array): void {
    this.pending.push(pcm);
    this.pendingLength += pcm.length;
    if (this.pendingLength >= this.sampleRate / 10) this.flush();
  }

  /** Schedules what is queued (the end of the stream). */
  flush(): void {
    const { ctx, out } = this;
    if (!ctx || !out || !this.pendingLength) return;
    const buffer = ctx.createBuffer(1, this.pendingLength, this.sampleRate);
    const data = buffer.getChannelData(0);
    let at = 0;
    for (const chunk of this.pending) {
      data.set(chunk, at);
      at += chunk.length;
    }
    this.pending = [];
    this.pendingLength = 0;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(out);
    const start = Math.max(this.next, ctx.currentTime + 0.05);
    source.start(start);
    // An underrun (the stream slower than real time) starts this block late: the clip's clock waits with it.
    if (Number.isNaN(this.t0)) this.t0 = start;
    else this.t0 += start - this.next;
    this.next = start + buffer.duration;
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
  }

  /** Plays a whole clip from `from` seconds (a replay or a seek). */
  async play(pcm: Float32Array, from: number): Promise<void> {
    await this.begin();
    const skip = Math.min(pcm.length - 1, Math.max(0, Math.round(from * this.sampleRate)));
    this.push(pcm.subarray(skip));
    this.flush();
    this.t0 -= skip / this.sampleRate;
  }

  /** Seconds into the clip while something plays (a buffer underrun mid-stream reads as not playing), else null. */
  position(): number | null {
    if (!this.ctx || !this.sources.size || Number.isNaN(this.t0)) return null;
    return Math.max(0, this.ctx.currentTime - this.t0);
  }

  stop(): void {
    for (const source of this.sources) source.stop();
    this.sources.clear();
    this.out?.disconnect();
    this.out = null;
    this.pending = [];
    this.pendingLength = 0;
  }
}

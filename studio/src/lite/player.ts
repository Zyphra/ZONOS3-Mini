/** Gapless WebAudio playback of float32 PCM as it streams in. */
export class StreamPlayer {
  private ctx: AudioContext | null = null;
  private out: GainNode | null = null;
  private next = 0;
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
    this.next = start + buffer.duration;
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
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

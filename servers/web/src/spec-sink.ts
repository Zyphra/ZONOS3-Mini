// The host half of the codec (iSTFT + FIR mixer at 44160 Hz, then the 44.1 kHz resampler) behind one interface:
// ZacStream runs it in its own thread (LocalSpecSink) or hands it to the codec worker (RemoteSpecSink over a
// MessagePort, served by src/codec-worker.ts) so it overlaps the GPU work of the following backbone steps.
import { IstftHead, type IstftParams } from './istft';
import { Resampler } from './resample';

export interface HostParams extends IstftParams { resampleKernel: Float32Array }

/** The host half of the codec: spectra columns in, final 44.1 kHz samples out, strictly in call order. */
export interface SpecSink {
  push(s0: Float32Array, s1: Float32Array, cols: number): Promise<Float32Array>;
  /** iSTFT flush, then resampler flush, concatenated: the utterance's remaining samples. */
  flush(): Promise<Float32Array>;
  reset(): void;
}

/** Messages on the codec port (RemoteSpecSink -> codec worker). `port` hands over a further codec port (its transferred
 *  MessagePort), served alike: one per decoder, each holding its own decoder's host half. */
export type SinkMsg =
  | ({ type: 'init' } & HostParams)
  | { type: 'push'; s0: Float32Array; s1: Float32Array; cols: number }
  | { type: 'flush' }
  | { type: 'reset' }
  | { type: 'port' };

/** A new codec port beside `port` (served by the same codec worker), for a further decoder's RemoteSpecSink. */
export function codecChannel(port: MessagePort): MessagePort {
  const ch = new MessageChannel();
  port.postMessage({ type: 'port' } satisfies SinkMsg, [ch.port2]);
  return ch.port1;
}

/** Replies on the codec port, one per push / flush, in order. */
export type SinkReply = { type: 'pcm'; pcm: Float32Array } | { type: 'error'; message: string };

export class LocalSpecSink implements SpecSink {
  private istft: IstftHead;
  private rs: Resampler;

  constructor(p: HostParams) {
    this.istft = new IstftHead(p);
    this.rs = new Resampler(p.resampleKernel);
  }

  async push(s0: Float32Array, s1: Float32Array, cols: number): Promise<Float32Array> {
    return this.rs.push(this.istft.push(s0, s1, cols));
  }

  async flush(): Promise<Float32Array> {
    const a = this.rs.push(this.istft.flush());
    const z = this.rs.flush();
    const out = new Float32Array(a.length + z.length);
    out.set(a);
    out.set(z, a.length);
    return out;
  }

  reset(): void {
    this.istft.reset();
    this.rs.reset();
  }
}

/** A SpecSink served by the codec worker; `push` transfers the spectra (the caller must not reuse them). */
export class RemoteSpecSink implements SpecSink {
  private fifo: { resolve: (pcm: Float32Array) => void; reject: (e: Error) => void }[] = [];

  constructor(private port: MessagePort, p: HostParams) {
    port.onmessage = (ev: MessageEvent<SinkReply>) => {
      const head = this.fifo.shift();
      if (!head) return;
      const m = ev.data;
      if (m.type === 'pcm') head.resolve(m.pcm);
      else head.reject(new Error('codec worker: ' + m.message));
    };
    this.post({ type: 'init', ...p });
  }

  private post(m: SinkMsg, transfer: Transferable[] = []): void {
    this.port.postMessage(m, transfer);
  }

  private reply(): Promise<Float32Array> {
    const { promise, resolve, reject } = Promise.withResolvers<Float32Array>();
    this.fifo.push({ resolve, reject });
    return promise;
  }

  push(s0: Float32Array, s1: Float32Array, cols: number): Promise<Float32Array> {
    const r = this.reply();
    this.post({ type: 'push', s0, s1, cols }, [s0.buffer, s1.buffer]);
    return r;
  }

  flush(): Promise<Float32Array> {
    const r = this.reply();
    this.post({ type: 'flush' });
    return r;
  }

  reset(): void {
    this.post({ type: 'reset' });
  }
}

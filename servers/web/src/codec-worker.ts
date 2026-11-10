// Codec worker: the host half of the codec (iSTFT + FIR mixer + resampler, src/spec-sink.ts LocalSpecSink) off the
// engine worker's thread. The page hands it one MessagePort ({ type: 'port' }); the engine worker's RemoteSpecSink
// talks to it over the other end, and hands over one more port per further decoder (codecChannel). Requests on a port
// are served strictly in arrival order; push and flush answer once each.
/// <reference lib="webworker" />
import { LocalSpecSink, type SinkMsg, type SinkReply } from './spec-sink';

declare const self: DedicatedWorkerGlobalScope;

function serve(port: MessagePort): void {
  let sink: LocalSpecSink | null = null;
  // init and reset send no reply: a failed init surfaces on the next push / flush instead.
  let initError = 'not initialized';
  let queue: Promise<void> = Promise.resolve();
  const reply = (m: SinkReply, transfer: Transferable[] = []) => port.postMessage(m, transfer);
  const handle = async (m: SinkMsg): Promise<void> => {
    if (m.type === 'init') {
      try {
        sink = new LocalSpecSink(m);
      } catch (e) {
        sink = null;
        initError = e instanceof Error ? e.message : String(e);
      }
      return;
    }
    if (m.type === 'reset') {
      sink?.reset();
      return;
    }
    try {
      if (!sink) throw new Error(initError);
      const pcm = m.type === 'push' ? await sink.push(m.s0, m.s1, m.cols) : await sink.flush();
      reply({ type: 'pcm', pcm }, [pcm.buffer]);
    } catch (e) {
      reply({ type: 'error', message: e instanceof Error ? e.message : String(e) });
    }
  };
  port.onmessage = (ev: MessageEvent<SinkMsg>) => {
    const m = ev.data;
    if (m.type === 'port') {
      if (ev.ports[0]) serve(ev.ports[0]);
      return;
    }
    queue = queue.then(() => handle(m));
  };
}

self.onmessage = (ev: MessageEvent<{ type: 'port' }>) => {
  if (ev.data.type === 'port' && ev.ports[0]) serve(ev.ports[0]);
};

/** happy-dom lacks a few browser APIs the state modules touch at import time. */
if (typeof globalThis.matchMedia !== 'function') {
  globalThis.matchMedia = ((query: string) =>
    ({ matches: false, media: query, onchange: null, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, dispatchEvent: () => false }) as MediaQueryList) as typeof matchMedia;
}

class FakeAudioContext {
  sampleRate = 44100;
  state = 'running';
  currentTime = 0;
  destination = {};
  createGain() {
    return { gain: { value: 1, setValueAtTime() {}, linearRampToValueAtTime() {} }, connect() {}, disconnect() {} };
  }
  createBufferSource() {
    return { buffer: null, connect() {}, disconnect() {}, start() {}, stop() {}, onended: null };
  }
  createBuffer(channels: number, length: number, sampleRate: number) {
    const data = Array.from({ length: channels }, () => new Float32Array(length));
    return { numberOfChannels: channels, length, sampleRate, duration: length / sampleRate, getChannelData: (i: number) => data[i] };
  }
  createAnalyser() {
    return { fftSize: 2048, connect() {}, disconnect() {}, getFloatTimeDomainData(a: Float32Array) { a.fill(0); } };
  }
  createMediaStreamSource() {
    return { connect() {}, disconnect() {} };
  }
  decodeAudioData() {
    return Promise.reject(new Error('no decoder in tests'));
  }
  resume() {
    return Promise.resolve();
  }
  close() {
    return Promise.resolve();
  }
}
// happy-dom ships no AudioContext; the fake above covers what the state modules touch. Lib DOM types see the real class.
const g: Record<string, unknown> = globalThis as unknown as Record<string, unknown>;
if (typeof g.AudioContext !== 'function') g.AudioContext = FakeAudioContext;

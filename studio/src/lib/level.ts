/**
 * A live input level for a microphone stream: an AnalyserNode sampled at ~10 fps, reported as RMS in 0…1
 * (scaled so normal speech fills most of a bar). Shared by the Cast recorder, the Edit recorder and the lite page.
 */
export interface LevelMeter {
  /** Stop sampling and release the audio graph. */
  stop(): void;
}

const FPS = 10;

export function levelMeter(stream: MediaStream, onLevel: (rms: number) => void): LevelMeter {
  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  const source = ctx.createMediaStreamSource(stream);
  source.connect(analyser);
  const buf = new Float32Array(analyser.fftSize);
  let last = 0;
  let raf = 0;
  const tick = (now: number) => {
    raf = requestAnimationFrame(tick);
    if (now - last < 1000 / FPS) return;
    last = now;
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    // Speech sits around −20 dBFS RMS (0.1): ×4 puts it near the top of the bar without clipping quiet rooms to zero.
    onLevel(Math.min(1, Math.sqrt(sum / buf.length) * 4));
  };
  raf = requestAnimationFrame(tick);
  return {
    stop() {
      cancelAnimationFrame(raf);
      source.disconnect();
      void ctx.close();
      onLevel(0);
    },
  };
}

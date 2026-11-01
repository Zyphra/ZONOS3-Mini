import { useEffect, useRef, useState } from 'preact/hooks';
import { nowPlaying, playhead } from '../audio/player';
import { theme } from '../state/theme';

interface Props {
  pcm: Float32Array | null | undefined;
  /** CSS colour (custom properties allowed). */
  color?: string;
  /** Player id whose playhead this waveform shows. */
  playId?: string;
  height?: number;
  busy?: boolean;
  /** Draw against this many samples (e.g. while a stream is still growing). */
  length?: number;
  onSeek?: (fraction: number) => void;
  label?: string;
}

export function Waveform({ pcm, color = 'var(--text)', playId, height = 56, busy, length, onSeek, label }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const [ink, setInk] = useState('currentColor');

  useEffect(() => {
    const el = canvas.current!;
    const ro = new ResizeObserver(([e]) => setWidth(Math.round(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const mode = theme.value;
  useEffect(() => {
    setInk(getComputedStyle(canvas.current!).color);
  }, [color, mode]);

  useEffect(() => {
    const el = canvas.current;
    if (!el || !width) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = width * dpr;
    el.height = height * dpr;
    const g = el.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, height);
    const mid = height / 2;
    g.fillStyle = ink;
    g.globalAlpha = 0.18;
    g.fillRect(0, mid - 0.5, width, 1);
    if (!pcm || !pcm.length) return;
    g.globalAlpha = 0.95;
    const total = Math.max(length ?? pcm.length, pcm.length);
    const bar = 2;
    const stepPx = 3;
    let peak = 0.05;
    for (let i = 0; i < pcm.length; i += 64) peak = Math.max(peak, Math.abs(pcm[i]));
    const gain = Math.min(4, 0.95 / peak);
    for (let x = 0; x < width; x += stepPx) {
      const a = Math.floor((x / width) * total);
      const b = Math.min(pcm.length, Math.floor(((x + stepPx) / width) * total));
      if (a >= pcm.length) break;
      let lo = 0;
      let hi = 0;
      for (let i = a; i < b; i += 4) {
        const v = pcm[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      const top = mid - Math.max(0.5, hi * gain * mid);
      const bottom = mid - Math.min(-0.5, lo * gain * mid);
      g.fillRect(x, top, bar, bottom - top);
    }
  }, [pcm, width, height, ink, length]);

  const np = nowPlaying.value;
  const playing = !!playId && np?.id === playId;
  const pct = playing && np!.duration ? Math.min(100, (playhead.value / np!.duration) * 100) : null;

  return (
    <div
      class={`wave${busy ? ' is-busy' : ''}${onSeek ? ' is-seekable' : ''}`}
      style={{ height: `${height}px`, color }}
      onPointerDown={onSeek ? (e) => {
        const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
        onSeek(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)));
      } : undefined}
      role={label ? 'img' : undefined}
      aria-label={label}
    >
      <canvas ref={canvas} style={{ width: '100%', height: `${height}px` }} />
      {pct !== null && <div class="wave-head" style={{ left: `${pct}%` }} />}
    </div>
  );
}

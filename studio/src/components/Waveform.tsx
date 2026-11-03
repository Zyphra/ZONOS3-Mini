import { useEffect, useRef, useState } from 'preact/hooks';
import { nowPlaying, playhead } from '../audio/player';
import { drawWave } from '../lib/wave';
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
    if (canvas.current && width) drawWave(canvas.current, pcm, width, height, ink, length);
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

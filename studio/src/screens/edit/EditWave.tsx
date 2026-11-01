import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'preact/hooks';
import { theme, token } from '../../state/theme';
import { nowPlaying, playhead } from '../../audio/player';
import type { Word } from '../../edit/regions';
import { clamp } from '../../lib/math';

export interface WaveRegion {
  id: string;
  start: number;
  end: number;
  n: number;
  manual: boolean;
  directed: boolean;
}

interface Props {
  pcm: Float32Array;
  sr: number;
  words: Word[];
  regions: WaveRegion[];
  /** Regions in an edited take (read-only). */
  overlays: [number, number][];
  /** The audio the model listened to (long clips are edited on windows around the regions). */
  contexts?: [number, number][];
  editable: boolean;
  playId: string;
  cursor: number;
  /** Zoom factor over fit-to-width (1 … 64). */
  zoom: number;
  setZoom: (z: number) => void;
  onSeek: (t: number) => void;
  onCreate: (s: number, e: number) => void;
  onEdges: (id: string, s: number, e: number) => void;
  onWord: (i: number) => void;
  focusRegion?: string | null;
  /** Bumped by the parent to zoom to a time range. */
  zoomTo?: { s: number; e: number; n: number } | null;
  /** Editable word timings: blocks with draggable edges. */
  timing?: {
    selected: number | null;
    edge: 'start' | 'end';
    adjusted: (i: number) => boolean;
    /** phase 'begin' / 'move' / 'end' of one drag; `detach` (Alt) opens a gap at a shared boundary. */
    onEdge: (i: number, edge: 'start' | 'end', t: number, phase: 'begin' | 'move' | 'end', detach: boolean) => void;
  };
  /** The contexts are the studio's estimate (before generating). */
  contextsEstimated?: boolean;
}

const H = 150;
const RULER = 16;
const DRAG_PX = 4;
const BLOCK = 256;
const TICKS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60];

/** Per-block min/max of the clip, computed once per clip. */
function blocks(pcm: Float32Array): { min: Float32Array; max: Float32Array } {
  const n = Math.ceil(pcm.length / BLOCK);
  const min = new Float32Array(n);
  const max = new Float32Array(n);
  for (let b = 0; b < n; b++) {
    let lo = 0;
    let hi = 0;
    for (let i = b * BLOCK, end = Math.min(pcm.length, i + BLOCK); i < end; i++) {
      const v = pcm[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    min[b] = lo;
    max[b] = hi;
  }
  return { min, max };
}

/** The contract makes `confidence` optional (null when the caller supplied the timings). */
const isLow = (w: Word) => typeof w.confidence === 'number' && w.confidence < 0.5;
const confText = (w: Word) => (typeof w.confidence === 'number' ? ` · confidence ${w.confidence.toFixed(2)}` : '');

export const fmtTime = (t: number) => {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, '0')}`;
};

export function EditWave(p: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [vw, setVw] = useState(800);
  const [scroll, setScroll] = useState(0);
  const [drag, setDrag] = useState<{ kind: 'new' | 'start' | 'end'; id?: string; a: number; b: number } | null>(null);
  const [wdrag, setWdrag] = useState<{ i: number; edge: 'start' | 'end' } | null>(null);
  const anchor = useRef<{ t: number; x: number } | null>(null);
  const duration = p.pcm.length / p.sr;
  const peaks = useMemo(() => blocks(p.pcm), [p.pcm]);
  const fit = vw / Math.max(0.5, duration);
  const pps = fit * p.zoom;
  const width = Math.max(vw, Math.ceil(duration * pps));
  const mode = theme.value;
  const colors = useMemo(() => ({ wave: token('--text-2'), dim: token('--text-3'), line: token('--line'), text: token('--text-2') }), [mode]);

  useEffect(() => {
    const el = scroller.current!;
    const ro = new ResizeObserver(([e]) => setVw(Math.max(200, Math.round(e.contentRect.width))));
    ro.observe(el);
    let raf = 0;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => setScroll(el.scrollLeft));
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const r = el.getBoundingClientRect();
        const x = e.clientX - r.left;
        anchor.current = { t: (el.scrollLeft + x) / ppsRef.current, x };
        setZoomRef.current(clamp(zoomRef.current * Math.exp(-e.deltaY * 0.0025), 1, 64));
      } else if (zoomRef.current > 1 && Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        e.preventDefault();
        el.scrollLeft += e.deltaY;
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      ro.disconnect();
      el.removeEventListener('scroll', onScroll);
      el.removeEventListener('wheel', onWheel);
    };
  }, []);
  const ppsRef = useRef(pps);
  ppsRef.current = pps;
  const zoomRef = useRef(p.zoom);
  zoomRef.current = p.zoom;
  const setZoomRef = useRef(p.setZoom);
  setZoomRef.current = p.setZoom;

  // Keep the anchored moment under the pointer (or the view centre) across zoom changes.
  useLayoutEffect(() => {
    const el = scroller.current!;
    const a = anchor.current ?? { t: (el.scrollLeft + vw / 2) / ppsRef.current, x: vw / 2 };
    el.scrollLeft = clamp(a.t * pps - a.x, 0, Math.max(0, width - vw));
    anchor.current = null;
    setScroll(el.scrollLeft);
  }, [pps]);

  useEffect(() => {
    const z = p.zoomTo;
    if (!z) return;
    const span = Math.max(0.3, z.e - z.s);
    const zoom = clamp(duration / span, 1, 64);
    anchor.current = { t: (z.s + z.e) / 2, x: vw / 2 };
    p.setZoom(zoom);
    if (zoom === p.zoom) {
      const el = scroller.current!;
      el.scrollLeft = clamp(((z.s + z.e) / 2) * pps - vw / 2, 0, Math.max(0, width - vw));
    }
  }, [p.zoomTo?.n]);

  // Draw the visible window: ruler, then the waveform.
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = Math.round(vw * dpr);
    el.height = Math.round(H * dpr);
    const g = el.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, vw, H);
    const c = colors;
    const t0 = scroll / pps;
    const step = TICKS.find((s) => s * pps >= 70) ?? 120;
    g.font = '10px "JetBrains Mono", ui-monospace, Menlo, monospace';
    g.fillStyle = c.dim;
    g.strokeStyle = c.line;
    g.lineWidth = 1;
    for (let t = Math.floor(t0 / step) * step; t <= t0 + vw / pps; t += step) {
      const x = Math.round(t * pps - scroll) + 0.5;
      g.beginPath();
      g.moveTo(x, RULER - 5);
      g.lineTo(x, H);
      g.globalAlpha = 0.35;
      g.stroke();
      g.globalAlpha = 1;
      g.fillText(step < 1 ? `${t.toFixed(step < 0.5 ? 2 : 1)}s` : fmtTime(t).replace(/\.\d+$/, ''), x + 3, 10);
    }
    const mid = RULER + (H - RULER) / 2;
    const amp = (H - RULER) / 2 - 4;
    let peak = 0.05;
    for (let b = 0; b < peaks.max.length; b += 4) peak = Math.max(peak, peaks.max[b], -peaks.min[b]);
    const gain = Math.min(3, 0.95 / peak);
    g.fillStyle = c.wave;
    const spp = p.sr / pps; // samples per pixel
    for (let x = 0; x < vw; x++) {
      const a = Math.floor((scroll + x) * spp);
      const b = Math.floor((scroll + x + 1) * spp);
      if (a >= p.pcm.length) break;
      let lo = 0;
      let hi = 0;
      if (b - a < BLOCK * 2) {
        for (let i = a; i < Math.min(b + 1, p.pcm.length); i++) {
          const v = p.pcm[i];
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      } else {
        for (let k = Math.floor(a / BLOCK); k < Math.min(peaks.max.length, Math.ceil(b / BLOCK)); k++) {
          if (peaks.min[k] < lo) lo = peaks.min[k];
          if (peaks.max[k] > hi) hi = peaks.max[k];
        }
      }
      const top = mid - Math.max(0.5, hi * gain * amp);
      const bot = mid - Math.min(-0.5, lo * gain * amp);
      g.fillRect(x, top, 1, bot - top);
    }
  }, [peaks, pps, scroll, vw, colors]);

  const np = nowPlaying.value;
  const playingHere = np?.id === p.playId;
  const head = playingHere ? playhead.value : null;
  // Follow the playhead when it leaves the view.
  useEffect(() => {
    if (head === null) return;
    const el = scroller.current!;
    const x = head * pps;
    if (x < el.scrollLeft || x > el.scrollLeft + vw - 24) el.scrollLeft = clamp(x - vw * 0.15, 0, Math.max(0, width - vw));
  }, [head !== null && Math.floor((head * pps) / (vw * 0.5))]);

  const timeAt = (clientX: number) => {
    const r = scroller.current!.getBoundingClientRect();
    return clamp((clientX - r.left + scroller.current!.scrollLeft) / pps, 0, duration);
  };

  const onDown = (e: PointerEvent) => {
    if (e.button !== 0 || (e.target as Element).closest('.ew-word, .ew-edge, .ew-words')) return;
    const el = e.currentTarget as HTMLElement;
    el.setPointerCapture(e.pointerId);
    const x0 = e.clientX;
    const a = timeAt(x0);
    let moved = false;
    const move = (ev: PointerEvent) => {
      if (!moved && Math.abs(ev.clientX - x0) < DRAG_PX) return;
      moved = true;
      if (p.editable) setDrag({ kind: 'new', a, b: timeAt(ev.clientX) });
    };
    const up = (ev: PointerEvent) => {
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      setDrag(null);
      if (moved && p.editable) {
        const b = timeAt(ev.clientX);
        if (Math.abs(b - a) * pps >= DRAG_PX) p.onCreate(Math.min(a, b), Math.max(a, b));
      } else if (!moved) p.onSeek(a);
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
  };

  const onEdge = (e: PointerEvent, r: WaveRegion, which: 'start' | 'end') => {
    if (e.button !== 0 || !p.editable) return;
    e.stopPropagation();
    const el = e.currentTarget as HTMLElement;
    el.setPointerCapture(e.pointerId);
    let cur = { kind: which, id: r.id, a: r.start, b: r.end };
    const move = (ev: PointerEvent) => {
      const t = timeAt(ev.clientX);
      cur = which === 'start' ? { ...cur, a: Math.min(t, r.end - 0.07) } : { ...cur, b: Math.max(t, r.start + 0.07) };
      setDrag(cur);
    };
    const up = () => {
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      setDrag(null);
      if (cur.a !== r.start || cur.b !== r.end) p.onEdges(r.id, cur.a, cur.b);
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
  };

  const onWordEdge = (e: PointerEvent, i: number, edge: 'start' | 'end') => {
    const tm = p.timing;
    if (e.button !== 0 || !tm) return;
    e.stopPropagation();
    e.preventDefault();
    const el = e.currentTarget as HTMLElement;
    el.setPointerCapture(e.pointerId);
    const w0 = p.words[i];
    tm.onEdge(i, edge, edge === 'start' ? w0.start : w0.end, 'begin', e.altKey);
    setWdrag({ i, edge });
    const move = (ev: PointerEvent) => tm.onEdge(i, edge, timeAt(ev.clientX), 'move', ev.altKey);
    const up = (ev: PointerEvent) => {
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      tm.onEdge(i, edge, timeAt(ev.clientX), 'end', ev.altKey);
      setWdrag(null);
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
  };

  const view0 = scroll / pps - 1;
  const view1 = (scroll + vw) / pps + 1;
  const showWords = p.words.filter((w) => w.end >= view0 && w.start <= view1);
  // A label that would run into the previous one shows on hover only.
  const quiet = new Set<Word>();
  let lastRight = -Infinity;
  for (const w of showWords) {
    const left = w.start * pps;
    if (left < lastRight + 3) quiet.add(w);
    else lastRight = left + w.word.length * 6.4 + 4;
  }

  return (
    <div class={`ewave${p.editable ? ' is-editable' : ''}`}>
      <div ref={scroller} class="ew-scroll" onPointerDown={onDown}>
        <div class="ew-inner" style={{ width: `${width}px` }}>
          {(p.contexts ?? []).map(([s, e]) => (
            <div class={`ew-context${p.contextsEstimated ? ' is-estimate' : ''}`} style={{ left: `${s * pps}px`, width: `${Math.max(2, (e - s) * pps)}px` }} title={`Model context ${fmtTime(s)}–${fmtTime(e)}: ${p.contextsEstimated ? 'the stretch the model will hear (estimate)' : 'the audio the model listened to for this edit'}`}>
              <span>{p.contextsEstimated ? 'model context · est.' : 'model context'}</span>
            </div>
          ))}
          <canvas ref={canvas} class="ew-canvas" style={{ width: `${vw}px`, height: `${H}px`, left: `${scroll}px` }} aria-hidden="true" />
          {p.overlays.map(([s, e], i) => (
            <div class="ew-overlay" style={{ left: `${s * pps}px`, width: `${Math.max(2, (e - s) * pps)}px` }} title={`Region ${i + 1} in this take`}>
              <span class="ew-n">{i + 1}</span>
            </div>
          ))}
          {p.regions.map((r) => {
            const d = drag && drag.id === r.id ? drag : null;
            const s = d ? d.a : r.start;
            const e = d ? d.b : r.end;
            return (
              <div class={`ew-region${r.manual ? ' is-manual' : ''}${r.directed ? ' is-directed' : ''}${p.focusRegion === r.id ? ' is-focus' : ''}`} style={{ left: `${s * pps}px`, width: `${Math.max(3, (e - s) * pps)}px` }}>
                <span class="ew-n">{r.n}</span>
                {p.editable && (
                  <>
                    <span class="ew-edge ew-edge-s" onPointerDown={(ev) => onEdge(ev, r, 'start')} title="Drag to move the start" />
                    <span class="ew-edge ew-edge-e" onPointerDown={(ev) => onEdge(ev, r, 'end')} title="Drag to move the end" />
                  </>
                )}
              </div>
            );
          })}
          {drag?.kind === 'new' && <div class="ew-region is-new" style={{ left: `${Math.min(drag.a, drag.b) * pps}px`, width: `${Math.abs(drag.b - drag.a) * pps}px` }} />}
          <div class="ew-cursor" style={{ left: `${p.cursor * pps}px` }} />
          {head !== null && <div class="ew-head" style={{ left: `${head * pps}px` }} />}
          {p.timing ? (
            <div class="ew-words is-timing" role="group" aria-label="Word timings: drag word edges to correct them">
              {showWords.map((w) => {
                const i = p.words.indexOf(w);
                const tm = p.timing!;
                const sel = tm.selected === i;
                const width = Math.max(2, (w.end - w.start) * pps);
                const active = wdrag?.i === i ? wdrag.edge : sel ? tm.edge : null;
                return (
                  <div
                    class={`ew-wblock${sel ? ' is-sel' : ''}${tm.adjusted(i) ? ' is-adjusted' : ''}${isLow(w) ? ' is-low' : ''}${head !== null && head >= w.start && head < w.end ? ' is-now' : ''}`}
                    style={{ left: `${w.start * pps}px`, width: `${width}px` }}
                  >
                    <button
                      class="ew-wlabel"
                      tabIndex={sel ? 0 : -1}
                      aria-pressed={sel}
                      aria-label={`${w.word}, ${w.start.toFixed(2)} to ${w.end.toFixed(2)} seconds${tm.adjusted(i) ? ', adjusted' : ''}${isLow(w) ? ', low confidence' : ''}`}
                      title={`${w.word} · ${fmtTime(w.start)}–${fmtTime(w.end)}${confText(w)}${tm.adjusted(i) ? ' · adjusted' : ''}`}
                      onClick={() => p.onWord(i)}
                    >
                      {width > 14 ? w.word : ''}
                    </button>
                    <span class={`ew-wedge ew-wedge-s${active === 'start' ? ' is-active' : ''}`} onPointerDown={(ev) => onWordEdge(ev, i, 'start')} aria-hidden="true" />
                    <span class={`ew-wedge ew-wedge-e${active === 'end' ? ' is-active' : ''}`} onPointerDown={(ev) => onWordEdge(ev, i, 'end')} aria-hidden="true" />
                  </div>
                );
              })}
              {wdrag &&
                (() => {
                  // The edge actually moving (the parent may redirect an Alt-drag to the neighbour).
                  const k = p.timing!.selected ?? wdrag.i;
                  const w = p.words[k];
                  if (!w) return null;
                  const t = p.timing!.edge === 'start' ? w.start : w.end;
                  return (
                    <div class="ew-readout" style={{ left: `${t * pps}px` }} aria-live="polite">
                      {w.word} {p.timing!.edge === 'start' ? 'start' : 'end'} {fmtTime(t)} · {Math.round((w.end - w.start) * 1000)} ms
                    </div>
                  );
                })()}
            </div>
          ) : (
            <div class="ew-words">
              {showWords.map((w) => {
                const i = p.words.indexOf(w);
                const left = w.start * pps;
                const wpx = Math.max(10, (w.end - w.start) * pps);
                return (
                  <button
                    class={`ew-word${quiet.has(w) ? ' is-quiet' : ''}${isLow(w) ? ' is-low' : ''}${head !== null && head >= w.start && head < w.end ? ' is-now' : ''}`}
                    style={{ left: `${left}px`, maxWidth: quiet.has(w) ? undefined : `${Math.max(wpx, 14, w.word.length * 6.4 + 4)}px` }}
                    title={`${w.word} · ${fmtTime(w.start)}${confText(w)}`}
                    onClick={() => p.onWord(i)}
                    tabIndex={-1}
                  >
                    {w.word}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

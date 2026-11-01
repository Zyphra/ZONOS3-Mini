/**
 * Named moods in the UI: a chip per mood and the measured landmarks on the valence / arousal pad (click to apply at the default dose, drag along the ray from
 * neutral to set strength, arrow keys between neighbouring landmarks).
 */
import { useLayoutEffect, useState } from 'preact/hooks';
import { hue } from '../state/app';
import { zoneOf, type Values } from '../state/engine';
import { moods, type Mood } from '../state/moods';
import type { Binding } from '../screens/cast/QuickControls';
import { drag } from './drag';

const fmt = (v: number) => `×${+v.toFixed(v % 1 ? 1 : 0)}`;

/** Hover text of a mood: what it does, as measured, and its default dose. */
export function moodTitle(m: Mood, v: number): string {
  return `${m.label}${m.effect ? `: ${m.effect}` : ''} · default ${fmt(m.dose)}${v > 0 ? ` · now ${fmt(v)} (click to remove)` : ' (click to add)'}`;
}

/** Toggle a mood at its default dose, leaving the others (moods combine: their doses add up). */
export function toggleMood(b: Binding, m: Mood) {
  const v = b.values[m.control.id] ?? 0;
  b.begin();
  b.set(m.control.id, v > 0 ? 0 : m.dose);
  b.end();
}

function MoodChip({ m, values, onPick }: { m: Mood; values: Values; onPick: (m: Mood) => void }) {
  const v = values[m.control.id] ?? 0;
  return (
    <button class={`chip mood-chip${v > 0 ? ' is-on' : ''} zone-${zoneOf(m.control, v)}`} aria-pressed={v > 0} title={moodTitle(m, v)} onClick={() => onPick(m)}>
      {m.label}
      {v > 0 && <b>{fmt(v)}</b>}
    </button>
  );
}

/** Every mood as a chip, ordered like the pad from most to least aroused. */
export function MoodChips({ values, onPick }: { values: Values; onPick: (m: Mood) => void }) {
  const all = moods.value;
  if (!all.length) return null;
  return (
    <div class="chips mood-chips" style={{ '--fam': hue('mood') }} role="group" aria-label="Moods">
      {all.map((m) => (
        <MoodChip m={m} values={values} onPick={onPick} />
      ))}
    </div>
  );
}

// ── Landmarks on the pad ──────────────────────────────────────────────────────────────────────────

/** Pad fractions (0…1, y down) of a landmark. */
const spot = (m: Mood) => ({ x: (m.at!.valence + 1) / 2, y: (1 - m.at!.arousal) / 2 });

type Side = 'r' | 'l' | 'u' | 'd' | 'dr' | 'dl' | 'hidden';

type Pt = { x: number; y: number };

/**
 * Where each landmark's dot is drawn (pad fractions): its measured spot, kept off the pad's edge, and pushed apart
 * from neighbours closer than `gap` px so each stays legible and clickable (a leader line marks the true spot).
 */
function layout(list: Mood[], w: number, gap: number): Map<string, Pt> {
  const m = 10 / w;
  const clampPt = (p: Pt) => ({ x: Math.min(1 - m, Math.max(m, p.x)), y: Math.min(1 - m, Math.max(m, p.y)) });
  const pts = list.map((x) => clampPt(spot(x)));
  const g = gap / w;
  for (let it = 0; it < 40; it++) {
    let moved = false;
    for (let i = 0; i < pts.length; i++)
      for (let j = i + 1; j < pts.length; j++) {
        let dx = pts[j].x - pts[i].x;
        let dy = pts[j].y - pts[i].y;
        let d = Math.hypot(dx, dy);
        if (d >= g) continue;
        if (d < 1e-6) (dx = 0), (dy = 1), (d = 1);
        const push = (g - Math.hypot(pts[j].x - pts[i].x, pts[j].y - pts[i].y)) / 2 + 1e-4;
        pts[i] = clampPt({ x: pts[i].x - (dx / d) * push, y: pts[i].y - (dy / d) * push });
        pts[j] = clampPt({ x: pts[j].x + (dx / d) * push, y: pts[j].y + (dy / d) * push });
        moved = true;
      }
    if (!moved) break;
  }
  return new Map(list.map((x, i) => [x.control.id, pts[i]]));
}

/** Label sides that keep labels inside the pad and off each other and the dots (greedy, active first). */
function placeLabels(list: Mood[], at: Map<string, Pt>, values: Values, w: number, font: number, blocked: { x: number; y: number; w: number; h: number }[]): Map<string, Side> {
  const out = new Map<string, Side>();
  const h = font + 4;
  const r = 5;
  const boxes = [...blocked, ...list.map((m) => ({ x: at.get(m.control.id)!.x * w - r, y: at.get(m.control.id)!.y * w - r, w: 2 * r, h: 2 * r }))];
  const hit = (a: (typeof boxes)[number]) => a.x < 1 || a.y < 1 || a.x + a.w > w - 1 || a.y + a.h > w - 1 || boxes.some((b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h);
  const order = [...list].sort((a, b) => Number((values[b.control.id] ?? 0) > 0) - Number((values[a.control.id] ?? 0) > 0));
  for (const m of order) {
    const { x, y } = at.get(m.control.id)!;
    const px = x * w;
    const py = y * w;
    const v = values[m.control.id] ?? 0;
    const lw = (m.label.length + (v > 0 ? fmt(v).length + 1 : 0)) * font * 0.6 + 6;
    const tries: [Side, { x: number; y: number; w: number; h: number }][] = [
      ['r', { x: px + r + 1, y: py - h / 2, w: lw, h }],
      ['l', { x: px - r - 1 - lw, y: py - h / 2, w: lw, h }],
      ['u', { x: px - lw / 2, y: py - r - h, w: lw, h }],
      ['d', { x: px - lw / 2, y: py + r, w: lw, h }],
      ['dr', { x: px + r, y: py + r - 2, w: lw, h }],
      ['dl', { x: px - r - lw, y: py + r - 2, w: lw, h }],
    ];
    const ok = tries.find(([, box]) => !hit(box));
    out.set(m.control.id, ok ? ok[0] : 'hidden');
    if (ok) boxes.push(ok[1]);
  }
  return out;
}

/** The pad width in px, kept current. */
export function useWidth(el: { current: HTMLElement | null }): number {
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const node = el.current;
    if (!node) return;
    setW(node.clientWidth);
    const ro = new ResizeObserver(() => setW(node.clientWidth));
    ro.observe(node);
    return () => ro.disconnect();
  }, [el.current]);
  return w;
}

/** Where the ray from neutral through a landmark leaves the pad, as a multiple of the landmark's offset. */
function edgeScale(dx: number, dy: number): number {
  return Math.min(Math.abs(dx) > 1e-6 ? 0.48 / Math.abs(dx) : Infinity, Math.abs(dy) > 1e-6 ? 0.48 / Math.abs(dy) : Infinity);
}

/** The dose for a pointer at (fx, fy): its projection on the mood's ray, in units of the landmark's offset. */
function doseAt(m: Mood, fx: number, fy: number): number {
  const { x, y } = spot(m);
  const lx = x - 0.5;
  const ly = y - 0.5;
  const len2 = lx * lx + ly * ly;
  const t = len2 > 0.0025 ? ((fx - 0.5) * lx + (fy - 0.5) * ly) / len2 : Math.hypot(fx - 0.5, fy - 0.5) / 0.3;
  const c = m.control;
  return Math.round(Math.min(c.range[1], Math.max(0, t * m.dose)) * 10) / 10;
}

/**
 * Every measured mood as a labelled landmark on the pad, the active ones glowing with a strength handle on their
 * ray. Rendered inside the pad element (`pad`), which positions them.
 */
export function MoodLandmarks({ b, pad, width, font, blocked }: { b: Binding; pad: { current: HTMLElement | null }; width: number; font: number; blocked: { x: number; y: number; w: number; h: number }[] }) {
  const list = moods.value.filter((m) => m.at);
  if (!list.length || !width) return null;
  const at = layout(list, width, font + 6);
  const sides = placeLabels(list, at, b.values, width, font, blocked);
  const focusNear = (from: Mood, key: string) => {
    const a = spot(from);
    const dir: Record<string, [number, number]> = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const [ux, uy] = dir[key];
    let best: Mood | null = null;
    let score = Infinity;
    for (const m of list) {
      if (m === from) continue;
      const p = spot(m);
      const along = (p.x - a.x) * ux + (p.y - a.y) * uy;
      if (along <= 1e-6) continue;
      const across = Math.abs((p.x - a.x) * uy - (p.y - a.y) * ux);
      const s = along + 2 * across;
      if (s < score) (score = s), (best = m);
    }
    if (best) pad.current?.querySelector<HTMLElement>(`[data-mood="${best.control.id}"]`)?.focus();
  };
  // Drag along the ray: press on a landmark or its handle, move to set strength (a press without a move toggles).
  const press = (m: Mood, e: PointerEvent, toggleOnClick: boolean) => {
    e.stopPropagation();
    (e.currentTarget as HTMLElement).focus();
    let start: [number, number] | null = null;
    let moved = false;
    drag(
      e,
      (fx, fy) => {
        if (!start) return void (start = [fx, fy]);
        if (!moved && Math.hypot(fx - start[0], fy - start[1]) < 0.025) return;
        if (!moved) (moved = true), b.begin();
        b.set(m.control.id, doseAt(m, fx, fy));
      },
      () => {
        if (moved) b.end();
        else if (toggleOnClick) toggleMood(b, m);
      },
      pad.current ?? undefined,
    );
  };
  const rays = list.filter((m) => (b.values[m.control.id] ?? 0) > 0);
  return (
    <>
      <svg class="mlm-rays" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        {list.map((m) => {
          const t = spot(m);
          const d = at.get(m.control.id)!;
          return Math.hypot(t.x - d.x, t.y - d.y) * width > 2 ? <line x1={t.x * 100} y1={t.y * 100} x2={d.x * 100} y2={d.y * 100} class="mlm-leader" vector-effect="non-scaling-stroke" /> : null;
        })}
        {rays.map((m) => {
          const { x, y } = spot(m);
          const k = edgeScale(x - 0.5, y - 0.5);
          const safeK = Math.min(k, m.control.safe[1] / m.dose);
          return (
            <g>
              <line x1="50" y1="50" x2={50 + (x - 0.5) * 100 * safeK} y2={50 + (y - 0.5) * 100 * safeK} class="mlm-ray" vector-effect="non-scaling-stroke" />
              {safeK < k && <line x1={50 + (x - 0.5) * 100 * safeK} y1={50 + (y - 0.5) * 100 * safeK} x2={50 + (x - 0.5) * 100 * k} y2={50 + (y - 0.5) * 100 * k} class="mlm-ray is-strong" vector-effect="non-scaling-stroke" />}
            </g>
          );
        })}
      </svg>
      {list.map((m) => {
        const v = b.values[m.control.id] ?? 0;
        const { x, y } = at.get(m.control.id)!;
        const side = sides.get(m.control.id) ?? 'hidden';
        return (
          <button
            data-mood={m.control.id}
            class={`mlm side-${side}${v > 0 ? ` is-on zone-${zoneOf(m.control, v)}` : ''}`}
            style={{ left: `${x * 100}%`, top: `${y * 100}%`, '--glow': Math.min(1.4, v / m.dose) }}
            aria-pressed={v > 0}
            aria-label={`${m.label}${v > 0 ? `, ${fmt(v)}` : ''}`}
            title={moodTitle(m, v)}
            onPointerDown={(e) => press(m, e, true)}
            onClick={(e) => e.detail === 0 && toggleMood(b, m)}
            onKeyDown={(e) => {
              if (e.key.startsWith('Arrow')) return void (e.preventDefault(), e.stopPropagation(), focusNear(m, e.key));
              const step = e.key === '+' || e.key === '=' || e.key === ']' ? 0.5 : e.key === '-' || e.key === '[' ? -0.5 : 0;
              if (step) {
                e.preventDefault();
                e.stopPropagation();
                b.begin();
                b.set(m.control.id, Math.min(m.control.range[1], Math.max(0, Math.round((v + step) * 10) / 10)));
                b.end();
              } else if (e.key === 'Delete' || e.key === 'Backspace') {
                e.preventDefault();
                e.stopPropagation();
                if (v) (b.begin(), b.set(m.control.id, 0), b.end());
              }
            }}
          >
            <span class="mlm-dot" />
            <span class="mlm-label">
              {m.label}
              {v > 0 && <b class="mlm-dose"> {fmt(v)}</b>}
            </span>
          </button>
        );
      })}
      {rays.map((m) => {
        const v = b.values[m.control.id] ?? 0;
        const { x, y } = spot(m);
        const k = Math.min(v / m.dose, edgeScale(x - 0.5, y - 0.5));
        const hx = 0.5 + (x - 0.5) * k;
        const hy = 0.5 + (y - 0.5) * k;
        // At about the default dose the handle would cover the landmark: the landmark itself is the handle then.
        const d = at.get(m.control.id)!;
        if (Math.hypot(hx - d.x, hy - d.y) * width < 9) return null;
        return (
          <span
            class={`mlm-handle zone-${zoneOf(m.control, v)}`}
            style={{ left: `${hx * 100}%`, top: `${hy * 100}%` }}
            title={`${m.label} ${fmt(v)}: drag along the line for strength`}
            onPointerDown={(e) => press(m, e, false)}
          >
            <span class="mlm-handle-tag">{fmt(v)}</span>
          </span>
        );
      })}
    </>
  );
}

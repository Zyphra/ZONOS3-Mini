import { useRef } from 'preact/hooks';
import type { ControlSpec } from '../../api';
import { drag } from '../../components/drag';
import { Slider } from '../../components/Slider';
import { clamp } from '../../lib/math';
import { hue, spec } from '../../state/app';
import type { Values } from '../../state/engine';
import { zoneOf } from '../../state/engine';
import { MoodChips, MoodLandmarks, toggleMood, useWidth } from '../../components/Moods';
import { moods, padAxes, type Mood } from '../../state/moods';

/** Value accessors so the same widgets drive the Cast design and Direct's brushes / baseline. */
export interface Binding {
  values: Values;
  begin: () => void;
  set: (id: string, v: number) => void;
  end: () => void;
}

const SCENE_ICON: Record<string, string> = {
  phone: 'M7 3h4l1 5-2.5 1.5a11 11 0 0 0 5 5L16 12l5 1v4a2 2 0 0 1-2 2A16 16 0 0 1 5 5a2 2 0 0 1 2-2z',
  podcast: 'M12 3a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM6 11a6 6 0 0 0 12 0M12 17v4M9 21h6',
  studio: 'M4 9v6M8 6v12M12 3v18M16 7v10M20 10v4',
};

export function RoomAxis({ b, large }: { b: Binding; large?: boolean }) {
  const c = spec.value.ui.controls.find((x) => x.group === 'room');
  if (!c) return null;
  const color = hue('room');
  const v = b.values[c.id] ?? 0;
  const [lo, hi] = c.range;
  const scenes = [
    { id: 'phone', label: c.poles?.[0] ?? 'phone', at: c.safe[0] },
    { id: 'podcast', label: 'podcast', at: Math.round((c.safe[1] / 4) * 10) / 10 },
    { id: 'studio', label: c.poles?.[1] ?? 'studio', at: c.safe[1] },
  ];
  const near = scenes.reduce((a, s) => (Math.abs(s.at - v) < Math.abs(a.at - v) ? s : a), scenes[0]);
  return (
    <div class={`room${large ? ' is-large' : ''}`} style={{ '--fam': color }}>
      <div class="scenes">
        {scenes.map((s) => (
          <button
            class={`room-scene${Math.abs(near.at - v) < 0.6 && near.id === s.id ? ' is-on' : ''}`}
            style={{ left: `${((s.at - lo) / (hi - lo)) * 100}%` }}
            onClick={() => (b.begin(), b.set(c.id, s.at), b.end())}
            title={`${s.label} (${s.at > 0 ? '+' : ''}${s.at})`}
          >
            <svg viewBox="0 0 24 24" width={large ? 34 : 18} height={large ? 34 : 18} aria-hidden="true">
              <path d={SCENE_ICON[s.id]} fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
            <span class="room-scene-label">{s.label}</span>
            {large && <span class="room-scene-at">{s.at > 0 ? '+' : ''}{s.at}</span>}
          </button>
        ))}
      </div>
      {large ? (
        <Slider control={c} value={v} color={color} label="Room" detailed onStart={b.begin} onChange={(x) => b.set(c.id, x)} onEnd={b.end} />
      ) : (
        <Slider control={c} value={v} color={color} label="Room" bare onStart={b.begin} onChange={(x) => b.set(c.id, x)} onEnd={b.end} />
      )}
      {!large && <div class="room-read">
        <span>{c.poles?.[0] ?? 'phone'}</span>
        <b>{v > 0 ? '+' : ''}{v.toFixed(1)}</b>
        <span>{c.poles?.[1] ?? 'studio'}</span>
      </div>}
    </div>
  );
}

export function SliderGroup({ group, b, only, compact, detailed }: { group: string; b: Binding; only?: (c: ControlSpec) => boolean; compact?: boolean; detailed?: boolean }) {
  const color = hue(group);
  const controls = spec.value.ui.controls.filter((c) => c.group === group && (!only || only(c)));
  return (
    <div class={`slider-group${detailed ? ' is-detailed' : ''}`}>
      {controls.map((c) => (
        <Slider key={c.id} control={c} value={b.values[c.id] ?? 0} color={color} compact={compact} detailed={detailed} onStart={b.begin} onChange={(v) => b.set(c.id, v)} onEnd={b.end} />
      ))}
    </div>
  );
}

/**
 * Valence × arousal pad (the first two bipolar mood controls) with every measured mood as a landmark, and a chip
 * per named mood (large: with sliders for the active ones).
 */
export function MoodControls({ b, compact, large }: { b: Binding; compact?: boolean; large?: boolean }) {
  const axes = padAxes.value;
  const cx = axes?.x;
  const cy = axes?.y;
  const all = moods.value;
  const pad = useRef<HTMLDivElement>(null);
  const width = useWidth(pad);
  const color = hue('mood');
  const vx = cx ? (b.values[cx.id] ?? 0) : 0;
  const vy = cy ? (b.values[cy.id] ?? 0) : 0;
  const px = cx ? ((vx - cx.range[0]) / (cx.range[1] - cx.range[0])) * 100 : 50;
  const py = cy ? (1 - (vy - cy.range[0]) / (cy.range[1] - cy.range[0])) * 100 : 50;
  const zx = cx ? zoneOf(cx, vx) : 'safe';
  const zy = cy ? zoneOf(cy, vy) : 'safe';
  const safeBox = cx && cy && {
    left: `${((cx.safe[0] - cx.range[0]) / (cx.range[1] - cx.range[0])) * 100}%`,
    width: `${((cx.safe[1] - cx.safe[0]) / (cx.range[1] - cx.range[0])) * 100}%`,
    top: `${(1 - (cy.safe[1] - cy.range[0]) / (cy.range[1] - cy.range[0])) * 100}%`,
    height: `${((cy.safe[1] - cy.safe[0]) / (cy.range[1] - cy.range[0])) * 100}%`,
  };
  const marked = all.some((m) => m.at);
  // Corner words stay on the large pad (landmark labels keep off them); small pads give the room to landmarks.
  const corners = large || !marked;
  const font = large ? 12 : 9;
  const cornerBoxes = corners && width ? [0, 1].flatMap((i) => [0, 1].map((j) => ({ x: i ? width - 64 : 0, y: j ? width - 20 : 0, w: 64, h: 20 }))) : [];
  // The large pad's pole words sit just above the middle of each side.
  if (large && width) cornerBoxes.push({ x: 0, y: width / 2 - 24, w: 70, h: 20 }, { x: width - 70, y: width / 2 - 24, w: 70, h: 20 });
  const active = all.filter((m) => (b.values[m.control.id] ?? 0) > 0);
  const pick = (m: Mood) => toggleMood(b, m);
  return (
    <div class={`mood${compact ? ' is-compact' : ''}${large ? ' is-large' : ''}${marked ? ' has-landmarks' : ''}`} style={{ '--fam': color }}>
      {cx && cy && (
        <div
          ref={pad}
          class={`mood-pad zone-${zx === 'safe' && zy === 'safe' ? 'safe' : 'strong'}`}
          role="group"
          aria-label={`Mood pad: ${cx.label} (x) and ${cy.label} (y)${marked ? ', with measured moods as landmarks' : ''}`}
          onDblClick={(e) => !(e.target as Element).closest('.mlm, .mlm-handle') && (b.begin(), b.set(cx.id, 0), b.set(cy.id, 0), b.end())}
          onPointerDown={(e) => {
            if ((e.target as Element).closest('.mlm, .mlm-handle')) return;
            b.begin();
            drag(
              e,
              (fx, fy) => {
                b.set(cx.id, Math.round((cx.range[0] + clamp(fx, 0, 1) * (cx.range[1] - cx.range[0])) * 10) / 10);
                b.set(cy.id, Math.round((cy.range[1] - clamp(fy, 0, 1) * (cy.range[1] - cy.range[0])) * 10) / 10);
              },
              b.end,
            );
          }}
        >
          {safeBox && <div class="mood-safe" style={safeBox} />}
          {corners && (
            <>
              <span class="q q-tl">tense</span>
              <span class="q q-tr">excited</span>
              <span class="q q-bl">low</span>
              <span class="q q-br">content</span>
            </>
          )}
          <span class="mood-axis-x" />
          <span class="mood-axis-y" />
          <MoodLandmarks b={b} pad={pad} width={width} font={font} blocked={cornerBoxes} />
          <span
            class="mood-dot"
            style={{ left: `${px}%`, top: `${py}%` }}
            role="slider"
            tabIndex={0}
            aria-label={`${cx.label} (x) and ${cy.label} (y)`}
            aria-valuetext={`${cx.label} ${vx.toFixed(1)}, ${cy.label} ${vy.toFixed(1)}`}
            title={`${cx.label} × ${cy.label}: drag anywhere on the pad, arrow keys to nudge. Double-click the pad to reset.`}
            onKeyDown={(e) => {
              const k = e.shiftKey ? 1 : 0.25;
              const dx = e.key === 'ArrowRight' ? k : e.key === 'ArrowLeft' ? -k : 0;
              const dy = e.key === 'ArrowUp' ? k : e.key === 'ArrowDown' ? -k : 0;
              if (!dx && !dy) return;
              e.preventDefault();
              b.begin();
              b.set(cx.id, vx + dx);
              b.set(cy.id, vy + dy);
              b.end();
            }}
          />
          {large && (
            <>
              <span class="mp mp-l">{cx.poles?.[0] ?? '−'}</span>
              <span class="mp mp-r">{cx.poles?.[1] ?? '+'}</span>
            </>
          )}
        </div>
      )}
      {large ? (
        <div class="mood-detail">
          {[cx, cy].filter((c): c is ControlSpec => !!c).map((c) => (
            <Slider key={c.id} control={c} value={b.values[c.id] ?? 0} color={color} detailed onStart={b.begin} onChange={(v) => b.set(c.id, v)} onEnd={b.end} />
          ))}
          <h4 class="eyebrow">Moods</h4>
          <MoodChips values={b.values} onPick={pick} />
          {active.length > 0 && <h4 class="eyebrow">Strength{active.length > 1 ? ' · the doses add up' : ''}</h4>}
          {active.map((m) => (
            <Slider key={m.control.id} control={m.control} value={b.values[m.control.id] ?? 0} color={color} detailed onStart={b.begin} onChange={(v) => b.set(m.control.id, v)} onEnd={b.end} />
          ))}
          {marked && <p class="hint">Landmarks sit where each mood measured at its default dose. Click one to add it, click again to remove; drag it or its handle along the line for strength. Arrow keys move between moods, + / − change the dose.</p>}
        </div>
      ) : (
        <MoodChips values={b.values} onPick={pick} />
      )}
      {!compact && !large && cx && cy && (
        <p class="hint">
          {cx.label} {vx.toFixed(1)} · {cy.label} {vy.toFixed(1)}
        </p>
      )}
    </div>
  );
}

import type { ControlSpec } from '../api';
import { clamp, fmt } from '../lib/math';
import { zoneOf } from '../state/engine';
import { drag } from './drag';

export interface Gesture {
  onStart?: () => void;
  onChange: (v: number) => void;
  onEnd?: () => void;
}

interface Props extends Gesture {
  control: ControlSpec;
  value: number;
  color: string;
  disabled?: boolean;
  vertical?: boolean;
  /** Hide the label row (when the surrounding UI labels it). */
  bare?: boolean;
  /** Override the label. */
  label?: string;
  /** One row: label, track, value (poles in the tooltip). */
  compact?: boolean;
  /** Roomy: value always shown, the measured effect and the tested ranges under the track. */
  detailed?: boolean;
  /** How the value reads (default: signed, one decimal). */
  format?: (v: number) => string;
}

export const ZONE_TEXT: Record<string, string> = { strong: 'strong', collapse: 'may glitch' };

/** Percent position of v along a range (bottom→top when vertical). */
const pos = (lo: number, hi: number, v: number) => ((clamp(v, lo, hi) - lo) / (hi - lo || 1)) * 100;

export function step(c: Pick<ControlSpec, 'range'>) {
  const span = c.range[1] - c.range[0];
  return span > 12 ? 0.5 : span > 5 ? 0.1 : 0.05;
}

export function Slider({ control: c, value, color, disabled, vertical, bare, label, compact, detailed, format = fmt, onStart, onChange, onEnd }: Props) {
  const [lo, hi] = c.range;
  const zone = zoneOf(c, value);
  const st = step(c);
  const snap = (v: number) => Math.round(clamp(v, lo, hi) / st) * st;
  const zero = pos(lo, hi, 0);
  const at = pos(lo, hi, value);
  const bands: { from: number; to: number; cls: string }[] = [];
  if (c.safe[0] > lo) bands.push({ from: 0, to: pos(lo, hi, c.safe[0]), cls: 'strong' });
  if (c.safe[1] < hi) bands.push({ from: pos(lo, hi, c.safe[1]), to: 100, cls: 'strong' });
  if (c.collapse?.[0] != null && c.collapse[0] > lo) bands.push({ from: 0, to: pos(lo, hi, c.collapse[0]), cls: 'collapse' });
  if (c.collapse?.[1] != null && c.collapse[1] < hi) bands.push({ from: pos(lo, hi, c.collapse[1]), to: 100, cls: 'collapse' });
  const side = vertical ? 'bottom' : 'left';
  const size = vertical ? 'height' : 'width';

  const key = (e: KeyboardEvent) => {
    let v: number | null = null;
    const big = e.shiftKey ? 5 : 1;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') v = value + st * 2 * big;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') v = value - st * 2 * big;
    else if (e.key === 'Home') v = lo;
    else if (e.key === 'End') v = hi;
    else if (e.key === '0' || e.key === 'Delete' || e.key === 'Backspace') v = 0;
    if (v === null) return;
    e.preventDefault();
    onStart?.();
    onChange(snap(v));
    onEnd?.();
  };

  const poles = c.type === 'bipolar' && c.poles ? `${c.poles[0]} ↔ ${c.poles[1]}` : null;
  const title = [compact ? poles : null, c.effect, `range ${lo}…${hi} · natural ${c.safe[0]}…${c.safe[1]}`].filter(Boolean).join(' · ');

  return (
    <div class={`slider${vertical ? ' is-vertical' : ''}${compact ? ' is-compact' : ''}${detailed ? ' is-detailed' : ''}${disabled ? ' is-disabled' : ''} zone-${zone}`} style={{ '--fam': color }} title={title}>
      {compact && <span class="slider-label">{label ?? c.label}</span>}
      {!bare && !compact && (
        <div class="slider-head">
          <span class="slider-label">{label ?? c.label}</span>
          <span class="slider-value">
            {ZONE_TEXT[zone] && <span class={`zone-tag zone-${zone}`}>{ZONE_TEXT[zone]}</span>}
            {format(value)}
          </span>
        </div>
      )}
      <div
        class="slider-track"
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label={label ?? c.label}
        aria-valuemin={lo}
        aria-valuemax={hi}
        aria-valuenow={value}
        aria-valuetext={`${format(value)}${ZONE_TEXT[zone] ? ` (${ZONE_TEXT[zone]})` : ''}`}
        aria-orientation={vertical ? 'vertical' : 'horizontal'}
        aria-disabled={disabled}
        onKeyDown={disabled ? undefined : key}
        onDblClick={disabled ? undefined : () => (onStart?.(), onChange(0), onEnd?.())}
        onPointerDown={
          disabled
            ? undefined
            : (e) => {
                onStart?.();
                drag(e, (fx, fy) => onChange(snap(lo + (vertical ? 1 - fy : fx) * (hi - lo))), onEnd);
              }
        }
      >
        <div class="slider-rail" />
        {bands.map((b) => (
          <div class={`slider-band band-${b.cls}`} style={{ [side]: `${b.from}%`, [size]: `${b.to - b.from}%` }} />
        ))}
        <div class="slider-fill" style={{ [side]: `${Math.min(zero, at)}%`, [size]: `${Math.abs(at - zero)}%` }} />
        {lo < 0 && <div class="slider-zero" style={{ [side]: `${zero}%` }} />}
        <div class="slider-thumb" style={{ [side]: `${at}%` }} />
      </div>
      {compact && (
        <span class="slider-value">
          {ZONE_TEXT[zone] && <span class={`zone-tag zone-${zone}`}>{ZONE_TEXT[zone]}</span>}
          {format(value)}
        </span>
      )}
      {!bare && !compact && c.type === 'bipolar' && c.poles && (
        <div class="slider-poles">
          <span>{c.poles[0]}</span>
          <span>{c.poles[1]}</span>
        </div>
      )}
      {detailed && (
        <p class="slider-effect">
          {c.effect && <span>{c.effect}</span>}
          <span class="slider-ranges">
            natural {fmt(c.safe[0])}…{fmt(c.safe[1])}
            {(c.safe[0] > lo || c.safe[1] < hi) && <> · strong past that</>}
            {c.collapse && (c.collapse[0] != null || c.collapse[1] != null) && (
              <> · may glitch {c.collapse[0] != null ? `below ${fmt(c.collapse[0])}` : ''}{c.collapse[0] != null && c.collapse[1] != null ? ', ' : ''}{c.collapse[1] != null ? `past ${fmt(c.collapse[1])}` : ''}</>
            )}
          </span>
        </p>
      )}
    </div>
  );
}

import type { ControlSpec } from '../../api';
import { Slider } from '../../components/Slider';
import { hue } from '../../state/app';
import type { Values } from '../../state/engine';
import { spannable } from '../../state/edit';
import { MoodChips } from '../../components/Moods';
import { moodIds, moods } from '../../state/moods';

interface Chip {
  control: ControlSpec;
  label: string;
  value: number;
}

const half = (x: number) => Math.max(0.5, Math.round(x * 2) / 2);

/** One quick chip per pole of every spannable control, at a moderate dose inside its tested-safe range. */
function chipsOf(cs: ControlSpec[], taken: Set<string>): Chip[] {
  // A pole named like a mood ("calm" arousal vs the Calm mood) says which it is.
  const pole = (c: ControlSpec, p: string) => (taken.has(p.toLowerCase()) ? `${p} · ${c.label.toLowerCase()}` : p);
  return cs.flatMap((c) => {
    if (c.type === 'bipolar' && c.poles) {
      return [
        { control: c, label: pole(c, c.poles[0]), value: -half(Math.abs(c.safe[0]) * 0.65) },
        { control: c, label: pole(c, c.poles[1]), value: half(c.safe[1] * 0.6) },
      ];
    }
    return [{ control: c, label: c.label.toLowerCase(), value: half(c.safe[1] * 0.5) }];
  });
}

interface Props {
  values: Values;
  set: (control: string, v: number) => void;
  clear?: () => void;
  /** Name for labels ("region 2", "whole edit"). */
  name: string;
}

/** "Say this part …": quick chips, and the fine sliders with each control's measured effect. */
export function Direction({ values, set, clear, name }: Props) {
  const cs = spannable.value;
  // Named moods get their own chips (at their measured default dose); sliders only for the active ones.
  const named = moodIds.value;
  const chips = chipsOf(cs.filter((c) => !named.has(c.id)), new Set(moods.value.map((m) => m.label.toLowerCase())));
  const fine = cs.filter((c) => !named.has(c.id) || values[c.id]);
  const active = cs.filter((c) => values[c.id]);
  return (
    <div class="direction">
      <div class="direction-chips" role="group" aria-label={`Direction for ${name}`}>
        {chips.map((ch) => {
          const v = values[ch.control.id] ?? 0;
          const on = v !== 0 && Math.sign(v) === Math.sign(ch.value);
          return (
            <button
              class={`chip dchip${on ? ' is-on' : ''}`}
              style={{ '--fam': hue(ch.control.group) }}
              aria-pressed={on}
              title={`${ch.control.label} ${ch.value > 0 ? '+' : ''}${ch.value}${ch.control.effect ? ` · ${ch.control.effect}` : ''}`}
              onClick={() => set(ch.control.id, on ? 0 : ch.value)}
            >
              {ch.label}
              {on && <b>×{+Math.abs(v).toFixed(2)}</b>}
            </button>
          );
        })}
        <MoodChips values={values} onPick={(m) => set(m.control.id, (values[m.control.id] ?? 0) > 0 ? 0 : m.dose)} />
        {clear && active.length > 0 && (
          <button class="link dclear" onClick={clear}>
            clear
          </button>
        )}
      </div>
      <details class="pro direction-fine">
        <summary>{active.length ? `Fine-tune ${active.map((c) => c.label.toLowerCase()).join(', ')}` : 'Fine-tune doses'}</summary>
        <div class="direction-sliders">
          {fine.map((c) => (
            <Slider key={c.id} control={c} value={values[c.id] ?? 0} color={hue(c.group)} detailed onChange={(v) => set(c.id, v)} />
          ))}
        </div>
      </details>
    </div>
  );
}

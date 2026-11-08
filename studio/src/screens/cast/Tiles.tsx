/** Phone Cast: the read-only summaries on the six control tiles (each tile opens its full control in a dialog). */
import { controlById, spec } from '../../state/app';
import { accentMix, baseSpeaker, design, identity } from '../../state/design';
import { genderWord, identityWord, roomScenes, zoneOf } from '../../state/engine';
import { moods, padAxes } from '../../state/moods';
import { femaleAt, spotOf, who, yearsAt } from '../../state/who';
import { WHO_RAIL } from './WhoMap';

const signed = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}`;
const moved = (v: number | undefined) => Math.abs(v ?? 0) > 1e-6;

export function WhoTile() {
  const w = who.value;
  const d = design.value;
  const cx = controlById.value.get(w.gender.control);
  const cy = controlById.value.get(w.age.control);
  const label = baseSpeaker.value?.label ?? 'Pick a voice';
  const base = spotOf(baseSpeaker.value);
  const x = base.x + (cx ? (d.values[cx.id] ?? 0) : 0);
  const y = base.y + (cy ? (d.values[cy.id] ?? 0) : 0);
  const pitch = d.values[WHO_RAIL] ?? 0;
  const word = identityWord(identity.value, spec.value.ui.identity_threshold ?? 0.8);
  const outside = word === 'New person';
  return (
    <div class="tile-sum" title={`Identity estimate ${identity.value.toFixed(2)}${cx && cy ? ` · P(female) ${femaleAt(x).toFixed(2)}` : ''}`}>
      <b class="tile-main">{label}</b>
      {cx && cy && (
        <>
          <span>≈{Math.round(yearsAt(y))} years</span>
          <span>{genderWord(femaleAt(x))}</span>
        </>
      )}
      <span class={outside ? 'tile-warn' : 'muted'}>{outside ? word : moved(pitch) ? `pitch ${signed(pitch)}` : word}</span>
    </div>
  );
}

export function AccentTile() {
  const accents = spec.value.ui.accents;
  const active = Object.entries(accentMix.value)
    .map(([id, v]) => ({ a: accents.find((x) => x.id === id)!, v }))
    .filter((x) => x.a && x.v > 0)
    .sort((a, b) => b.v - a.v);
  const top = active[0];
  if (!top) {
    return (
      <div class="tile-sum">
        <b class="tile-main">No accent</b>
        <span class="muted">pin at home</span>
      </div>
    );
  }
  const strong = zoneOf(top.a, top.v) !== 'safe';
  return (
    <div class="tile-sum">
      <b class="tile-main">{top.a.label}</b>
      <span class={strong ? 'tile-warn' : undefined}>
        {top.v.toFixed(2)}
        {strong ? ' · strong' : ''}
      </span>
      {active.length > 1 && <span class="muted">+ {active.slice(1).map((x) => x.a.label).join(', ')}</span>}
    </div>
  );
}

export function RoomTile() {
  const c = spec.value.ui.controls.find((x) => x.group === 'room');
  if (!c) return <span class="muted">No room control</span>;
  const v = design.value.values[c.id] ?? 0;
  const [lo, hi] = c.range;
  // The same scenes as RoomAxis: the nearest one names the room.
  const scenes = roomScenes(c);
  const near = scenes.reduce((a, s) => (Math.abs(s.at - v) < Math.abs(a.at - v) ? s : a), scenes[0]);
  return (
    <div class="tile-sum">
      <b class="tile-main">{signed(v)}</b>
      <span>≈ {near.label}</span>
      <span class="tile-track" aria-hidden="true">
        <i style={{ left: `${((v - lo) / (hi - lo)) * 100}%` }} />
      </span>
    </div>
  );
}

export function DeliveryTile() {
  const values = design.value.values;
  const on = spec.value.ui.controls.filter((c) => c.group === 'delivery' && moved(values[c.id]));
  if (!on.length) {
    return (
      <div class="tile-sum">
        <b class="tile-main">Neutral</b>
        <span class="muted">no delivery moves</span>
      </div>
    );
  }
  return (
    <div class="tile-sum">
      {on.slice(0, 3).map((c) => (
        <span key={c.id} class="tile-kv">
          <span>{c.label}</span>
          <b>{signed(values[c.id] ?? 0)}</b>
        </span>
      ))}
      {on.length > 3 && <span class="muted">+{on.length - 3} more</span>}
    </div>
  );
}

export function MoodTile() {
  const values = design.value.values;
  const axes = padAxes.value;
  const active = moods.value.filter((m) => (values[m.control.id] ?? 0) > 0);
  const pos = (c: { id: string; range: [number, number] }) => ((values[c.id] ?? 0) - c.range[0]) / (c.range[1] - c.range[0]);
  return (
    <div class="tile-mood">
      {axes && (
        <span class="tile-pad" aria-hidden="true">
          <i style={{ left: `${pos(axes.x) * 100}%`, top: `${(1 - pos(axes.y)) * 100}%` }} />
        </span>
      )}
      <div class="tile-sum">
        {active.length ? (
          active.slice(0, 3).map((m) => <span key={m.control.id}>{m.label}</span>)
        ) : axes && (moved(values[axes.x.id]) || moved(values[axes.y.id])) ? (
          <>
            <span>
              {axes.x.label} {signed(values[axes.x.id] ?? 0)}
            </span>
            <span>
              {axes.y.label} {signed(values[axes.y.id] ?? 0)}
            </span>
          </>
        ) : (
          <b class="tile-main">Neutral</b>
        )}
        {active.length > 3 && <span class="muted">+{active.length - 3} more</span>}
      </div>
    </div>
  );
}

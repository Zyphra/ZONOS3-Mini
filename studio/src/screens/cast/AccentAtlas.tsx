import { useEffect, useRef, useState } from 'preact/hooks';
import { drag } from '../../components/drag';
import { Icon } from '../../components/Icon';
import { clamp } from '../../lib/math';
import { hue, spec } from '../../state/app';
import { geoOf, homePin, tabsOf } from '../../state/atlas';
import { accentMix, design, designs, setPin } from '../../state/design';
import { zoneOf } from '../../state/engine';
import { ATLAS_H, ATLAS_W } from '../../state/geo';

const TAB_LABEL: Record<string, string> = { english: 'English accents', world: 'Other languages' };

/** `large`: the phone dialog's atlas, as tall as its width allows instead of the bench's `--map-h`. */
export function AccentAtlas({ large }: { large?: boolean }) {
  const s = spec.value;
  const accents = s.ui.accents;
  const tabs = tabsOf(accents);
  const pin = design.value.pin;
  const [tab, setTab] = useState(tabs.includes(pin.tab) ? pin.tab : tabs[0]);
  // Follow the pin when it changes map (presets, undo, variations).
  useEffect(() => {
    if (tabs.includes(pin.tab)) setTab(pin.tab);
  }, [pin.tab]);
  const svg = useRef<SVGSVGElement>(null);
  /** A burst of arrow presses is one undo step (ends after 400 ms idle). */
  const keyIdle = useRef(0);
  useEffect(() => () => clearTimeout(keyIdle.current), []);
  const color = hue('accent');
  const geo = geoOf(tab, accents);
  const mix = accentMix.value;
  const here = accents.filter((a) => a.tab === tab);
  const pinHere = pin.tab === tab;
  const [hx, hy] = geo.home;
  const active = Object.entries(mix)
    .map(([id, v]) => ({ a: accents.find((x) => x.id === id)!, v }))
    .filter((x) => x.a)
    .sort((a, b) => b.v - a.v);
  const top = active[0];

  const moveTo = (fx: number, fy: number) => setPin({ tab, x: clamp(fx * ATLAS_W, 6, ATLAS_W - 6), y: clamp(fy * ATLAS_H, 6, ATLAS_H - 6) });
  const key = (e: KeyboardEvent) => {
    const k = e.shiftKey ? 40 : 10;
    const p = pinHere ? pin : homePin(tab, accents);
    const dx = e.key === 'ArrowRight' ? k : e.key === 'ArrowLeft' ? -k : 0;
    const dy = e.key === 'ArrowDown' ? k : e.key === 'ArrowUp' ? -k : 0;
    if (e.key === 'Home' || e.key === '0' || e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      designs.apply({ ...design.value, pin: homePin(tab, accents) });
      return;
    }
    if (!dx && !dy) return;
    e.preventDefault();
    if (!keyIdle.current) designs.begin();
    clearTimeout(keyIdle.current);
    setPin({ tab, x: clamp(p.x + dx, 6, ATLAS_W - 6), y: clamp(p.y + dy, 6, ATLAS_H - 6) });
    keyIdle.current = window.setTimeout(() => {
      keyIdle.current = 0;
      designs.end();
    }, 400);
  };
  const tabKeys = (e: KeyboardEvent) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const i = tabs.indexOf(tab);
    const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
    setTab(next);
    (e.currentTarget as HTMLElement).querySelector<HTMLElement>(`#atlas-tab-${next}`)?.focus();
  };
  const pinLabel = top ? `${top.a.label} ${top.v.toFixed(2)}${zoneOf(top.a, top.v) === 'safe' ? '' : ' · strong'}` : '';
  const labelLeft = pin.x > ATLAS_W - 300;

  return (
    <div class={`atlas${large ? ' is-large' : ''}`} style={{ '--fam': color }}>
      <div class="atlas-tabs" role="tablist" aria-label="Accent maps" onKeyDown={tabKeys}>
        {tabs.map((t) => (
          <button id={`atlas-tab-${t}`} role="tab" aria-selected={t === tab} aria-controls={`atlas-panel-${t}`} tabIndex={t === tab ? 0 : -1} class={`tab${t === tab ? ' is-on' : ''}`} onClick={() => setTab(t)}>
            {TAB_LABEL[t] ?? t}
            {pin.tab === t && active.length > 0 && <span class="fam-dot" />}
          </button>
        ))}
        <div class="atlas-readout" title={active.map(({ a, v }) => `${a.label} ${v.toFixed(2)}`).join(' + ')}>
          {!pinHere && active.length > 0 ? (
            <span class="muted">Dragging here replaces {TAB_LABEL[pin.tab] ?? pin.tab}</span>
          ) : active.length ? (
            <span class="muted">{active.map(({ a, v }) => `${a.label} ${v.toFixed(2)}`).join(' + ')}</span>
          ) : (
            <span class="muted">no accent</span>
          )}
        </div>
        <button class="link home-btn" onClick={() => designs.apply({ ...design.value, pin: homePin(tab, accents) })} title="Back home: no accent">
          <Icon name="home" size={14} /> home
        </button>
      </div>
      <svg
        ref={svg}
        id={`atlas-panel-${tab}`}
        class="atlas-map"
        viewBox={`0 0 ${ATLAS_W} ${ATLAS_H}`}
        role="slider"
        tabIndex={0}
        aria-label={`${TAB_LABEL[tab] ?? tab} map. Drag the pin into a region; deeper is stronger, borders blend. Arrow keys move it, Home returns.`}
        aria-valuetext={active.length ? active.map((x) => `${x.a.label} ${x.v.toFixed(2)}`).join(', ') : 'no accent'}
        onKeyDown={key}
        onPointerDown={(e) => {
          designs.begin();
          drag(e, moveTo, () => designs.end(), svg.current!);
        }}
      >
        <defs>
          {geo.panels.map((p) => (
            <clipPath id={`clip-${tab}-${p.id}`}>
              <rect x={p.rect[0]} y={p.rect[1]} width={p.rect[2]} height={p.rect[3]} rx="14" />
            </clipPath>
          ))}
        </defs>
        {geo.panels.map((p) => (
          <g class="atlas-panel">
            <rect class="panel-sea" x={p.rect[0]} y={p.rect[1]} width={p.rect[2]} height={p.rect[3]} rx="14" />
            <path class="land" d={geo.context.find((c) => c.panel === p.id)?.path} clip-path={`url(#clip-${tab}-${p.id})`} />
            <text class="panel-label" x={p.rect[0] + 12} y={p.rect[1] + p.rect[3] - 12}>
              {p.label}
            </text>
          </g>
        ))}
        {here.map((a) => {
          const r = geo.regions[a.region];
          if (!r) return null;
          const v = pinHere ? (mix[a.id] ?? 0) : 0;
          return (
            <g class={`region strength-${a.strength}${v > 0 ? ' is-active' : ''}`} style={{ '--w': String(v / a.range[1]) }}>
              <title>{`${a.label} — ${a.strength}${a.effect ? `: ${a.effect}` : ''}`}</title>
              <path class="region-shape" d={r.path} />
            </g>
          );
        })}
        {here.map((a) => {
          const r = geo.regions[a.region];
          if (!r) return null;
          const multi = here.filter((x) => x.region === a.region).length > 1;
          const text = multi ? accents.filter((x) => x.region === a.region && x.tab === tab).map((x) => x.label).join(' / ') : a.label;
          const half = text.length * 7.6;
          return (
            <text class={`region-label strength-${a.strength}`} x={clamp(r.pole[0], half + 4, ATLAS_W - half - 4)} y={r.pole[1] + 9}>
              {text}
            </text>
          );
        })}
        <g class="home" transform={`translate(${hx},${hy})`}>
          <circle r="26" />
          <text y="56">home · no accent</text>
        </g>
        {pinHere && (
          <g class="pin" transform={`translate(${pin.x},${pin.y})`}>
            <path d="M0,0 C-15,-21 -21,-33 -21,-45 A21,21 0 1 1 21,-45 C21,-33 15,-21 0,0Z" />
            <circle cy="-45" r="7" />
            {top && <g class={`pin-tag zone-${zoneOf(top.a, top.v)}`} transform={`translate(${labelLeft ? -30 : 30},-45)`}>
              <rect x={labelLeft ? -pinLabel.length * 16 - 24 : 0} y="-25" width={pinLabel.length * 16 + 24} height="50" rx="25" />
              <text x={labelLeft ? -12 : 12} y="10" text-anchor={labelLeft ? 'end' : 'start'}>
                {pinLabel}
              </text>
            </g>}
          </g>
        )}
      </svg>
    </div>
  );
}

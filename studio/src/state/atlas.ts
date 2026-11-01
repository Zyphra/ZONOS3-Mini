/**
 * Accent atlas semantics: the pin's depth into a region (geo.ts) sets that accent's strength (range max at the
 * region's deepest point, a fifth of it at the outline, fading to 0 just outside), so pins near a border blend
 * neighbouring accents.
 */
import type { AccentSpec } from '../api';
import type { AccentMix } from './engine';
import { depth, tabGeo, type Pt, type TabGeo } from './geo';

export interface Pin {
  tab: string;
  x: number;
  y: number;
}

export function tabsOf(accents: AccentSpec[]): string[] {
  return [...new Set(accents.map((a) => a.tab))];
}

export function geoOf(tab: string, accents: AccentSpec[]): TabGeo {
  return tabGeo(tab, [...new Set(accents.filter((a) => a.tab === tab).map((a) => a.region))]);
}

export function homePin(tab = 'english', accents: AccentSpec[] = []): Pin {
  const [x, y] = geoOf(tab, accents).home;
  return { tab, x, y };
}

export function strengthAt(a: AccentSpec, geo: TabGeo, p: Pt): number {
  const r = geo.regions[a.region];
  return r ? Math.round(a.range[1] * depth(r, p) * 20) / 20 : 0;
}

/** Accent strengths at a pin position. */
export function mixAt(pin: Pin, accents: AccentSpec[]): AccentMix {
  const geo = geoOf(pin.tab, accents);
  const out: AccentMix = {};
  for (const a of accents) {
    if (a.tab !== pin.tab) continue;
    const s = strengthAt(a, geo, [pin.x, pin.y]);
    if (s > 0) out[a.id] = s;
  }
  return out;
}

/** Where to put the pin so that accent `a` has strength `s`: on the line from the region's heart toward home. */
export function placeFor(a: AccentSpec, s: number, accents: AccentSpec[]): Pin {
  const geo = geoOf(a.tab, accents);
  const r = geo.regions[a.region];
  if (!r) return homePin(a.tab, accents);
  const [hx, hy] = geo.home;
  const at = (t: number): Pt => [r.pole[0] + (hx - r.pole[0]) * t, r.pole[1] + (hy - r.pole[1]) * t];
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 30; i++) {
    const m = (lo + hi) / 2;
    if (strengthAt(a, geo, at(m)) >= s) lo = m;
    else hi = m;
  }
  const [x, y] = at(lo);
  return { tab: a.tab, x, y };
}

/** Pin for a preset-style mix (the strongest accent wins; mixes come from dragging). */
export function pinForMix(mix: AccentMix, accents: AccentSpec[], tab = 'english'): Pin {
  const top = Object.entries(mix)
    .filter(([, s]) => s > 0)
    .sort((a, b) => b[1] - a[1])[0];
  const a = top && accents.find((x) => x.id === top[0]);
  return a ? placeFor(a, top[1], accents) : homePin(tab, accents);
}

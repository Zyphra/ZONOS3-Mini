/**
 * Real geography for the accent atlas: Natural Earth 110m countries (world-atlas TopoJSON), Mercator per panel,
 * each tab a set of panels (insets) laid out in one SVG. England / Scotland / Wales are cut out of the United
 * Kingdom along approximate borders; Northern Ireland joins Ireland (the island is one accent region).
 */
import { feature } from 'topojson-client';
import type { GeometryCollection, Topology } from 'topojson-specification';
import world from 'world-atlas/countries-110m.json';

export type Pt = [number, number];
type Ring = Pt[];

export const ATLAS_W = 1000;
export const ATLAS_H = 960;

export interface Panel {
  id: string;
  label: string;
  /** lon0, lat0, lon1, lat1 */
  bbox: [number, number, number, number];
  /** x, y, w, h in atlas units */
  rect: [number, number, number, number];
}

export interface GeoRegion {
  id: string;
  polys: Ring[];
  path: string;
  /** Deepest interior point (label / pin target) and its distance to the outline. */
  pole: Pt;
  inradius: number;
}

export interface TabGeo {
  panels: Panel[];
  context: { panel: string; path: string }[];
  regions: Record<string, GeoRegion>;
  home: Pt;
}

interface Layout {
  panels: Panel[];
  home: Pt;
  regions: Record<string, string[]>; // region id → piece names (country names or derived pieces)
}

const LAYOUTS: Record<string, Layout> = {
  english: {
    home: [170, 640],
    panels: [
      { id: 'na', label: 'North America', bbox: [-150, 24, -52, 70], rect: [10, 10, 540, 360] },
      { id: 'uk', label: 'British Isles', bbox: [-10.8, 49.8, 2.2, 59.2], rect: [560, 10, 430, 560] },
      { id: 'in', label: 'South Asia', bbox: [66, 5, 98, 36], rect: [300, 390, 250, 260] },
      { id: 'au', label: 'Australia', bbox: [112, -44.5, 155, -9.5], rect: [560, 590, 430, 360] },
    ],
    regions: {
      north_america: ['United States of America', 'Canada'],
      england: ['England'],
      scotland: ['Scotland'],
      wales: ['Wales'],
      ireland: ['Ireland', 'Northern Ireland'],
      india: ['India'],
      australia: ['Australia'],
    },
  },
  world: {
    home: [220, 720],
    panels: [
      { id: 'eu', label: 'Europe', bbox: [-10.5, 35.5, 25.5, 56.5], rect: [10, 10, 560, 475] },
      { id: 'as', label: 'Asia', bbox: [66, 6, 128, 54], rect: [480, 495, 510, 455] },
    ],
    regions: {
      spain: ['Spain'],
      france: ['France'],
      portugal: ['Portugal'],
      germany: ['Germany'],
      poland: ['Poland'],
      india: ['India'],
      china: ['China', 'Taiwan'],
      vietnam: ['Vietnam'],
    },
  },
};

// ── Countries (lon/lat) ───────────────────────────────────────────────────────────────────────────

const topo = world as unknown as Topology<{ countries: GeometryCollection<{ name: string }> }>;
const countries = feature(topo, topo.objects.countries).features;

function ringsOf(name: string): Ring[] {
  const f = countries.find((c) => c.properties?.name === name);
  if (!f) return [];
  const g = f.geometry;
  if (g.type === 'Polygon') return [g.coordinates[0] as Ring];
  if (g.type === 'MultiPolygon') return g.coordinates.map((p) => p[0] as Ring);
  return [];
}

/** Sutherland–Hodgman against the half-plane where `keep(p) >= 0` (keep is affine). */
function clipHalf(ring: Ring, keep: (p: Pt) => number): Ring {
  const out: Ring = [];
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i];
    const b = ring[(i + 1) % ring.length];
    const ka = keep(a);
    const kb = keep(b);
    if (ka >= 0) out.push(a);
    if (ka >= 0 !== kb >= 0) {
      const t = ka / (ka - kb);
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
  }
  return out;
}

const clipAll = (rings: Ring[], ...keeps: ((p: Pt) => number)[]) =>
  rings.map((r) => keeps.reduce((acc, k) => clipHalf(acc, k), r)).filter((r) => r.length >= 3);

/** Signed side of p relative to the directed line a→b (> 0: left). */
const side = (a: Pt, b: Pt) => (p: Pt) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);

// Approximate internal borders of the United Kingdom.
const SCOT_A: Pt = [-3.4, 54.95];
const SCOT_B: Pt = [-2.0, 55.8];
const WALES_A: Pt = [-2.65, 51.6];
const WALES_B: Pt = [-3.1, 53.45];

function pieces(name: string): Ring[] {
  if (!['England', 'Scotland', 'Wales', 'Northern Ireland'].includes(name)) return ringsOf(name);
  const uk = ringsOf('United Kingdom');
  const isNI = (r: Ring) => r.every(([lon, lat]) => lon < -5.4 && lat > 53.8 && lat < 55.4);
  if (name === 'Northern Ireland') return uk.filter(isNI);
  const gb = uk.filter((r) => !isNI(r));
  const scot = side(SCOT_A, SCOT_B); // > 0: north-west of the border = Scotland
  if (name === 'Scotland') return clipAll(gb, scot);
  const ew = clipAll(gb, (p) => -scot(p));
  const westOf = side(WALES_A, WALES_B); // > 0: west of the border
  const inWalesBand = (p: Pt) => Math.min(p[1] - 51.38, 53.45 - p[1]);
  if (name === 'Wales') return clipAll(ew, westOf, inWalesBand);
  return [...clipAll(ew, (p) => -westOf(p)), ...clipAll(ew, westOf, (p) => 51.38 - p[1]), ...clipAll(ew, westOf, (p) => p[1] - 53.45)];
}

// ── Projection per panel ──────────────────────────────────────────────────────────────────────────

const mercY = (lat: number) => (180 / Math.PI) * Math.log(Math.tan(Math.PI / 4 + (Math.max(-85, Math.min(85, lat)) * Math.PI) / 360));

function projector(p: Panel): (c: Pt) => Pt {
  const [lon0, lat0, lon1, lat1] = p.bbox;
  const [x, y, w, h] = p.rect;
  const y0 = mercY(lat0);
  const y1 = mercY(lat1);
  const s = Math.min(w / (lon1 - lon0), h / (y1 - y0));
  const ox = x + (w - (lon1 - lon0) * s) / 2;
  const oy = y + (h - (y1 - y0) * s) / 2;
  return ([lon, lat]) => [ox + (lon - lon0) * s, oy + (y1 - mercY(lat)) * s];
}

function clipRect(ring: Ring, [x, y, w, h]: Panel['rect']): Ring {
  return [(p: Pt) => p[0] - x, (p: Pt) => x + w - p[0], (p: Pt) => p[1] - y, (p: Pt) => y + h - p[1]].reduce((r, k) => clipHalf(r, k), ring);
}

const pathOf = (rings: Ring[]) => rings.map((r) => 'M' + r.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join('L') + 'Z').join('');

function inside(rings: Ring[], [px, py]: Pt): boolean {
  let hit = false;
  for (const r of rings) {
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [xi, yi] = r[i];
      const [xj, yj] = r[j];
      if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) hit = !hit;
    }
  }
  return hit;
}

function edgeDistance(rings: Ring[], [px, py]: Pt): number {
  let best = Infinity;
  for (const r of rings) {
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [ax, ay] = r[j];
      const [bx, by] = r[i];
      const dx = bx - ax;
      const dy = by - ay;
      const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy || 1)));
      best = Math.min(best, Math.hypot(px - ax - t * dx, py - ay - t * dy));
    }
  }
  return best;
}

function poleOf(rings: Ring[]): { pole: Pt; inradius: number } {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const r of rings) for (const [x, y] of r) (x0 = Math.min(x0, x)), (y0 = Math.min(y0, y)), (x1 = Math.max(x1, x)), (y1 = Math.max(y1, y));
  let best: Pt = [(x0 + x1) / 2, (y0 + y1) / 2];
  let bestD = 0;
  const n = 28;
  for (let i = 0; i <= n; i++)
    for (let j = 0; j <= n; j++) {
      const p: Pt = [x0 + ((x1 - x0) * i) / n, y0 + ((y1 - y0) * j) / n];
      if (!inside(rings, p)) continue;
      const d = edgeDistance(rings, p);
      if (d > bestD) (bestD = d), (best = p);
    }
  return { pole: best, inradius: Math.max(bestD, 4) };
}

function bboxHits(rings: Ring[], [lon0, lat0, lon1, lat1]: Panel['bbox']) {
  return rings.some((r) => r.some(([lon, lat]) => lon >= lon0 - 5 && lon <= lon1 + 5 && lat >= lat0 - 5 && lat <= lat1 + 5));
}

const cache = new Map<string, TabGeo>();

/** Geometry of a tab; regions the layout doesn't know get a circle in a free row at the bottom. */
export function tabGeo(tab: string, regionIds: string[]): TabGeo {
  const key = `${tab}|${regionIds.join(',')}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const layout = LAYOUTS[tab] ?? { home: [ATLAS_W / 2, ATLAS_H - 60] as Pt, panels: [], regions: {} };
  const context: TabGeo['context'] = [];
  for (const p of layout.panels) {
    const proj = projector(p);
    const rings = countries
      .flatMap((c) => ringsOf(c.properties?.name ?? ''))
      .filter((r) => bboxHits([r], p.bbox))
      .map((r) => clipRect(r.map(proj), p.rect))
      .filter((r) => r.length >= 3);
    context.push({ panel: p.id, path: pathOf(rings) });
  }
  const regions: Record<string, GeoRegion> = {};
  const strays: string[] = [];
  for (const id of regionIds) {
    const names = layout.regions[id];
    if (!names) {
      strays.push(id);
      continue;
    }
    const polys: Ring[] = [];
    for (const p of layout.panels) {
      const proj = projector(p);
      for (const n of names) {
        for (const r of pieces(n)) {
          if (!bboxHits([r], p.bbox)) continue;
          const pr = clipRect(r.map(proj), p.rect);
          if (pr.length >= 3) polys.push(pr);
        }
      }
    }
    if (polys.length) regions[id] = { id, polys, path: pathOf(polys), ...poleOf(polys) };
    else strays.push(id);
  }
  strays.forEach((id, i) => {
    const step = ATLAS_W / (strays.length + 1);
    const c: Pt = [step * (i + 1), ATLAS_H - 40];
    const ring: Ring = Array.from({ length: 24 }, (_, k) => [c[0] + Math.cos((k / 24) * Math.PI * 2) * 34, c[1] + Math.sin((k / 24) * Math.PI * 2) * 26]);
    regions[id] = { id, polys: [ring], path: pathOf([ring]), pole: c, inradius: 26 };
  });
  const geo: TabGeo = { panels: layout.panels, context, regions, home: layout.home };
  cache.set(key, geo);
  return geo;
}

/** Influence just outside a region's outline (atlas units), so neighbours blend at borders. */
export const HALO = 18;
/** Share of the strength reached right at the outline. */
export const EDGE_W = 0.2;

/** 0..1 depth of p in a region: 1 at its deepest point, EDGE_W at the outline, 0 beyond the halo. */
export function depth(r: GeoRegion, p: Pt): number {
  const d = edgeDistance(r.polys, p);
  if (inside(r.polys, p)) return EDGE_W + (1 - EDGE_W) * Math.min(1, d / r.inradius);
  return EDGE_W * Math.max(0, 1 - d / HALO);
}

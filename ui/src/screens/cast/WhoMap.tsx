import { useEffect, useRef } from 'preact/hooks';
import { drag } from '../../components/drag';
import { Slider } from '../../components/Slider';
import { clamp } from '../../lib/math';
import { controlById, hue, spec } from '../../state/app';
import { accentMix, baseSpeaker, design, designs, identity, setBase, setValue } from '../../state/design';
import { identityEstimate } from '../../state/engine';
import { speakers } from '../../state/library';
import { extent, femaleAt, spotOf, who, yAtYears, yearsAt } from '../../state/who';

const W = 420;
const H = 420;
const PAD_L = 30;
const PAD_R = 12;
const PAD_T = 12;
const PAD_B = 26;
const LABEL_CHAR = 6.3; // approx. width of one label character in viewBox units (12px font)

export const WHO_RAIL = 'pitch';

/** Contour of the identity threshold around the base voice (dose offsets), other controls held. */
function identityRing(gx: string, gy: string, threshold: number): [number, number][] | null {
  const s = spec.value;
  const d = design.value;
  const f = (x: number, y: number) => identityEstimate(s, { ...d.values, [gx]: x, [gy]: y }, accentMix.value, d.mode);
  if (f(0, 0) < threshold) return null;
  const pts: [number, number][] = [];
  for (let k = 0; k < 72; k++) {
    const t = (k / 72) * Math.PI * 2;
    let lo = 0;
    let hi = 6;
    if (f(hi * Math.cos(t), hi * Math.sin(t)) >= threshold) lo = hi;
    else
      for (let i = 0; i < 22; i++) {
        const m = (lo + hi) / 2;
        if (f(m * Math.cos(t), m * Math.sin(t)) >= threshold) lo = m;
        else hi = m;
      }
    pts.push([lo * Math.cos(t), lo * Math.sin(t)]);
  }
  return pts;
}

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}
const overlaps = (a: Box, b: Box) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/** Place each label right / left / above / below its dot, skipping spots that collide; null = hover only. */
function placeLabels(items: { id: string; x: number; y: number; text: string }[], taken: Box[]) {
  const out = new Map<string, { dx: number; dy: number; anchor: string } | null>();
  const dots: Box[] = items.map((i) => ({ x: i.x - 5, y: i.y - 5, w: 10, h: 10 }));
  for (const it of items) {
    const w = it.text.length * LABEL_CHAR;
    const inward = it.x < W / 2 ? 1 : -1;
    const cands = [
      { dx: 9 * inward, dy: 4, anchor: inward > 0 ? 'start' : 'end', box: { x: inward > 0 ? it.x + 8 : it.x - 8 - w, y: it.y - 7, w, h: 13 } },
      { dx: -9 * inward, dy: 4, anchor: inward > 0 ? 'end' : 'start', box: { x: inward > 0 ? it.x - 8 - w : it.x + 8, y: it.y - 7, w, h: 13 } },
      { dx: 0, dy: -10, anchor: 'middle', box: { x: it.x - w / 2, y: it.y - 20, w, h: 13 } },
      { dx: 0, dy: 17, anchor: 'middle', box: { x: it.x - w / 2, y: it.y + 6, w, h: 13 } },
    ];
    const ok = cands.find(
      (c) =>
        c.box.x >= PAD_L && c.box.x + c.box.w <= W - PAD_R && c.box.y >= PAD_T && c.box.y + c.box.h <= H - PAD_B && !taken.some((t) => overlaps(t, c.box)) && !dots.some((d) => overlaps(d, c.box)),
    );
    if (ok) taken.push(ok.box);
    out.set(it.id, ok ? { dx: ok.dx, dy: ok.dy, anchor: ok.anchor } : null);
  }
  return out;
}

export function WhoMap() {
  const svg = useRef<SVGSVGElement>(null);
  const w = who.value;
  const cx = controlById.value.get(w.gender.control);
  const cy = controlById.value.get(w.age.control);
  const rail = controlById.value.get(WHO_RAIL);
  const d = design.value;
  const color = hue('who');

  useEffect(() => {
    const el = svg.current;
    if (!el || !rail) return;
    let idle = 0;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      designs.begin();
      setValue(rail.id, (design.value.values[rail.id] ?? 0) + (e.deltaY < 0 ? 0.1 : -0.1));
      clearTimeout(idle);
      idle = window.setTimeout(() => designs.end(), 400);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [rail]);

  if (!cx || !cy) return <p class="muted">This bundle has no gender / age controls.</p>;

  const { x: [x0, x1], y: [y0, y1] } = extent.value;
  const toX = (v: number) => PAD_L + ((v - x0) / (x1 - x0)) * (W - PAD_L - PAD_R);
  const toY = (v: number) => PAD_T + ((y1 - v) / (y1 - y0)) * (H - PAD_T - PAD_B);
  const inX = (v: number) => clamp(v, x0, x1);
  const inY = (v: number) => clamp(v, y0, y1);

  const base = spotOf(baseSpeaker.value);
  const gx = d.values[cx.id] ?? 0;
  const gy = d.values[cy.id] ?? 0;
  const dotX = base.x + gx;
  const dotY = base.y + gy;
  const offMap = dotX < x0 || dotX > x1 || dotY < y0 || dotY > y1;
  const threshold = spec.value.ui.identity_threshold ?? 0.8;
  const ring = identityRing(cx.id, cy.id, threshold);
  const outside = identity.value < threshold;

  const marks = speakers.value
    .filter((s) => s.id !== d.baseId)
    .map((s) => ({ s, spot: spotOf(s) }))
    .filter((m) => m.spot.measured || m.spot.from);
  const baseText = baseSpeaker.value?.label ?? 'base';
  const taken: Box[] = [{ x: toX(inX(base.x)) - (baseText.length * LABEL_CHAR) / 2, y: toY(inY(base.y)) - 24, w: baseText.length * LABEL_CHAR, h: 13 }];
  const labels = placeLabels(
    marks.map((m) => ({ id: m.s.id, x: toX(inX(m.spot.x)), y: toY(inY(m.spot.y)), text: m.s.label })),
    taken,
  );

  const moveTo = (fx: number, fy: number) => {
    const vx = x0 + ((fx * W - PAD_L) / (W - PAD_L - PAD_R)) * (x1 - x0);
    const vy = y1 - ((fy * H - PAD_T) / (H - PAD_T - PAD_B)) * (y1 - y0);
    const nd = design.value;
    designs.set({
      ...nd,
      values: {
        ...nd.values,
        [cx.id]: Math.round(clamp(vx - base.x, cx.range[0], cx.range[1]) * 20) / 20,
        [cy.id]: Math.round(clamp(vy - base.y, cy.range[0], cy.range[1]) * 20) / 20,
      },
    });
  };

  const key = (e: KeyboardEvent) => {
    const k = e.shiftKey ? 0.5 : 0.1;
    const dx = e.key === 'ArrowRight' ? k : e.key === 'ArrowLeft' ? -k : 0;
    const dy = e.key === 'ArrowUp' ? k : e.key === 'ArrowDown' ? -k : 0;
    if (!dx && !dy && e.key !== '0') return;
    e.preventDefault();
    designs.begin();
    setValue(cx.id, e.key === '0' ? 0 : gx + dx);
    setValue(cy.id, e.key === '0' ? 0 : gy + dy);
    designs.end();
  };

  const ages: number[] = [];
  for (let a = Math.ceil(yearsAt(y0) / 10) * 10; a <= yearsAt(y1); a += 10) ages.push(a);
  const pTicks = [0.1, 0.5, 0.9].map((p) => ({ p, x: Math.log(p / (1 - p)) / w.gender.logit_per_dose }));
  const safe = { x: [base.x + cx.safe[0], base.x + cx.safe[1]], y: [base.y + cy.safe[0], base.y + cy.safe[1]] };
  const years = Math.round(yearsAt(dotY));
  const pf = femaleAt(dotX);

  return (
    <div class="who" style={{ '--fam': color }}>
      <div class="who-main">
        <svg
          ref={svg}
          class={`who-map${outside ? ' is-outside' : ''}`}
          viewBox={`0 0 ${W} ${H}`}
          role="slider"
          tabIndex={0}
          aria-label={`Who map: masculine to feminine (x), age in years (y). Arrow keys move, scroll changes pitch.`}
          aria-valuetext={`${cx.label} ${gx.toFixed(1)}, ${cy.label} ${gy.toFixed(1)} (about ${years} years)`}
          onKeyDown={key}
          onPointerDown={(e) => {
            if ((e.target as Element).closest('.landmark')) return;
            designs.begin();
            drag(e, moveTo, () => designs.end(), svg.current!);
          }}
        >
          <rect class="who-bg" x={PAD_L} y={PAD_T} width={W - PAD_L - PAD_R} height={H - PAD_T - PAD_B} rx="12" />
          {ages.map((a) => (
            <g class="tick">
              <line class="grid" x1={PAD_L} x2={W - PAD_R} y1={toY(yAtYears(a))} y2={toY(yAtYears(a))} />
              <text x={PAD_L - 6} y={toY(yAtYears(a)) + 4} text-anchor="end">
                {a}
              </text>
            </g>
          ))}
          {pTicks.map((t) => (
            <g class="tick">
              <line class={t.p === 0.5 ? 'axis' : 'grid'} x1={toX(t.x)} x2={toX(t.x)} y1={PAD_T} y2={H - PAD_B} />
            </g>
          ))}
          <text class="pole" x={PAD_L + 6} y={H - 8}>
            ← {cx.poles?.[0] ?? 'masculine'}
          </text>
          <text class="pole" x={toX(0)} y={H - 8} text-anchor="middle">
            neutral
          </text>
          <text class="pole" x={W - PAD_R - 6} y={H - 8} text-anchor="end">
            {cx.poles?.[1] ?? 'feminine'} →
          </text>
          <text class="pole axis-title" x={11} y={PAD_T + 2} transform={`rotate(-90 11 ${PAD_T + 2})`} text-anchor="end">
            age (years)
          </text>
          <rect class="who-safe" x={toX(inX(safe.x[0]))} y={toY(inY(safe.y[1]))} width={toX(inX(safe.x[1])) - toX(inX(safe.x[0]))} height={toY(inY(safe.y[0])) - toY(inY(safe.y[1]))} rx="10" />
          {ring && <path class="identity-ring" d={`M${ring.map(([a, b]) => `${toX(inX(base.x + a)).toFixed(1)},${toY(inY(base.y + b)).toFixed(1)}`).join('L')}Z`} />}
          {marks.map(({ s, spot }) => {
            const l = labels.get(s.id);
            return (
              <g class={`landmark${l ? '' : ' is-quiet'}`} transform={`translate(${toX(inX(spot.x))},${toY(inY(spot.y))})`} onClick={() => setBase(s.id)} role="button" aria-label={`Switch base voice to ${s.label}`}>
                <title>{`${s.label}${spot.measured ? '' : ' (saved from ' + spot.from + ')'} — click to make it the base voice`}</title>
                <circle r="5" />
                <text x={l?.dx ?? 9} y={l?.dy ?? 4} text-anchor={l?.anchor ?? 'start'}>
                  {s.label}
                </text>
              </g>
            );
          })}
          <line class="who-tether" x1={toX(inX(base.x))} y1={toY(inY(base.y))} x2={toX(inX(dotX))} y2={toY(inY(dotY))} />
          <g class={`origin${base.measured ? '' : ' is-unmeasured'}`} transform={`translate(${toX(inX(base.x))},${toY(inY(base.y))})`}>
            <circle r="7" />
            <text y="-12">{baseText}</text>
          </g>
          <g class={`who-dot${offMap ? ' is-off' : ''}`} transform={`translate(${toX(inX(dotX))},${toY(inY(dotY))})`}>
            <circle class="halo" r="16" />
            <circle r="9" />
          </g>
        </svg>
        {rail && (
          <div class="who-rail">
            <Slider control={rail} value={d.values[rail.id] ?? 0} color={color} vertical bare onStart={() => designs.begin()} onChange={(v) => setValue(rail.id, v)} onEnd={() => designs.end()} />
            <span class="rail-label" title={rail.effect}>
              {rail.label}
              <br />
              <b>{(d.values[rail.id] ?? 0).toFixed(1)}</b>
            </span>
          </div>
        )}
      </div>
      <div class={`who-status${outside ? ' is-outside' : ''}`} aria-live="polite">
        {outside ? <b>New person</b> : <span>Still {baseText}</span>}
        <span>
          identity ≈ {identity.value.toFixed(2)} · ≈{years} y · P(female) {pf.toFixed(2)}
        </span>
        <span class="muted" title={w.note}>
          {base.from ? `placed from ${base.from} + its saved moves` : base.measured ? 'calibrated estimates' : 'position not measured: starts at the centre'}
        </span>
      </div>
      {spec.value.ui.controls
        .filter((c) => c.group === 'who' && ![cx.id, cy.id, WHO_RAIL].includes(c.id))
        .map((c) => (
          <Slider key={c.id} control={c} value={d.values[c.id] ?? 0} color={color} compact onStart={() => designs.begin()} onChange={(v) => setValue(c.id, v)} onEnd={() => designs.end()} />
        ))}
    </div>
  );
}

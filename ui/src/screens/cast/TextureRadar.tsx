import { useRef, useState } from 'preact/hooks';
import { drag } from '../../components/drag';
import { Slider, ZONE_TEXT } from '../../components/Slider';
import { clamp } from '../../lib/math';
import { hue, spec } from '../../state/app';
import { design, designs, setValue } from '../../state/design';
import { zoneOf } from '../../state/engine';

const S = 320;
const C = S / 2;
const R = 112;

/** Pull each spoke outward: one unipolar texture control per spoke. */
export function TextureRadar({ large }: { large?: boolean }) {
  const controls = spec.value.ui.controls.filter((c) => c.group === 'texture');
  const svg = useRef<SVGSVGElement>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const color = hue('texture');
  const values = design.value.values;
  const n = controls.length;
  if (!n) return null;
  const angle = (i: number) => -Math.PI / 2 + (i / n) * Math.PI * 2;
  const point = (i: number, f: number): [number, number] => [C + Math.cos(angle(i)) * R * f, C + Math.sin(angle(i)) * R * f];
  const frac = (i: number) => clamp((values[controls[i].id] ?? 0) / controls[i].range[1], 0, 1);
  const poly = controls.map((_, i) => point(i, frac(i)).map((v) => v.toFixed(1)).join(',')).join(' ');

  const grab = (e: PointerEvent, only?: number) => {
    const rect = svg.current!.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * S - C;
    const py = ((e.clientY - rect.top) / rect.height) * S - C;
    let i = only;
    if (i === undefined) {
      const a = Math.atan2(py, px);
      let best = 0;
      let bestD = 9;
      controls.forEach((_, k) => {
        const d = Math.abs(Math.atan2(Math.sin(a - angle(k)), Math.cos(a - angle(k))));
        if (d < bestD) {
          bestD = d;
          best = k;
        }
      });
      i = best;
    }
    const c = controls[i];
    setFocus(c.id);
    designs.begin();
    drag(
      e,
      (fx, fy) => {
        const x = fx * S - C;
        const y = fy * S - C;
        const along = x * Math.cos(angle(i!)) + y * Math.sin(angle(i!));
        setValue(c.id, Math.round(clamp(along / R, 0, 1) * c.range[1] * 10) / 10);
      },
      () => designs.end(),
      svg.current!,
    );
  };

  const rings = [0.25, 0.5, 0.75, 1];
  const safeF = controls[0].safe[1] / controls[0].range[1];
  const fc = controls.find((c) => c.id === focus);

  return (
    <div class={`radar${large ? ' is-large' : ''}`} style={{ '--fam': color }}>
      <svg ref={svg} class="radar-svg" viewBox={`0 0 ${S} ${S}`} onPointerDown={(e) => grab(e)} aria-hidden="true">
        {rings.map((f) => (
          <circle class="radar-ring" cx={C} cy={C} r={R * f} />
        ))}
        <circle class="radar-safe" cx={C} cy={C} r={R * safeF} />
        {controls.map((c, i) => {
          const [x, y] = point(i, 1);
          const [lx, ly] = point(i, 1.2);
          const v = values[c.id] ?? 0;
          // Tested-safe span of the spoke, then the strong zone, then where renders may break.
          const safeEnd = clamp(c.safe[1] / c.range[1], 0, 1);
          const breakAt = c.collapse?.[1] != null ? clamp(c.collapse[1] / c.range[1], 0, 1) : 1;
          const [sx, sy] = point(i, safeEnd);
          const [bx, by] = point(i, breakAt);
          return (
            <g>
              <line class="radar-spoke" x1={C} y1={C} x2={x} y2={y} />
              {large && (
                <>
                  <line class="spoke-safe" x1={C} y1={C} x2={sx} y2={sy} />
                  <line class="spoke-strong" x1={sx} y1={sy} x2={bx} y2={by} />
                  {breakAt < 1 && <line class="spoke-collapse" x1={bx} y1={by} x2={x} y2={y} />}
                </>
              )}
              <text class={`radar-label${focus === c.id ? ' is-on' : ''}`} x={lx} y={ly + 4}>
                {c.label}
                {large && <tspan class={`radar-value zone-${zoneOf(c, v)}`}>{` ${v.toFixed(1)}`}</tspan>}
              </text>
            </g>
          );
        })}
        <polygon class="radar-shape" points={poly} />
        {controls.map((c, i) => {
          const [x, y] = point(i, frac(i));
          const z = zoneOf(c, values[c.id] ?? 0);
          return <circle class={`radar-handle zone-${z}`} cx={x} cy={y} r="7" onPointerDown={(e) => (e.stopPropagation(), grab(e, i))} />;
        })}
      </svg>
      {large ? (
        <div class="radar-detail">
          {controls.map((c) => (
            <div class={`radar-row${focus === c.id ? ' is-on' : ''}`} onFocusIn={() => setFocus(c.id)} onPointerEnter={() => setFocus(c.id)}>
              <Slider control={c} value={values[c.id] ?? 0} color={color} detailed onStart={() => designs.begin()} onChange={(v) => setValue(c.id, v)} onEnd={() => designs.end()} />
            </div>
          ))}
        </div>
      ) : (
        <details class="pro">
          <summary>{fc ? `${fc.label}: ${fc.effect ?? ''}` : 'Numbers'}</summary>
          <div class="radar-keys">
          {controls.map((c) => {
            const v = values[c.id] ?? 0;
            const z = zoneOf(c, v);
            return (
              <label class={`radar-key zone-${z}`} title={c.effect}>
                <span>{c.label}</span>
                <input
                  type="range"
                  min={c.range[0]}
                  max={c.range[1]}
                  step={0.1}
                  value={v}
                  aria-label={`${c.label} (${c.effect ?? ''})`}
                  onFocus={() => setFocus(c.id)}
                  onPointerDown={() => designs.begin()}
                  onKeyDown={() => designs.begin()}
                  onInput={(e) => setValue(c.id, Number((e.currentTarget as HTMLInputElement).value))}
                  onChange={() => designs.end()}
                />
                <b>{v.toFixed(1)}</b>
                {ZONE_TEXT[z] && <em class={`zone-tag zone-${z}`}>{ZONE_TEXT[z]}</em>}
              </label>
            );
          })}
          </div>
        </details>
      )}
    </div>
  );
}

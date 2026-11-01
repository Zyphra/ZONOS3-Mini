/**
 * Start a pointer drag on the element that received `e`: `move` runs for the down event and every move,
 * `end` once on release/cancel. Coordinates are reported in the element's box (0..1 on both axes, unclamped).
 */
export function drag(
  e: PointerEvent,
  move: (fx: number, fy: number, ev: PointerEvent) => void,
  end?: () => void,
  target: Element = e.currentTarget as Element,
) {
  if (e.button !== 0) return;
  e.preventDefault();
  const el = e.currentTarget as Element;
  el.setPointerCapture?.(e.pointerId);
  const at = (ev: PointerEvent) => {
    const r = target.getBoundingClientRect();
    move((ev.clientX - r.left) / (r.width || 1), (ev.clientY - r.top) / (r.height || 1), ev);
  };
  const onMove = (ev: Event) => at(ev as PointerEvent);
  const onUp = () => {
    el.removeEventListener('pointermove', onMove);
    el.removeEventListener('pointerup', onUp);
    el.removeEventListener('pointercancel', onUp);
    end?.();
  };
  el.addEventListener('pointermove', onMove);
  el.addEventListener('pointerup', onUp);
  el.addEventListener('pointercancel', onUp);
  at(e);
}

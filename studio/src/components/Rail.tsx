import type { ComponentChildren } from 'preact';
import { useEffect, useRef } from 'preact/hooks';
import { signal } from '@preact/signals';
import { phone } from '../lib/media';
import { modalOpen } from '../state/app';
import { load, persist } from '../state/persist';
import { Icon } from './Icon';

/** Desktop: the rails folded away, by id (browser-wide). */
const folded = signal<Record<string, boolean>>(load('rails', {}));
persist('rails', folded);

/** Phone: the rail open as a drawer (one at a time) and the toggle that opened it (gets focus back). */
const drawer = signal<string | null>(null);
let opener: HTMLElement | null = null;

function closeDrawer() {
  drawer.value = null;
  opener?.focus({ preventScroll: true });
  opener = null;
}

/** Bring rail `id` into view: on a phone as a drawer, on desktop by unfolding it. */
export function openRail(id: string) {
  if (phone.peek()) {
    opener = document.activeElement as HTMLElement | null;
    drawer.value = id;
  } else if (folded.peek()[id]) folded.value = { ...folded.peek(), [id]: false };
}

interface Props {
  id: string;
  side: 'left' | 'right';
  /** The drawer's title and the folded strip's label. */
  title: string;
  /** The column itself (`aside.rail`, `aside.monitor` …). */
  children: ComponentChildren;
}

/**
 * A screen's side column. Desktop: in its grid column, with a fold toggle on its inner edge (folded, it is a thin strip
 * that unfolds it; remembered per rail). Phone: an overlay drawer opened by its `RailToggle`, closed by the backdrop,
 * the close button or Esc. The column stays mounted either way (its effects and file drops keep working).
 */
export function Rail({ id, side, title, children }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const isFolded = !!folded.value[id];
  const onPhone = phone.value;
  const open = onPhone && drawer.value === id;
  // A screen that goes away takes its open drawer with it.
  useEffect(
    () => () => {
      if (drawer.peek() === id) drawer.value = null;
    },
    [],
  );
  useEffect(() => {
    if (!open) return;
    box.current?.querySelector<HTMLElement>('.rail-drawer-head button')?.focus({ preventScroll: true });
    // Bubbling, after any dialog opened from inside the drawer (CardModal, pickers) has taken its own Esc.
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented || modalOpen.peek()) return;
      e.preventDefault();
      closeDrawer();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);
  const chevron = side === 'left' ? (isFolded ? 'right' : 'left') : isFolded ? 'left' : 'right';
  return (
    <>
      <div
        ref={box}
        id={`rail-${id}`}
        class={`rail-box is-${side}${isFolded ? ' is-folded' : ''}${open ? ' is-open' : ''}`}
        role={open ? 'dialog' : undefined}
        aria-label={open ? title : undefined}
        aria-modal={open ? 'true' : undefined}
        inert={onPhone && !open ? true : undefined}
      >
        <div class="rail-drawer-head">
          <h3>{title}</h3>
          <button class="icon-btn" onClick={closeDrawer} aria-label={`Close ${title}`} title="Close (Esc)">
            <Icon name="x" />
          </button>
        </div>
        <button
          class="rail-fold"
          onClick={() => (folded.value = { ...folded.value, [id]: !isFolded })}
          aria-expanded={!isFolded}
          aria-label={isFolded ? `Show ${title}` : `Hide ${title}`}
          title={isFolded ? `Show ${title}` : `Hide ${title}`}
        >
          <Icon name={chevron} size={14} />
          {isFolded && <span class="rail-fold-label">{title}</span>}
        </button>
        {children}
      </div>
      {open && <div class="rail-scrim" onClick={closeDrawer} />}
    </>
  );
}

/** Phone: opens (or closes) rail `id` as a drawer. */
export function RailToggle({ id, title, icon }: { id: string; title: string; icon: string }) {
  const on = drawer.value === id;
  return (
    <button
      class={`btn sm rail-toggle${on ? ' is-on' : ''}`}
      aria-expanded={on}
      aria-controls={`rail-${id}`}
      onClick={(e) => {
        opener = e.currentTarget;
        drawer.value = on ? null : id;
      }}
    >
      <Icon name={icon} size={14} /> {title}
    </button>
  );
}

/** Phone: the row of drawer toggles at the top of a screen (nothing on desktop). */
export function RailBar({ children }: { children: ComponentChildren }) {
  if (!phone.value) return null;
  return <div class="rail-bar">{children}</div>;
}

import type { ComponentChildren } from 'preact';
import { useLayoutEffect, useRef } from 'preact/hooks';
import { hue, modalOpen } from '../state/app';
import { Icon } from './Icon';

interface Props {
  family: string;
  title: string;
  /** The card it grows from (and shrinks back into). */
  origin: HTMLElement;
  /** Gets focus back on close. */
  returnFocus: HTMLElement;
  onClose: () => void;
  aside?: ComponentChildren;
  /** Shown between the title and the body (e.g. the preview). */
  strip?: ComponentChildren;
  children: ComponentChildren;
}

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
const EASE = 'cubic-bezier(0.2, 0.7, 0.2, 1)';
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Transform that maps the dialog's box onto the card's box (transform-origin: top left). */
function flip(from: DOMRect, to: DOMRect): string {
  return `translate(${from.left - to.left}px, ${from.top - to.top}px) scale(${from.width / to.width}, ${from.height / to.height})`;
}

/**
 * A bench card opened large: the dialog grows out of the card's rectangle (FLIP) and shrinks back into it on
 * close. Modal: focus is trapped inside and returns to the card; Esc, the close button or the backdrop close it.
 */
export function CardModal({ family, title, origin, returnFocus, onClose, aside, strip, children }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const backdrop = useRef<HTMLDivElement>(null);
  const closing = useRef(false);
  const downOnBackdrop = useRef(false);
  const titleId = `card-modal-${family}`;

  useLayoutEffect(() => {
    const el = box.current!;
    modalOpen.value = true;
    if (!reduced()) {
      el.animate([{ transform: flip(origin.getBoundingClientRect(), el.getBoundingClientRect()) }, { transform: 'none' }], { duration: 300, easing: EASE });
      el.querySelector('.card-modal-inner')?.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 180, delay: 120, easing: 'ease-out', fill: 'backwards' });
      backdrop.current!.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 220, easing: 'ease-out' });
    }
    const first = el.querySelector<HTMLElement>('.card-modal-body ' + FOCUSABLE) ?? el.querySelector<HTMLElement>(FOCUSABLE);
    first?.focus({ preventScroll: true });
    return () => {
      modalOpen.value = false;
    };
  }, []);

  const close = () => {
    if (closing.current) return;
    closing.current = true;
    const done = () => {
      onClose();
      returnFocus.focus({ preventScroll: true });
    };
    const el = box.current;
    if (!el || reduced()) return done();
    el.querySelector('.card-modal-inner')?.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 120, fill: 'forwards' });
    backdrop.current!.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 240, fill: 'forwards' });
    el.animate([{ transform: 'none' }, { transform: flip(origin.getBoundingClientRect(), el.getBoundingClientRect()) }], { duration: 240, easing: EASE, fill: 'forwards' }).finished.then(done, done);
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      close();
      return;
    }
    if (e.key !== 'Tab') return;
    const items = [...box.current!.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((x) => x.offsetParent !== null);
    if (!items.length) return;
    const i = items.indexOf(document.activeElement as HTMLElement);
    const next = e.shiftKey ? (i <= 0 ? items.length - 1 : i - 1) : i === items.length - 1 ? 0 : i + 1;
    e.preventDefault();
    items[next].focus();
  };

  return (
    <div
      ref={backdrop}
      class="card-modal-backdrop"
      onPointerDown={(e) => (downOnBackdrop.current = e.target === backdrop.current)}
      onClick={(e) => {
        if (downOnBackdrop.current && e.target === backdrop.current) close();
        downOnBackdrop.current = false;
      }}
    >
      <div ref={box} class="card-modal" role="dialog" aria-modal="true" aria-labelledby={titleId} style={{ '--fam': hue(family) }} onKeyDown={onKeyDown}>
        <div class="card-modal-inner">
          <header class="card-modal-head">
            <h2 id={titleId}>
              <span class="fam-dot" />
              {title}
            </h2>
            {aside && <div class="card-modal-aside">{aside}</div>}
            <button class="icon-btn" onClick={close} aria-label={`Close ${title}`} title="Close (Esc)">
              <Icon name="x" />
            </button>
          </header>
          {strip && <div class="card-modal-strip">{strip}</div>}
          <div class="card-modal-body">{children}</div>
        </div>
      </div>
    </div>
  );
}

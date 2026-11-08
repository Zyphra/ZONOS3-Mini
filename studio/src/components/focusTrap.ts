/** Keyboard plumbing shared by every dialog: the focusable selector, Tab trapping, first-focus and Esc-to-close. */

export const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Tab stays inside `box` (hidden items skipped). */
export function trapTab(e: KeyboardEvent, box: HTMLElement) {
  const items = [...box.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((x) => x.offsetParent !== null);
  if (!items.length) return;
  const i = items.indexOf(document.activeElement as HTMLElement);
  e.preventDefault();
  items[e.shiftKey ? (i <= 0 ? items.length - 1 : i - 1) : i === items.length - 1 ? 0 : i + 1].focus();
}

/** Focus the first focusable item of `box`, preferring one under `within` (a selector relative to `box`). */
export function focusFirst(box: HTMLElement, within?: string) {
  const first = (within && box.querySelector<HTMLElement>(`${within} ${FOCUSABLE}`)) || box.querySelector<HTMLElement>(FOCUSABLE);
  first?.focus({ preventScroll: true });
}

/**
 * A dialog's keydown handler: keys never reach the screens or the shell underneath; Esc closes the dialog (and only
 * it: the event is marked handled so the window listeners of lower layers stay put); Tab stays inside.
 */
export function dialogKeys(box: () => HTMLElement | null, onClose: () => void): (e: KeyboardEvent) => void {
  return (e) => {
    e.stopPropagation();
    if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    } else if (e.key === 'Tab') {
      const el = box();
      if (el) trapTab(e, el);
    }
  };
}

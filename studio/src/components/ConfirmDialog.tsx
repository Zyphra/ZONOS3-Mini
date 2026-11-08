/**
 * One modal question with Cancel and a confirming button, asked from anywhere (`confirm()` resolves to the answer).
 * Mounted once by the App shell. Focus goes to Cancel, stays inside, and returns to the opener afterwards.
 */
import { signal } from '@preact/signals';
import { useLayoutEffect, useRef } from 'preact/hooks';
import { modalOpen } from '../state/app';
import { dialogKeys } from './focusTrap';

export interface ConfirmOptions {
  /** The confirming button's label (default "Continue"). */
  ok?: string;
  /** The action discards or deletes something: the button is red. */
  danger?: boolean;
  title?: string;
}

interface Pending extends ConfirmOptions {
  text: string;
  resolve: (yes: boolean) => void;
}

const pending = signal<Pending | null>(null);

/** Ask the reader; a second question while one is open answers the first with "no". */
export function confirm(text: string, opts: ConfirmOptions = {}): Promise<boolean> {
  pending.value?.resolve(false);
  const { promise, resolve } = Promise.withResolvers<boolean>();
  pending.value = { ...opts, text, resolve };
  return promise;
}

export function ConfirmDialog() {
  const q = pending.value;
  return q ? <Dialog q={q} /> : null;
}

function Dialog({ q }: { q: Pending }) {
  const box = useRef<HTMLDivElement>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  const answer = (yes: boolean) => {
    if (pending.peek() !== q) return;
    pending.value = null;
    q.resolve(yes);
  };
  useLayoutEffect(() => {
    const back = document.activeElement as HTMLElement | null;
    const wasModal = modalOpen.peek();
    modalOpen.value = true;
    cancel.current?.focus({ preventScroll: true });
    return () => {
      modalOpen.value = wasModal;
      if (back?.isConnected) back.focus({ preventScroll: true });
    };
  }, []);
  const onKey = dialogKeys(() => box.current, () => answer(false));
  return (
    <div class="confirm-backdrop" onClick={(e) => e.target === e.currentTarget && answer(false)}>
      <div ref={box} class="confirm-dialog" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" aria-describedby="confirm-text" onKeyDown={onKey}>
        {q.title && <h2 id="confirm-title">{q.title}</h2>}
        <p id="confirm-text" class={q.title ? undefined : 'confirm-only'}>
          {q.text}
        </p>
        <div class="confirm-actions">
          <button ref={cancel} class="btn sm ghost" onClick={() => answer(false)}>
            Cancel
          </button>
          <button class={`btn sm ${q.danger ? 'proj-danger' : 'primary'}`} onClick={() => answer(true)}>
            {q.ok ?? 'Continue'}
          </button>
        </div>
      </div>
    </div>
  );
}

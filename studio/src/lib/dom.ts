/** Input types that take no text: keys on them (Space, letters) belong to the screen, not the control. */
const NON_TEXT_INPUT = /^(checkbox|radio|range|button|submit|reset|file|color)$/i;

/** The event target is entering text (shortcuts must stay out of its way). */
export function isTyping(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  if (!el || typeof el.tagName !== 'string') return false;
  if (el.isContentEditable || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') return true;
  return el.tagName === 'INPUT' && !NON_TEXT_INPUT.test((el as HTMLInputElement).type);
}

/**
 * The text of a `contenteditable="plaintext-only"` element: its text nodes, a `<br>` as a line break, without the
 * extra line break browsers keep after a trailing newline. Reads the DOM only (no layout), so it works while hidden.
 */
export function editableText(el: HTMLElement): string {
  let t = '';
  for (const n of el.childNodes) t += n.nodeName === 'BR' ? '\n' : (n.textContent ?? '');
  return t.endsWith('\n') ? t.slice(0, -1) : t;
}

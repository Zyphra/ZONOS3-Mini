/** Input types that take no text: keys on them (Space, letters) belong to the screen, not the control. */
const NON_TEXT_INPUT = /^(checkbox|radio|range|button|submit|reset|file|color)$/i;

/** The event target is entering text (shortcuts must stay out of its way). */
export function isTyping(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  if (!el || typeof el.tagName !== 'string') return false;
  if (el.isContentEditable || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') return true;
  return el.tagName === 'INPUT' && !NON_TEXT_INPUT.test((el as HTMLInputElement).type);
}

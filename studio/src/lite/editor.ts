/**
 * The Script box: a `contenteditable="plaintext-only"` element the browser edits natively (undo, IME, spellcheck
 * off), colored with the CSS Custom Highlight API from `./assist`, with the completion popup.
 */
import { editableText } from '../lib/dom';
import { clamp } from '../lib/math';
import { remapSpans, type Span } from '../state/requests';
import { completeAt, highlight, TOKEN_CLASSES, type AssistEnv, type Completion, type ItemKind, type TokenClass } from './assist';
import type { Issue } from './script';

export interface ScriptEditor {
  /** The compile environment: suggestions and voice coloring follow it. */
  setEnv(env: AssistEnv): void;
  /** The last check of the current text: errors and warnings get wavy underlines on their ranges (line 0 has none). */
  setIssues(result: { errors: Issue[]; warnings: Issue[] }): void;
}

const KIND_GLYPH: Record<ItemKind, string> = { directive: '@', value: '·', control: '●', keyword: '{', preset: '★', accent: '◆', voice: '♪', tag: '<>' };

/** Firefox before 149 has no Highlight API: it edits and completes, uncolored. */
const painting = typeof Highlight === 'function';

export function mountScriptEditor({ el, root, popup }: { el: HTMLElement; root: HTMLElement; popup: HTMLElement }): ScriptEditor {
  const list = popup.querySelector('ul')!;
  const info = popup.querySelector<HTMLElement>('.sc-info')!;
  const keys = popup.querySelector<HTMLElement>('.sc-keys')!;
  let env: AssistEnv | null = null;
  /** The last check's issue ranges, carried across edits made since (`remapSpans`). */
  let issueSpans: Span[] = [];
  /** The text `issueSpans` are offsets into. */
  let issueText = '';
  let open: { c: Completion; active: number } | null = null;
  /** An accepted item is being inserted: its input event must not reopen the popup. */
  let accepting = false;

  // Offsets ↔ DOM points. The children are text nodes (the browser doesn't merge them) and maybe `<br>`s; never
  // `normalize()` while editing: it rewrites the DOM outside the undo stack.
  const len = (n: Node) => (n.nodeType === Node.TEXT_NODE ? (n as Text).length : n.nodeName === 'BR' ? 1 : (n.textContent ?? '').length);
  const toPoint = (off: number): [Node, number] => {
    const kids = el.childNodes;
    let acc = 0;
    for (let i = 0; i < kids.length; i++) {
      const n = kids[i];
      if (off <= acc + len(n)) return n.nodeType === Node.TEXT_NODE ? [n, off - acc] : [el, off === acc ? i : i + 1];
      acc += len(n);
    }
    return [el, kids.length];
  };
  const fromPoint = (node: Node, o: number): number => {
    const kids = el.childNodes;
    let s = 0;
    if (node === el) {
      for (let i = 0; i < o; i++) s += len(kids[i]);
      return s;
    }
    let k = node;
    while (k.parentNode && k.parentNode !== el) k = k.parentNode;
    for (const n of kids) {
      if (n === k) break;
      s += len(n);
    }
    return s + (node.nodeType === Node.TEXT_NODE ? o : 0);
  };
  /** The collapsed caret's offset in the text, or null. */
  const caret = (): number | null => {
    const sel = getSelection();
    if (!sel?.rangeCount || !sel.isCollapsed || !sel.focusNode || !el.contains(sel.focusNode)) return null;
    return fromPoint(sel.focusNode, sel.focusOffset);
  };
  const rangeOf = (from: number, to: number) => {
    const r = document.createRange();
    r.setStart(...toPoint(from));
    r.setEnd(...toPoint(to));
    return r;
  };
  const select = (from: number, to: number) => {
    const sel = getSelection();
    sel?.removeAllRanges();
    sel?.addRange(rangeOf(from, to));
  };

  function paint() {
    if (!painting) return;
    const v = editableText(el);
    const range = (a: number, b: number) => rangeOf(Math.min(a, v.length), Math.min(b, v.length));
    const by = new Map<TokenClass, Range[]>(TOKEN_CLASSES.map((c) => [c, []]));
    for (const t of env ? highlight(v, env) : []) by.get(t.cls)!.push(range(t.from, t.to));
    for (const [cls, ranges] of by) CSS.highlights.set(`sx-${cls}`, new Highlight(...ranges));
    if (issueText !== v) {
      issueSpans = remapSpans(issueSpans, issueText, v);
      issueText = v;
    }
    const issues = (kind: string, priority: number) => {
      const h = new Highlight(...issueSpans.filter((s) => s.id.startsWith(kind)).map((s) => range(s.start, s.end)));
      h.priority = priority;
      CSS.highlights.set(`sx-${kind}`, h);
    };
    issues('warn', 1);
    issues('error', 2);
  }

  /** Where the text at `off` is drawn: the caret there, else its character, else the box's first line. */
  function caretRect(off: number): { left: number; top: number; bottom: number } {
    const at = rangeOf(off, off).getClientRects()[0];
    if (at) return at;
    if (off < editableText(el).length) {
      const ch = rangeOf(off, off + 1).getClientRects()[0];
      if (ch) return ch;
    }
    const box = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    const top = box.top + parseFloat(cs.paddingTop);
    return { left: box.left + parseFloat(cs.paddingLeft), top, bottom: top + parseFloat(cs.lineHeight) };
  }

  function setActive(i: number) {
    if (!open) return;
    open.active = i;
    const item = open.c.items[i];
    list.querySelectorAll('li').forEach((li, k) => {
      li.classList.toggle('is-active', k === i);
      li.setAttribute('aria-selected', String(k === i));
    });
    const li = list.children[i] as HTMLElement | undefined;
    li?.scrollIntoView({ block: 'nearest' });
    if (li) el.setAttribute('aria-activedescendant', li.id);
    else el.removeAttribute('aria-activedescendant');
    info.textContent = item?.info ?? open.c.hint ?? '';
    info.hidden = !info.textContent;
  }

  function show() {
    if (!open) return;
    const { c } = open;
    list.replaceChildren(
      ...c.items.map((item, i) => {
        const li = document.createElement('li');
        li.id = `script-complete-${i}`;
        li.setAttribute('role', 'option');
        li.dataset.i = String(i);
        const kind = Object.assign(document.createElement('span'), { className: 'sc-kind', textContent: KIND_GLYPH[item.kind] });
        if (item.hue) kind.style.color = `var(--${item.hue})`;
        li.append(kind, Object.assign(document.createElement('span'), { className: 'sc-label', textContent: item.label }), Object.assign(document.createElement('span'), { className: 'sc-detail', textContent: item.detail }));
        return li;
      }),
    );
    list.hidden = keys.hidden = !c.items.length;
    el.setAttribute('aria-expanded', 'true');
    popup.hidden = false;
    setActive(0);
    const r = caretRect(c.from);
    const box = root.getBoundingClientRect();
    const h = popup.offsetHeight;
    const flip = r.bottom + 4 + h > innerHeight - 8 && r.top - 4 - h >= 8;
    popup.style.left = `${clamp(r.left - box.left, 0, Math.max(0, box.width - popup.offsetWidth))}px`;
    popup.style.top = `${flip ? r.top - box.top - 4 - h : r.bottom - box.top + 4}px`;
  }

  function close() {
    popup.hidden = true;
    open = null;
    el.setAttribute('aria-expanded', 'false');
    el.removeAttribute('aria-activedescendant');
  }

  /** Opens (or refreshes) the popup for the caret, or closes it when the caret's place offers nothing. */
  function update() {
    const k = caret();
    const c = k === null || !env ? null : completeAt(editableText(el), k, env);
    if (!c) return close();
    open = { c, active: 0 };
    show();
  }

  function accept(i: number) {
    if (!open) return;
    const { from, to } = open.c;
    const item = open.c.items[i];
    const v = editableText(el);
    accepting = true;
    select(from, to);
    // One native undo step; `insert` never holds a newline (that would insert a <div>).
    if (!document.execCommand('insertText', false, item.insert)) {
      // An engine without insertText: the edit leaves the undo stack.
      el.textContent = v.slice(0, from) + item.insert + v.slice(to);
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }
    accepting = false;
    const p = from + (item.caret ?? item.insert.length);
    select(p, p);
    // Chains: `happy ` says what it takes, `@language ` lists the languages.
    update();
  }

  el.addEventListener('input', (e) => {
    if ([...el.children].some((c) => c.nodeName !== 'BR')) {
      // Markup from another engine or an extension: flatten to text (leaves the undo stack; a fallback only).
      const k = caret() ?? 0;
      const v = el.innerText.replace(/\n$/, '');
      const tail = Math.max(0, v.length - k);
      el.textContent = v;
      select(v.length - tail, v.length - tail);
    }
    paint();
    if (accepting) return;
    const t = (e as InputEvent).inputType;
    if (t === 'insertText') update();
    else if (t?.startsWith('delete') && open) update();
    // Enter, paste, composition.
    else close();
  });
  el.addEventListener('paste', (e) => {
    e.preventDefault();
    const t = (e.clipboardData?.getData('text/plain') ?? '').replace(/\r\n?/g, '\n');
    // One text node with its newlines, one undo step (insertText would turn each newline into a <div>).
    document.execCommand('insertHTML', false, t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'));
  });
  el.addEventListener('keydown', (e) => {
    if (e.isComposing) return;
    if (e.key === ' ' && e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      return update();
    }
    if (!open) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      return close();
    }
    const n = open.c.items.length;
    if (!n) return;
    switch (e.key) {
      case 'ArrowDown':
      case 'ArrowUp':
        e.preventDefault();
        setActive((open.active + (e.key === 'ArrowDown' ? 1 : n - 1)) % n);
        break;
      case 'Tab':
        if (e.shiftKey) break;
        e.preventDefault();
        accept(open.active);
        break;
      case 'ArrowLeft':
      case 'ArrowRight':
      case 'Home':
      case 'End':
      case 'PageUp':
      case 'PageDown':
        close();
        break;
    }
  });
  for (const type of ['blur', 'scroll', 'pointerdown']) el.addEventListener(type, close);
  // Clicks on the popup keep the focus (and the caret) in the box.
  popup.addEventListener('pointerdown', (e) => e.preventDefault());
  const row = (e: Event) => (e.target as Element).closest<HTMLElement>('li[data-i]');
  list.addEventListener('click', (e) => {
    const li = row(e);
    if (li) accept(Number(li.dataset.i));
  });
  list.addEventListener('pointermove', (e) => {
    const li = row(e);
    if (li && open && Number(li.dataset.i) !== open.active) setActive(Number(li.dataset.i));
  });
  addEventListener('resize', close);

  el.setAttribute('aria-autocomplete', 'list');
  el.setAttribute('aria-controls', 'script-complete-list');
  el.setAttribute('aria-expanded', 'false');
  paint();

  return {
    setEnv(e) {
      env = e;
      paint();
    },
    setIssues({ errors, warnings }) {
      const spans = (kind: string, issues: Issue[]) => issues.filter((x) => x.line > 0 && x.to > x.from).map((x, i): Span => ({ id: `${kind}:${i}`, start: x.from, end: x.to, values: {} }));
      issueSpans = [...spans('error', errors), ...spans('warn', warnings)];
      issueText = editableText(el);
      paint();
    },
  };
}

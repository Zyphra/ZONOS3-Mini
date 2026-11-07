/**
 * Pronunciation input UI shared by every text that is synthesized (Direct lines, the Cast preview
 * line): IPA chips for "(X-SAMPA)" groups, flagged problems, "Pronounce…" (lexicon lookup of a selection via
 * /v2/phonemize) and a small editor for one group.
 */
import { computed, effect, signal } from '@preact/signals';
import { useEffect, useRef, useState } from 'preact/hooks';
import * as api from '../api';
import { errorText } from '../audio/render';
import { clamp } from '../lib/math';
import { ipaToXsampa, parsePron, xsampaToIpa } from '../lib/pron';
import { backend, notify, projectSettings, pronTokens } from '../state/app';
import { Icon } from './Icon';

export const SERVE_HINT = 'SERVE_MODEL=<phoneme checkpoint> PHONEME_LEXICON=<pronunciation lexicon dir>';

/** A render refused locally for its pronunciation groups (the input's marks already list the problems). */
export const isPronError = (e: string | null | undefined) => !!e?.startsWith('Pronunciation:');

export const pronOn = computed(() => !!pronTokens.value);
export const phonemizeOn = computed(() => pronOn.value && !!backend.value?.pron.phonemize_enabled);
const languages = computed(() => backend.value?.pron.phonemize_languages ?? []);
/** The lexicon language: the project's language, until picked in the editor (for the session). */
export const pronLanguage = signal<string>(projectSettings.value.language);
effect(() => void (pronLanguage.value = projectSettings.value.language));
export const lookupLanguage = computed(() => (languages.value.includes(pronLanguage.value) ? pronLanguage.value : (languages.value.includes('en_us') ? 'en_us' : languages.value[0] ?? 'en_us')));

/** Groups the lexicon only guessed (espeak-ng), by X-SAMPA: shown with "check" until confirmed. */
export const guessed = signal<Record<string, string>>({});

import { langLabel } from '../lib/languages';

// ── Phonemize ─────────────────────────────────────────────────────────────────────────────────────

export const phonemizing = signal(false);

/**
 * Replace the words of text[start, end) with X-SAMPA groups from the lexicon. Existing groups are kept. Returns
 * the new text (null when nothing changed or the lookup failed); explains what was left as text.
 */
export async function phonemizeRange(text: string, start: number, end: number): Promise<string | null> {
  if (end <= start) {
    notify('Select the words to pronounce first.');
    return null;
  }
  const language = lookupLanguage.value;
  const parts = text.slice(start, end).split(/(\([^()]*\))/);
  phonemizing.value = true;
  try {
    const results = await Promise.all(
      parts.map(async (part, i) => (i % 2 === 1 || !part.trim() ? [{ text: part, status: null, ipa: null }] : (await api.phonemize(part, language, true)).segments)),
    );
    const segs = results.flat();
    const marks: Record<string, string> = { ...guessed.value };
    const out = segs
      .map((s) => {
        if (!s.ipa || (s.status !== 'ok' && s.status !== 'guessed')) return s.text;
        const x = ipaToXsampa(s.ipa);
        if (s.status === 'guessed') marks[x] = s.text;
        else delete marks[x];
        return `(${x})`;
      })
      .join('');
    guessed.value = marks;
    const words = (st: string) => segs.filter((s) => s.status === st).map((s) => s.text);
    const count = words('ok').length + words('guessed').length;
    const notes: string[] = [];
    if (words('ambiguous').length) notes.push(`several pronunciations: ${words('ambiguous').join(', ')}`);
    if (words('unknown').length) notes.push(`not in the lexicon: ${words('unknown').join(', ')}`);
    if (words('skipped').length) notes.push(`no reading on its own: ${words('skipped').join(', ')}`);
    if (words('unsupported').length) notes.push(`symbols this model has no token for: ${words('unsupported').join(', ')}`);
    const check = words('guessed').length ? ` Guessed by rule, check: ${words('guessed').join(', ')}.` : '';
    notify(
      `Pronounced ${count} word${count === 1 ? '' : 's'} (${langLabel(language)}).${check}${notes.length ? ` Left as text (${notes.join('; ')}): write their X-SAMPA by hand.` : ''}`,
      notes.length || check ? 'info' : 'ok',
    );
    const next = text.slice(0, start) + out + text.slice(end);
    return count ? next : null;
  } catch (e) {
    notify(`Couldn’t look up pronunciations: ${errorText(e)}`, 'error');
    return null;
  } finally {
    phonemizing.value = false;
  }
}

/** Replace the input's current selection with `text` so the browser's own undo (⌘Z while focused) takes it back. */
export function replaceInInput(el: HTMLInputElement | HTMLTextAreaElement, start: number, end: number, text: string) {
  el.focus();
  el.setSelectionRange(start, end);
  if (!document.execCommand('insertText', false, text)) {
    el.setRangeText(text, start, end, 'end');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }
}

/** "Pronounce…" for an input: looks up the selected words and swaps them for X-SAMPA groups (undo: ⌘Z). */
export function PronounceButton({ input, onDone }: { input: () => HTMLInputElement | HTMLTextAreaElement | null; onDone?: () => void }) {
  if (!phonemizeOn.value) return null;
  return (
    <button
      class="icon-btn sm pron-btn"
      disabled={phonemizing.value}
      onMouseDown={(e) => e.preventDefault()}
      onClick={async () => {
        const el = input();
        if (!el) return;
        const s = el.selectionStart ?? 0;
        const e = el.selectionEnd ?? 0;
        const before = el.value;
        const next = await phonemizeRange(before, s, e);
        if (next === null || el.value !== before) return;
        // Replace only the selected stretch: the rest of the text is untouched.
        const tail = before.length - e;
        replaceInInput(el, s, e, next.slice(s, next.length - tail));
        onDone?.();
      }}
      aria-label="Pronounce the selected words"
      title={`Pronounce…: look up the selected words in the ${langLabel(lookupLanguage.value)} lexicon and write them as X-SAMPA`}
    >
      <span aria-hidden="true" class="pron-btn-glyph">/ə/</span>
    </button>
  );
}

// ── Group editor ──────────────────────────────────────────────────────────────────────────────────

interface EditState {
  x: number;
  y: number;
  start: number;
  end: number;
  body: string;
  replace: (start: number, end: number, text: string) => void;
  returnFocus: HTMLElement | null;
}
const editing = signal<EditState | null>(null);

export function openPronEditor(anchor: HTMLElement, group: { start: number; end: number; body: string }, replace: EditState['replace']) {
  const r = anchor.getBoundingClientRect();
  editing.value = { x: r.left, y: r.bottom + 6, ...group, replace, returnFocus: anchor };
}

/** The X-SAMPA field with live IPA, each symbol marked when this checkpoint has no token for it. */
export function PronEditor() {
  const st = editing.value;
  return st ? <GroupEditor key={`${st.start}:${st.body}:${st.x}:${st.y}`} st={st} /> : null;
}

function GroupEditor({ st }: { st: EditState }) {
  const [value, setValue] = useState(st.body);
  const box = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLInputElement>(null);
  useEffect(() => {
    requestAnimationFrame(() => field.current?.select());
    const outside = (e: PointerEvent) => !box.current?.contains(e.target as Node) && close();
    window.addEventListener('pointerdown', outside);
    return () => window.removeEventListener('pointerdown', outside);
  }, []);
  const tokens = pronTokens.value;
  const known = new Set(tokens?.symbols ?? []);
  const ipa = xsampaToIpa(value);
  const bad = [...ipa].filter((c) => !known.has(c));
  const close = () => {
    const f = editing.value?.returnFocus;
    editing.value = null;
    f?.focus?.();
  };
  const save = () => {
    if (!ipa || bad.length) return;
    const g = `(${value.trim()})`;
    st.replace(st.start, st.end, g);
    if (guessed.value[value.trim()] !== undefined || guessed.value[st.body] !== undefined) {
      const { [st.body]: _, [value.trim()]: __, ...rest } = guessed.value;
      guessed.value = rest;
    }
    close();
  };
  const left = clamp(st.x, 12, window.innerWidth - 352);
  const top = st.y + 220 > window.innerHeight ? Math.max(12, st.y - 250) : st.y;
  return (
    <div ref={box} class="pron-editor" style={{ left: `${left}px`, top: `${top}px` }} role="dialog" aria-label="Edit pronunciation">
      <label class="pron-field">
        <span class="eyebrow">X-SAMPA</span>
        <input
          ref={field}
          class="input mono"
          value={value}
          spellcheck={false}
          aria-describedby="pron-ipa"
          onInput={(e) => setValue((e.currentTarget as HTMLInputElement).value.replace(/[()]/g, ''))}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.preventDefault(), save());
            if (e.key === 'Escape') (e.preventDefault(), e.stopPropagation(), close());
          }}
        />
      </label>
      <div class="pron-ipa" id="pron-ipa" aria-live="polite">
        <span class="eyebrow">IPA sent</span>
        <span class="pron-ipa-symbols">
          {ipa ? [...ipa].map((c) => <span class={known.has(c) ? '' : 'is-bad'} title={known.has(c) ? `${c} has a token` : `no token for ${c} (U+${c.codePointAt(0)!.toString(16).toUpperCase().padStart(4, '0')})`}>{c}</span>) : <span class="muted">empty</span>}
        </span>
      </div>
      {bad.length > 0 && <p class="pron-warn">No token on this model for {[...new Set(bad)].map((c) => `“${c}”`).join(' ')}: the server would refuse it.</p>}
      {guessed.value[st.body] && <p class="pron-check">Guessed by rule for “{guessed.value[st.body]}”: check it and save to confirm.</p>}
      <p class="hint">“_” ties (t_S = t͡ʃ); stress " % and dots are dropped. Symbols without an X-SAMPA spelling are typed as-is.</p>
      <div class="row gap-s">
        <button class="btn sm ghost" onClick={close}>
          Cancel
        </button>
        <button class="btn sm primary" disabled={!ipa || bad.length > 0} onClick={save}>
          Save
        </button>
      </div>
    </div>
  );
}

// ── Chips ─────────────────────────────────────────────────────────────────────────────────────────

/**
 * The groups of `text` as IPA chips (click to edit), and what would make the server refuse it. Without phoneme
 * tokens: one quiet note that parentheses are read as text.
 */
export function PronMarks({ text, replace, issuesOnly }: { text: string; replace?: EditState['replace']; issuesOnly?: boolean }) {
  const tokens = pronTokens.value;
  const parsed = parsePron(text, tokens);
  if (!tokens) {
    return parsed.groups.length ? (
      <span class="pron-marks">
        <span class="pron-off" title={`Pronunciation input needs the phoneme-token checkpoint: ${SERVE_HINT}`}>
          ( ) read as text
        </span>
      </span>
    ) : null;
  }
  if (!parsed.issues.length && (issuesOnly || !parsed.groups.length)) return null;
  return (
    <span class="pron-marks" aria-label="Pronunciations">
      {parsed.groups
        .filter((g) => !g.empty && !issuesOnly)
        .map((g) => {
          const check = guessed.value[g.body];
          return (
            <button
              class={`pron-chip${g.unknown.length ? ' is-bad' : ''}${check ? ' is-check' : ''}`}
              title={`(${g.body}) → ${g.ipa}${check ? ` · guessed for “${check}”, check it` : ''}${g.unknown.length ? ` · no token for ${g.unknown.join(' ')}` : ''}${replace ? ' · click to edit' : ''}`}
              disabled={!replace}
              onClick={(e) => replace && openPronEditor(e.currentTarget as HTMLElement, g, replace)}
            >
              <span class="pron-chip-ipa">{g.ipa}</span>
              {check && <span class="pron-chip-check">check</span>}
            </button>
          );
        })}
      {parsed.issues
        .filter((x) => issuesOnly || !parsed.groups.some((g) => g.start === x.at && g.unknown.length && !g.empty))
        .map((x) => (
          <span class="pron-issue" role="note">
            <Icon name="x" size={10} /> {x.text}
          </span>
        ))}
    </span>
  );
}

/** Rail section: how to write pronunciations, the lookup language, or why it is off. */
export function PronHelp() {
  const b = backend.value;
  if (!b) return null;
  if (!pronOn.value) {
    return (
      <p class="hint pron-help" title={SERVE_HINT}>
        Pronunciation input (X-SAMPA in parentheses) needs the phoneme-token checkpoint; on this server parentheses are read as text.
      </p>
    );
  }
  return (
    <div class="pron-help">
      <p class="hint">
        Write a pronunciation in X-SAMPA between parentheses: “I met <span class="mono">(S@vQn)</span>” reads <span class="mono">ʃəvɒn</span>.
        {phonemizeOn.value ? ' Or select words and press /ə/ to look them up.' : ''}
      </p>
      {phonemizeOn.value && languages.value.length > 1 && (
        <label class="pron-lang">
          lexicon
          <select class="input" value={lookupLanguage.value} onChange={(e) => (pronLanguage.value = (e.currentTarget as HTMLSelectElement).value)} aria-label="Pronunciation lexicon language">
            {languages.value.map((l) => (
              <option value={l}>{langLabel(l)}</option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}

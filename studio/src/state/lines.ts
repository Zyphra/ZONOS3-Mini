/** Direct's lines without the signals: splitting pasted text, matching edited rows to lines, why a take is stale. */
import type { GenerateRequest } from '../api';
import { uid } from '../lib/math';
import { remapSpans, type Span } from './requests';

export interface Line {
  id: string;
  text: string;
  spans: Span[];
  /** Persona id; absent / null: the script's voice. */
  voice?: string | null;
  /** Speech from the first frame (start_immediately); absent = off. */
  startImmediately?: boolean;
}

/** Silence between lines when a script plays or exports (s): the default and the gap slider's range. */
export const DEFAULT_GAP = 0.35;
export const GAP_RANGE: [number, number] = [0, 1.5];

export const newLine = (text: string, voice?: string | null): Line => ({ id: uid(), text, spans: [], ...(voice ? { voice } : {}) });

/** Rows of a pasted script: one per line break, and (unless `sentences: false`) per sentence end inside a line. */
export function splitScript(text: string, opts: { sentences?: boolean } = {}): string[] {
  const paras = text.split(/\n+/);
  return (opts.sentences === false ? paras : paras.flatMap((para) => para.split(/(?<=[.!?…])\s+(?=\S)/))).map((s) => s.trim()).filter(Boolean);
}

/**
 * Rows of edited text matched to the lines they came from. Each old line is used once: first by identical text
 * (in order), then by position (same index, different text: the line keeps its id, voice and painting, and its
 * spans follow the words). `dropped` are the old lines no row claimed.
 */
export function matchLines(old: Line[], rows: string[]): { lines: Line[]; dropped: Line[] } {
  const byText = new Map<string, Line[]>();
  for (const l of old) byText.set(l.text, [...(byText.get(l.text) ?? []), l]);
  const used = new Set<string>();
  const exact = rows.map((t) => {
    const hit = byText.get(t)?.shift() ?? null;
    if (hit) used.add(hit.id);
    return hit;
  });
  const lines = exact.map((l, i) => {
    if (l) return l;
    const t = rows[i];
    const at = old[i];
    if (at && !used.has(at.id)) {
      used.add(at.id);
      return { ...at, text: t, spans: remapSpans(at.spans, at.text, t) };
    }
    return newLine(t);
  });
  return { lines, dropped: old.filter((l) => !used.has(l.id)) };
}

/**
 * Why a take no longer matches its line (the first difference found), or null when it still does.
 * `voiceEdited`: the line's cast voice was reshaped in Cast since (the whole-line controls differ because of the
 * voice's own design, not the script baseline).
 */
export function staleReason(stored: GenerateRequest, fresh: GenerateRequest, opts: { voiceEdited?: boolean } = {}): string | null {
  if (stored.text !== fresh.text) return 'text changed';
  if (stored.speaker_embedding_id !== fresh.speaker_embedding_id || stored.speaker_mode !== fresh.speaker_mode) return 'voice changed';
  if ((stored.language ?? 'en_us') !== (fresh.language ?? 'en_us')) return 'language changed';
  if ((stored.start_immediately ?? false) !== (fresh.start_immediately ?? false)) return 'start changed';
  if (JSON.stringify(stored.control_spans ?? []) !== JSON.stringify(fresh.control_spans ?? [])) return 'painting changed';
  // The whole-line controls: the voice's design edited since, or the script baseline.
  if (JSON.stringify(stored.controls ?? {}) !== JSON.stringify(fresh.controls ?? {})) return opts.voiceEdited ? 'voice edited' : 'baseline changed';
  if ((stored.laci ?? false) !== (fresh.laci ?? false) || (stored.stream_decode ?? null) !== (fresh.stream_decode ?? null)) return 'LACI changed';
  if ((stored.decoder ?? null) !== (fresh.decoder ?? null)) return 'decoder changed';
  return null;
}

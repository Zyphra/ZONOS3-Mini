/**
 * The player widget's input: `structuredContent` of every widget tool. Imported by the server and the widget.
 * Small on purpose (no audio): the widget fetches each take's WAV through `zonos_take_audio`, so audio never
 * enters the model's context.
 */
export interface WidgetPayload {
  kind: 'voice' | 'speak' | 'edit';
  /** speak: the Direct line id, so Regenerate adds a take to the same line. */
  lineId?: string;
  /** e.g. 'Gravelly narrator · preview', 'Line 3', 'Edit · Line 1 · take 2'. */
  title: string;
  /** The text as spoken (line text / target text). */
  text: string;
  takes: TakeInfo[];
  params: GenerationParams;
  project: { id: string; name: string; revision: number; url: string } | null;
  warnings: string[];
}

export interface TakeInfo {
  id: string;
  label: string;
  media?: string;
  seed: number | null;
  durationSec: number;
  totalMs?: number;
  ttfbMs?: number;
  spanFollow?: 'rate' | 'reading';
  /** Edit takes. */
  verify?: { wer: number; cer: number; asr_text: string } | null;
  /** The edit's source version. */
  original?: boolean;
  keepable?: boolean;
  canSendBack?: boolean;
  /** speak: the line's starred take. */
  starred?: boolean;
}

export interface GenerationParams {
  voice: { speakerId: string; label: string; mode: 'embedding' | 'reference'; persona?: { id: string; name: string } };
  language: string;
  values: { id: string; label: string; group: string; value: number; zone: 'safe' | 'strong' | 'collapse'; poles?: [string, string]; dropped?: boolean }[];
  accents: { id: string; label: string; requested?: number; applied: number }[];
  spans: { start: number; end: number; text: string; values: { id: string; label: string; value: number }[] }[];
  /** request.controls */
  doses: Record<string, number>;
  controlSpans: { start: number; end: number; controls: Record<string, number> }[];
  identityEstimate: number | null;
  durationFactor: number;
  /** temperature, topk, top_p, min_p, repetition_*, speaker_cfg_scale, tag_cfg_scale, stream_decode. */
  sampling: Record<string, number | string | null>;
  /** The exact body sent; audio_base64 replaced by '<wav, N bytes>'. */
  request: Record<string, unknown>;
}

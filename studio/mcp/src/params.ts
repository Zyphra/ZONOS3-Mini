/** Widget payload pieces and the concise text descriptions the model reads. */
import openapi from '../../../servers/api/openapi.json';
import type { ControlsSpec, GenerateRequest, Speaker } from '../../src/api';
import { parseWav } from '../../src/lib/wav';
import { langLabel } from '../../src/lib/languages';
import type { Design } from '../../src/state/design';
import type { AccentMix, Values } from '../../src/state/engine';
import type { Persona } from '../../src/state/library';
import type { Span } from '../../src/state/requests';
import { designView, type DesignView } from './design';
import type { GenerationParams, TakeInfo } from './payload';

const SAMPLING = ['temperature', 'topk', 'top_p', 'min_p', 'repetition_window', 'repetition_penalty', 'repetition_codebooks', 'speaker_cfg_scale', 'tag_cfg_scale', 'stream_decode'];

/** Each sampling key as sent, else the API's default for it. */
export function samplingOf(req: Record<string, unknown>, schema: 'GenerateRequest' | 'EditRequest'): GenerationParams['sampling'] {
  const props: Record<string, object> = openapi.components.schemas[schema].properties;
  const out: GenerationParams['sampling'] = {};
  for (const k of SAMPLING) {
    const prop = props[k];
    if (!prop && !(k in req)) continue;
    const v = req[k] ?? (prop && 'default' in prop ? prop.default : null);
    out[k] = typeof v === 'number' || typeof v === 'string' ? v : null;
  }
  return out;
}

/** Seconds of a WAV. */
export function wavSeconds(wav: Uint8Array): number {
  const { pcm, sr } = parseWav(wav.buffer.slice(wav.byteOffset, wav.byteOffset + wav.byteLength) as ArrayBuffer);
  return Math.round((pcm.length / sr) * 100) / 100;
}

export function paintedSpans(spec: ControlsSpec, text: string, spans: Span[]): GenerationParams['spans'] {
  return spans.map((sp) => ({
    start: sp.start,
    end: sp.end,
    text: text.slice(sp.start, sp.end),
    values: Object.entries(sp.values)
      .filter(([, v]) => v)
      .map(([id, value]) => ({ id, label: spec.ui.controls.find((c) => c.id === id)?.label ?? id, value })),
  }));
}

export interface ParamsInput {
  spec: ControlsSpec;
  design: Design;
  speakers: Speaker[];
  persona?: Persona;
  language: string;
  request: GenerateRequest;
  text: string;
  spans?: Span[];
  requestedAccents?: AccentMix;
  view?: DesignView;
}

/** Everything a generate request used, for the widget. */
export function generationParams(p: ParamsInput): GenerationParams {
  const view = p.view ?? designView(p.spec, p.design);
  const base = p.speakers.find((s) => s.id === p.design.baseId);
  const req = p.request;
  return {
    voice: { speakerId: p.design.baseId, label: base?.label ?? p.design.baseId, mode: p.design.mode, ...(p.persona ? { persona: { id: p.persona.id, name: p.persona.name } } : {}) },
    language: p.language,
    values: view.values,
    accents: view.accents.map((a) => ({ ...a, ...(p.requestedAccents?.[a.id] !== undefined && p.requestedAccents[a.id] !== a.applied ? { requested: p.requestedAccents[a.id] } : {}) })),
    spans: paintedSpans(p.spec, p.text, p.spans ?? []),
    doses: req.controls ?? {},
    controlSpans: req.control_spans ?? [],
    identityEstimate: Math.round(view.identity * 1000) / 1000,
    durationFactor: Math.round(view.duration * 1000) / 1000,
    sampling: samplingOf({ ...req }, 'GenerateRequest'),
    request: { ...req },
  };
}

// ── Text ──────────────────────────────────────────────────────────────────────────────────────────

const signed = (v: number) => (v > 0 ? `+${v}` : `${v}`);

export function valuesText(values: GenerationParams['values']): string {
  return values.map((v) => `${v.label.toLowerCase()} ${signed(v.value)} ${v.dropped ? 'dropped' : v.zone}`).join(', ');
}

export function accentsText(accents: GenerationParams['accents']): string {
  return accents.map((a) => `accent ${a.id} ${a.applied}${a.requested !== undefined ? ` (asked ${a.requested})` : ''}`).join(', ');
}

export function deliveryText(spec: ControlsSpec, spans: GenerationParams['spans'], textLength: number): string {
  const label = (vals: { id: string; value: number }[]) => vals.map((v) => `${spec.ui.controls.find((c) => c.id === v.id)?.label.toLowerCase() ?? v.id} ${v.value}`).join(', ');
  return spans
    .map((s) => (s.start === 0 && s.end === textLength ? `whole line ${label(s.values)}` : `“${s.text}” ${label(s.values)}`))
    .join('; ');
}

export function voiceText(params: GenerationParams): string {
  const v = params.voice;
  return `voice “${v.persona?.name ?? v.label}” (${v.label}, ${v.mode})`;
}

/** One-line summary of the design part of a result. */
export function designSummary(spec: ControlsSpec, params: GenerationParams, textLength: number): string {
  const parts = [voiceText(params)];
  if (params.values.length) parts.push(valuesText(params.values));
  if (params.accents.length) parts.push(accentsText(params.accents));
  if (params.spans.length) parts.push(`delivery: ${deliveryText(spec, params.spans, textLength)}`);
  if (params.language !== 'en_us') parts.push(langLabel(params.language));
  return parts.join(' · ');
}

export function takeText(t: TakeInfo): string {
  return `${t.label} ${t.id}: ${t.durationSec} s${t.seed != null ? `, seed ${t.seed}` : ''}${t.verify ? `, WER ${Math.round(t.verify.wer * 1000) / 10}%` : ''}`;
}

export const hasValues = (v: Values | undefined) => !!v && Object.values(v).some(Boolean);

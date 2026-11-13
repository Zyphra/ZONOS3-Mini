/**
 * The lite page's Script input: a line-oriented text for what Studio's Cast, Direct and project settings do
 * (designed voices, a voice per line, whole-line and per-word delivery, the script baseline, the gap between lines,
 * language, seed, LACI, decoder), compiled into the requests Studio's own composer (`buildRequest`) sends for the
 * same settings. Pure: no DOM, no signals; the page loads it on first use of Script.
 */
import type { ControlsSpec, SpeakerMode } from '../api';
import { unknownDecoder } from '../lib/decoders';
import { LACI_MODES, type LaciMode } from '../lib/laci';
import { langLabel } from '../lib/languages';
import { pronProblem } from '../lib/pron';
import { voiceServerLanguage } from '../lib/voices';
import { homePin, mixAt, pinForMix, type Pin } from '../state/atlas';
import { controlUsable, presetToValues, type Values } from '../state/engine';
import { DEFAULT_GAP, GAP_RANGE } from '../state/lines';
import { buildRequest, capsContext } from '../state/requests';
import { findSpeaker, layerSpans, modeFor, resolveAccents, resolveValues, type AccentResolution } from '../state/resolve';
import type { Capabilities, GenerateRequest, Speaker } from './api';
import { DIRECTIVES, NUMBER, parseScript, speakerNames, VOICE_NAME_MAX, type ItemList, type Loc, type ScriptLine } from './syntax';

export interface Issue {
  /** 1-based line of the script; 0: the whole script. */
  line: number;
  /** The characters it is about (absolute offsets); 0, 0 for line 0. */
  from: number;
  to: number;
  message: string;
}

export interface ScriptEnv {
  /** `GET /v2/controls`, else `NO_CONTROLS`. */
  spec: ControlsSpec;
  /** False: `/v2/controls` answered 404. */
  steering: boolean;
  /** `GET /v2/speakers`, every mode (not the lite select's embedding-only list). */
  speakers: Speaker[];
  caps: Pick<Capabilities, 'phoneme_tokens' | 'decoders' | 'laci_languages' | 'start_token'>;
  /** The Language select's codes. */
  languages: readonly string[];
  /** Step 1's voice. */
  selected: Speaker | null;
  /** Why step 1 has no voice; null: none needed, or Generate clones the clip first. */
  selectedWhy: string | null;
  /** Step 2's Language select. */
  language: string;
  /** Advanced → Seed (null: blank). */
  seed: number | null;
  /** Advanced → LACI ('off' when the select is hidden or empty). */
  laci: LaciMode;
}

export interface ScriptPart {
  line: number;
  label: string;
  text: string;
  request: GenerateRequest;
}

export interface ScriptRender {
  parts: ScriptPart[];
  /** Silence between parts (s). */
  gap: number;
  seed: number;
}

export interface ScriptResult {
  errors: Issue[];
  warnings: Issue[];
  /** A line speaks with step 1's voice (an unprefixed line, or a @voice without `= library voice`). */
  usesSelected: boolean;
  /** Null with errors, and (without errors) while step 1's voice is a clip still to clone: clone, then compile again. */
  render: ScriptRender | null;
}

/** The spec of a server without a steering bundle: nothing to steer, requests without `controls`. */
export const NO_CONTROLS: ControlsSpec = {
  bundle: '',
  embedding_dim: null,
  sets: {},
  directions: {},
  timing: {},
  ui: { groups: [], controls: [], accents: [], presets: {}, preview_text: '', request_defaults: {}, identity_threshold: 0 },
  steer_release: null,
};

export const NO_STEERING = 'This server has no steering bundle (GET /v2/controls answers 404): controls, presets and accents are off.';
const SEED_MAX = 4294967295;

const isMode = (s: string): s is SpeakerMode => s === 'embedding' || s === 'reference';

/** An error about `loc`, not the whole line. */
class LocError extends Error {
  constructor(
    message: string,
    readonly loc: Loc,
  ) {
    super(message);
  }
}

/** `loc`, or the trimmed line when `loc` is empty. */
function fix(l: { loc: Loc }, loc: Loc): Loc {
  const r = loc.to > loc.from ? loc : l.loc;
  return { from: r.from, to: r.to };
}

type Warn = (messages: readonly string[], loc: Loc) => void;

interface Items {
  values: Record<string, number>;
  accents: Record<string, number>;
  /** The preset's key in `spec.ui.presets`. */
  preset?: string;
  mode?: SpeakerMode;
  /** Where each was written, by control id, `accent:<id>`, `preset` or `mode`; a later repeat wins. */
  locs: Record<string, Loc>;
}

/** `control value, …` (on a @voice also `preset Name`, `accent id strength`, `mode m`); a later repeat wins. */
function readItems(list: ItemList | null, spec: ControlsSpec, steering: boolean, onVoice: boolean): Items {
  const out: Items = { values: {}, accents: {}, locs: {} };
  for (const item of list?.items ?? []) {
    const words = [item.head.text, ...item.args.map((w) => w.text)];
    const head = words[0].toLowerCase();
    const special = head === 'preset' || head === 'accent' || head === 'mode';
    if (special && !onVoice) throw new LocError('preset, accent and mode go on a @voice line.', item);
    if (head === 'mode') {
      const mode = words[1]?.toLowerCase() ?? '';
      if (words.length !== 2 || !isMode(mode)) throw new LocError('mode takes embedding or reference.', item);
      out.mode = mode;
      out.locs.mode = item;
      continue;
    }
    if (!special && (words.length !== 2 || !NUMBER.test(words[1]))) throw new LocError(`Expected “control value” (e.g. age 1), got “${item.text}”.`, item);
    if (!steering) throw new LocError(NO_STEERING, item);
    if (head === 'preset') {
      const name = item.text.slice(item.head.text.length).trim();
      const presets = Object.keys(spec.ui.presets);
      const key = Object.hasOwn(spec.ui.presets, name) ? name : presets.find((k) => k.toLowerCase() === name.toLowerCase());
      if (!key) throw new LocError(`Unknown preset '${name}'. Presets: ${presets.join(', ')}`, item);
      out.preset = key;
      out.locs.preset = item;
    } else if (head === 'accent') {
      if (words.length !== 3 || !NUMBER.test(words[2])) throw new LocError('accent takes an id and a strength: accent irish 1.2.', item);
      const id = words[1].toLowerCase();
      out.accents[id] = Number(words[2]);
      out.locs[`accent:${id}`] = item;
    } else {
      out.values[head] = Number(words[1]);
      out.locs[head] = item;
    }
  }
  return out;
}

/** `resolveValues` per item, so each error and warning lands on its item; the warnings once all resolve. */
function resolveItems(spec: ControlsSpec, items: Items, kind: 'design' | 'delivery', warn: Warn): Values {
  const values: Values = {};
  const said: [string[], Loc][] = [];
  for (const [id, v] of Object.entries(items.values)) {
    let r: ReturnType<typeof resolveValues>;
    try {
      r = resolveValues(spec, { [id]: v }, kind);
    } catch (e) {
      throw new LocError(e instanceof Error ? e.message : String(e), items.locs[id]);
    }
    Object.assign(values, r.values);
    said.push([r.warnings, items.locs[id]]);
  }
  for (const [messages, loc] of said) warn(messages, loc);
  return values;
}

/** Delivery over a whole line (or every line: @baseline): delivery controls, none of them span-only. */
function wholeLine(spec: ControlsSpec, items: Items, warn: Warn): Values {
  const values = resolveItems(spec, items, 'delivery', warn);
  for (const id of Object.keys(values)) {
    const c = spec.ui.controls.find((x) => x.id === id)!;
    if (c.span_only) throw new LocError(`${c.label} goes on words, not a whole line: [words]{${c.id} 4}.`, items.locs[id]);
  }
  return values;
}

interface VoiceDef {
  name: string;
  line: number;
  /** The voice's item list, else its name: where errors about the voice as a whole go. */
  loc: Loc;
  /** The library voice after `=`; null: step 1's voice (`fromSelected`), or a reference that failed. */
  base: Speaker | null;
  fromSelected: boolean;
  values: Values;
  pin: Pin;
  /** Set by `mode`, or embedding by a preset; unset: the base voice's mode for @mode. */
  mode?: SpeakerMode;
  /** The mode it speaks in on its base, once checked against it. */
  speaks?: SpeakerMode;
}

/** The mode a @voice speaks in on `base`; throws what `base` can't do in it. */
function voiceMode(spec: ControlsSpec, def: VoiceDef, base: Speaker, preferred: SpeakerMode): SpeakerMode {
  const mode = def.mode ?? modeFor(base, preferred);
  if (!base.speaker_modes.includes(mode)) throw new Error(`${base.label} has no ${mode} mode.`);
  for (const id of Object.keys(def.values)) {
    const c = spec.ui.controls.find((x) => x.id === id);
    if (c && !controlUsable(spec, c, mode)) throw new Error(`${c.label} needs mode embedding (mode reference keeps the recording: only delivery and mood apply).`);
  }
  const mix = mixAt(def.pin, spec.ui.accents);
  if (spec.ui.accents.some((a) => mix[a.id] && !controlUsable(spec, a, mode))) throw new Error('Accents need mode embedding.');
  return mode;
}

interface Speech {
  line: number;
  /** The line's text after its prefix. */
  loc: Loc;
  voice: VoiceDef | null;
  /** A library voice named at the start of the line. */
  speaker: Speaker | null;
  text: string;
  delivery: Values | null;
  placed: { start: number; end: number; values: Values }[];
}

type DirectiveLine = Extract<ScriptLine, { kind: 'directive' }>;

export function compileScript(text: string, env: ScriptEnv, random: () => number = Math.random): ScriptResult {
  const { spec, steering } = env;
  const errors: Issue[] = [];
  const warnings: Issue[] = [];
  /** Runs `run` for line `l`: what it throws is an error on its LocError's range, else on `loc`. */
  const at = (l: { line: number; loc: Loc }, loc: Loc, run: () => void) => {
    try {
      run();
    } catch (e) {
      errors.push({ line: l.line, ...fix(l, e instanceof LocError ? e.loc : loc), message: e instanceof Error ? e.message : String(e) });
    }
  };
  const warnAt =
    (l: { line: number; loc: Loc }): Warn =>
    (messages, loc) => {
      for (const message of messages) warnings.push({ line: l.line, ...fix(l, loc), message });
    };

  const tree = parseScript(text, speakerNames(env.speakers));
  // Pass 1: directives (global, anywhere).
  const set: { language?: string; mode?: SpeakerMode; seed?: number; laci?: LaciMode; decoder?: string; gap?: number; baseline?: Values } = {};
  const seen = new Map<string, DirectiveLine>();
  const voices = new Map<string, VoiceDef>();
  for (const l of tree) {
    if (l.kind !== 'directive') continue;
    // Errors about the argument go on it.
    at(l, l.arg, () => {
      const kw = l.name ?? l.at.text.slice(1);
      if (!l.name || !DIRECTIVES.includes(kw)) throw new LocError(`Unknown directive @${kw}. Directives: ${DIRECTIVES.map((d) => `@${d}`).join(', ')}.`, l.at);
      const arg = l.arg.text;
      const a = arg.toLowerCase();
      if (kw !== 'voice') {
        const first = seen.get(kw);
        if (first) throw new LocError(`@${kw} is set twice (first on line ${first.line}).`, l.at);
        seen.set(kw, l);
      }
      switch (kw) {
        case 'language': {
          const code = env.languages.find((c) => c.toLowerCase() === a || langLabel(c).toLowerCase() === a);
          if (!code) throw new Error(`No language “${arg}”. Languages: ${env.languages.map((c) => `${c} (${langLabel(c)})`).join(', ')}`);
          set.language = code;
          break;
        }
        case 'mode':
          if (!isMode(a)) throw new Error('@mode takes embedding or reference.');
          set.mode = a;
          break;
        case 'seed':
          if (!/^\d+$/.test(arg) || Number(arg) > SEED_MAX) throw new Error(`@seed takes a whole number from 0 to ${SEED_MAX}.`);
          set.seed = Number(arg);
          break;
        case 'laci': {
          const mode = LACI_MODES.find((x) => x === a);
          if (!mode) throw new Error('@laci takes off, stream or whole.');
          if (mode !== 'off' && !env.caps.laci_languages.length) throw new Error('This server has no LACI alignment heads.');
          set.laci = mode;
          break;
        }
        case 'decoder': {
          const served = env.caps.decoders ?? [];
          if (!served.some((d) => d.id === arg)) throw new Error(unknownDecoder(arg, served));
          set.decoder = arg;
          break;
        }
        case 'gap': {
          const [lo, hi] = GAP_RANGE;
          if (!NUMBER.test(arg) || Number(arg) < lo || Number(arg) > hi) throw new Error(`@gap takes seconds from ${lo} to ${hi}.`);
          set.gap = Number(arg);
          break;
        }
        case 'baseline':
          set.baseline = wholeLine(spec, readItems(l.items, spec, steering, false), warnAt(l));
          break;
        case 'voice': {
          const v = l.voice!;
          if (!v.valid) throw new Error('Write @voice Name = library voice {…}, or @voice Name {…} for the voice of step 1.');
          const name = v.name.text;
          if (name.length > VOICE_NAME_MAX) throw new LocError(`A voice name has at most ${VOICE_NAME_MAX} characters.`, v.name);
          const first = voices.get(name.toLowerCase());
          if (first) throw new LocError(`Voice “${name}” is defined twice (first on line ${first.line}).`, v.name);
          // Registered before its checks: lines naming a broken voice report only the voice's own error.
          const def: VoiceDef = { name, line: l.line, loc: fix(l, v.items ?? v.name), base: null, fromSelected: v.eq === null, values: {}, pin: homePin() };
          voices.set(name.toLowerCase(), def);
          if (v.ref) {
            const base = findSpeaker(env.speakers, v.ref.text);
            if (!base) throw new LocError(`No library voice “${v.ref.text}”. Voices: ${env.speakers.map((s) => s.label).join(', ')}.`, v.ref);
            def.base = base;
          }
          // Studio's order whatever the written one: preset, values, accents, mode.
          const items = readItems(v.items, spec, steering, true);
          if (items.preset) {
            if (items.mode === 'reference') throw new LocError('A preset sets mode embedding.', items.locs.preset);
            const p = presetToValues(spec.ui.presets[items.preset]);
            def.values = p.values;
            def.pin = pinForMix(p.accents, spec.ui.accents, def.pin.tab);
            def.mode = 'embedding';
          }
          if (Object.keys(items.values).length) {
            const values = resolveItems(spec, items, 'design', warnAt(l));
            for (const id of Object.keys(items.values)) {
              if (values[id]) def.values[id] = values[id];
              else delete def.values[id];
            }
          }
          if (Object.keys(items.accents).length) {
            const list = v.items!;
            let r: AccentResolution;
            try {
              r = resolveAccents(spec, items.accents);
            } catch (e) {
              const bad = Object.keys(items.accents).find((id) => !spec.ui.accents.some((x) => x.id === id));
              throw new LocError(e instanceof Error ? e.message : String(e), bad ? items.locs[`accent:${bad}`] : list);
            }
            warnAt(l)(r.warnings, list);
            def.pin = r.pin;
          }
          if (items.mode) def.mode = items.mode;
          break;
        }
      }
    });
  }

  const preferred = set.mode ?? 'embedding';
  // Each voice against its base: a library voice's now, step 1's when it has one (after the clone, else).
  for (const def of voices.values()) {
    const base = def.fromSelected ? env.selected : def.base;
    if (base && !errors.some((e) => e.line === def.line)) at(def, def.loc, () => (def.speaks = voiceMode(spec, def, base, preferred)));
  }
  const laci = set.laci ?? env.laci;
  const ctx = capsContext(env.caps, { decoder: set.decoder ?? null, laci });

  // Pass 2: speech lines (the parse resolved their names against every @voice, wherever it is defined).
  const speech: Speech[] = [];
  let anySpeech = false;
  for (const l of tree) {
    if (l.kind !== 'speech') continue;
    anySpeech = true;
    at(l, l.loc, () => {
      const warn = warnAt(l);
      // The parse's pick among voices, labels and ids; the same order resolves it here.
      const k = l.prefix?.name?.text.toLowerCase();
      const voice = k === undefined ? null : (voices.get(k) ?? null);
      const speaker = k === undefined || voice ? null : (env.speakers.find((s) => s.label.toLowerCase() === k) ?? env.speakers.find((s) => s.id.toLowerCase() === k) ?? null);
      if (l.stray) warn([`“${l.stray.text}:” is spoken: no voice is named ${l.stray.text}.`], l.stray);
      const delivery = l.prefix?.items ? wholeLine(spec, readItems(l.prefix.items, spec, steering, false), warn) : null;
      // Spans: `[words]{items}` → the words, painted; other brackets stay text.
      const rest = l.text.text;
      let said = '';
      let from = 0;
      const placed: Speech['placed'] = [];
      for (const sp of l.spans) {
        said += rest.slice(from, sp.from - l.text.from);
        const words = sp.words.text;
        if (!words.trim()) throw new LocError('A [ ]{…} span needs words.', sp);
        const values = resolveItems(spec, readItems(sp.items, spec, steering, false), 'delivery', warn);
        const start = said.length + words.length - words.trimStart().length;
        placed.push({ start, end: start + words.trim().length, values });
        said += words;
        from = sp.to - l.text.from;
      }
      said += rest.slice(from);
      const lead = said.length - said.trimStart().length;
      const spoken = said.trim();
      if (!spoken) throw new Error('Nothing to say on this line.');
      speech.push({ line: l.line, loc: l.text, voice, speaker, text: spoken, delivery, placed: placed.map((p) => ({ ...p, start: p.start - lead, end: p.end - lead })) });
    });
  }
  if (!anySpeech) errors.push({ line: 0, from: 0, to: 0, message: 'Nothing to say: write a line of text.' });
  const usesSelected = speech.some((s) => (s.voice ? s.voice.fromSelected : !s.speaker));

  let render: ScriptRender | null = null;
  if (!errors.length) {
    // One seed for every line, as Direct renders every first take.
    const seed = set.seed ?? env.seed ?? Math.floor(random() * 1e6);
    const language = set.language ?? env.language;
    const parts: ScriptPart[] = [];
    const noHeads = new Set<string>();
    let waiting = false;
    for (const s of speech) {
      at(s, s.loc, () => {
        const base = s.voice ? (s.voice.base ?? env.selected) : (s.speaker ?? env.selected);
        if (!base) {
          if (env.selectedWhy) throw new Error(`This line speaks with the voice of step 1: ${env.selectedWhy}.`);
          waiting = true;
          return;
        }
        // A voice with a base was checked against it above (`speaks`), or that check is an error already.
        const design = s.voice
          ? { baseId: base.id, mode: s.voice.speaks!, values: s.voice.values, pin: s.voice.pin }
          : { baseId: base.id, mode: modeFor(base, preferred), values: {}, pin: homePin() };
        // Direct's rule: a default voice speaks its own language.
        const lang = voiceServerLanguage(base, language);
        const pron = pronProblem(s.text, ctx.phonemeTokens);
        if (pron) throw new Error(pron);
        if (laci !== 'off' && !env.caps.laci_languages.includes(lang) && !noHeads.has(lang)) {
          noHeads.add(lang);
          warnAt(s)([`No alignment heads for ${langLabel(lang)}: its lines run without LACI.`], s.loc);
        }
        const spans = layerSpans(s.text.length, s.delivery, s.placed);
        const request = buildRequest(spec, ctx, design, s.text, seed, { extra: set.baseline, spans, language: lang });
        parts.push({ line: s.line, label: s.voice?.name ?? base.label, text: s.text, request: { ...request, stream: true, format: 'pcm' } });
      });
    }
    if (!errors.length && !waiting) render = { parts, gap: set.gap ?? DEFAULT_GAP, seed };
  }
  const byLine = (x: Issue, y: Issue) => x.line - y.line;
  return { errors: errors.sort(byLine), warnings: warnings.sort(byLine), usesSelected, render };
}

/** What the script's items may name on this server, as one line. */
export function vocabulary(spec: ControlsSpec, steering: boolean): string {
  if (!steering) return NO_STEERING;
  const voice = spec.ui.controls.filter((c) => !c.spannable).map((c) => c.id);
  const delivery = spec.ui.controls.filter((c) => c.spannable).map((c) => (c.span_only ? `${c.id} (words only)` : c.id));
  return `Voice: ${voice.join(', ')} · Delivery (lines and words): ${delivery.join(', ')} · Accents: ${spec.ui.accents.map((a) => a.id).join(', ')} · Presets: ${Object.keys(spec.ui.presets).join(', ')}`;
}

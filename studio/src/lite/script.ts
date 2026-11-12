/**
 * The lite page's Script input: a line-oriented text for what Studio's Cast, Direct and project settings do
 * (designed voices, a voice per line, whole-line and per-word delivery, the script baseline, the gap between lines,
 * language, seed, LACI, decoder, tag strength), compiled into the requests Studio's own composer (`buildRequest`)
 * sends for the same settings. Pure: no DOM, no signals; the page loads it on first use of Script.
 */
import type { ControlsSpec, SpeakerMode } from '../api';
import { unknownDecoder } from '../lib/decoders';
import { LACI_MODES, type LaciMode } from '../lib/laci';
import { langLabel } from '../lib/languages';
import { pronProblem } from '../lib/pron';
import { tagProblem } from '../lib/tags';
import { voiceServerLanguage } from '../lib/voices';
import { homePin, mixAt, pinForMix, type Pin } from '../state/atlas';
import { controlUsable, presetToValues, type Values } from '../state/engine';
import { DEFAULT_GAP, GAP_RANGE } from '../state/lines';
import { DEFAULT_SETTINGS } from '../state/projectSchema';
import { buildRequest, capsContext } from '../state/requests';
import { findSpeaker, layerSpans, modeFor, resolveAccents, resolveValues } from '../state/resolve';
import type { Capabilities, GenerateRequest, Speaker } from './api';

export interface Issue {
  /** 1-based line of the script; 0: the whole script. */
  line: number;
  message: string;
}

export interface ScriptEnv {
  /** `GET /v2/controls`, else `NO_CONTROLS`. */
  spec: ControlsSpec;
  /** False: `/v2/controls` answered 404. */
  steering: boolean;
  /** `GET /v2/speakers`, every mode (not the lite select's embedding-only list). */
  speakers: Speaker[];
  caps: Pick<Capabilities, 'phoneme_tokens' | 'tag_cfg' | 'decoders' | 'laci_languages'>;
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
};

const DIRECTIVES = ['language', 'mode', 'seed', 'laci', 'decoder', 'tags', 'gap', 'baseline', 'voice'];
const NUMBER = /^[+-]?(\d+(\.\d*)?|\.\d+)$/;
const VOICE = /^([^:{}\[\]=@#,\\]+?)\s*(?:=\s*([^{}]+?))?\s*(?:\{([^{}]*)\})?$/;
/** After a voice name at the start of a line: optional `{delivery}`, then the colon. */
const PREFIX = /^\s*(?:\{([^{}]*)\})?\s*:/;
const SPAN = /\[([^\[\]]*)\]\{([^{}]*)\}/g;
const NO_STEERING = 'This server has no steering bundle (GET /v2/controls answers 404): controls, presets and accents are off.';
const SEED_MAX = 4294967295;

const isMode = (s: string): s is SpeakerMode => s === 'embedding' || s === 'reference';

interface Items {
  values: Record<string, number>;
  accents: Record<string, number>;
  /** The preset's key in `spec.ui.presets`. */
  preset?: string;
  mode?: SpeakerMode;
}

/** `control value, …` (on a @voice also `preset Name`, `accent id strength`, `mode m`); a later repeat wins. */
function parseItems(body: string, spec: ControlsSpec, steering: boolean, onVoice: boolean): Items {
  const out: Items = { values: {}, accents: {} };
  for (const raw of body.split(',')) {
    const item = raw.trim();
    if (!item) continue;
    const words = item.split(/\s+/);
    const head = words[0].toLowerCase();
    const special = head === 'preset' || head === 'accent' || head === 'mode';
    if (special && !onVoice) throw new Error('preset, accent and mode go on a @voice line.');
    if (head === 'mode') {
      const mode = words[1]?.toLowerCase() ?? '';
      if (words.length !== 2 || !isMode(mode)) throw new Error('mode takes embedding or reference.');
      out.mode = mode;
      continue;
    }
    if (!special && (words.length !== 2 || !NUMBER.test(words[1]))) throw new Error(`Expected “control value” (e.g. age 1), got “${item}”.`);
    if (!steering) throw new Error(NO_STEERING);
    if (head === 'preset') {
      const name = item.slice(words[0].length).trim();
      const presets = Object.keys(spec.ui.presets);
      const key = Object.hasOwn(spec.ui.presets, name) ? name : presets.find((k) => k.toLowerCase() === name.toLowerCase());
      if (!key) throw new Error(`Unknown preset '${name}'. Presets: ${presets.join(', ')}`);
      out.preset = key;
    } else if (head === 'accent') {
      if (words.length !== 3 || !NUMBER.test(words[2])) throw new Error('accent takes an id and a strength: accent irish 1.2.');
      out.accents[words[1].toLowerCase()] = Number(words[2]);
    } else out.values[head] = Number(words[1]);
  }
  return out;
}

/** Delivery over a whole line (or every line: @baseline): delivery controls, none of them span-only. */
function wholeLine(spec: ControlsSpec, items: Items, warn: (messages: string[]) => void): Values {
  const r = resolveValues(spec, items.values, 'delivery');
  warn(r.warnings);
  for (const id of Object.keys(r.values)) {
    const c = spec.ui.controls.find((x) => x.id === id)!;
    if (c.span_only) throw new Error(`${c.label} goes on words, not a whole line: [words]{${c.id} 4}.`);
  }
  return r.values;
}

interface VoiceDef {
  name: string;
  line: number;
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
  voice: VoiceDef | null;
  /** A library voice named at the start of the line. */
  speaker: Speaker | null;
  text: string;
  delivery: Values | null;
  placed: { start: number; end: number; values: Values }[];
}

export function compileScript(text: string, env: ScriptEnv, random: () => number = Math.random): ScriptResult {
  const { spec, steering } = env;
  const errors: Issue[] = [];
  const warnings: Issue[] = [];
  const at = (line: number, run: () => void) => {
    try {
      run();
    } catch (e) {
      errors.push({ line, message: e instanceof Error ? e.message : String(e) });
    }
  };
  const warnOn = (line: number) => (messages: string[]) => {
    for (const message of messages) warnings.push({ line, message });
  };

  // Pass 1: directives (global, anywhere) and the speech lines.
  const set: { language?: string; mode?: SpeakerMode; seed?: number; laci?: LaciMode; decoder?: string; tags?: number; gap?: number; baseline?: Values } = {};
  const seen = new Map<string, number>();
  const voices = new Map<string, VoiceDef>();
  const lines: { line: number; text: string; escaped: boolean }[] = [];
  text.split(/\r?\n/).forEach((raw, i) => {
    const line = i + 1;
    const s = raw.trim();
    if (!s || s.startsWith('#')) return;
    if (s.startsWith('\\')) return void lines.push({ line, text: s.slice(1), escaped: true });
    if (!s.startsWith('@')) return void lines.push({ line, text: s, escaped: false });
    at(line, () => {
      const m = /^@([a-z]+)\b\s*(.*)$/i.exec(s);
      const kw = m?.[1].toLowerCase() ?? s.slice(1).split(/\s/)[0];
      if (!m || !DIRECTIVES.includes(kw)) throw new Error(`Unknown directive @${kw}. Directives: ${DIRECTIVES.map((d) => `@${d}`).join(', ')}.`);
      const arg = m[2];
      const a = arg.toLowerCase();
      if (kw !== 'voice') {
        const first = seen.get(kw);
        if (first) throw new Error(`@${kw} is set twice (first on line ${first}).`);
        seen.set(kw, line);
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
        case 'tags':
          if (!NUMBER.test(arg) || Number(arg) < 1) throw new Error('@tags takes a number of 1 or more (1 = off).');
          set.tags = Number(arg);
          break;
        case 'gap': {
          const [lo, hi] = GAP_RANGE;
          if (!NUMBER.test(arg) || Number(arg) < lo || Number(arg) > hi) throw new Error(`@gap takes seconds from ${lo} to ${hi}.`);
          set.gap = Number(arg);
          break;
        }
        case 'baseline':
          set.baseline = wholeLine(spec, parseItems(arg, spec, steering, false), warnOn(line));
          break;
        case 'voice': {
          const v = VOICE.exec(arg);
          if (!v) throw new Error('Write @voice Name = library voice {…}, or @voice Name {…} for the voice of step 1.');
          const name = v[1].trim();
          if (name.length > 40) throw new Error('A voice name has at most 40 characters.');
          const first = voices.get(name.toLowerCase());
          if (first) throw new Error(`Voice “${name}” is defined twice (first on line ${first.line}).`);
          // Registered before its checks: lines naming a broken voice report only the voice's own error.
          const def: VoiceDef = { name, line, base: null, fromSelected: v[2] === undefined, values: {}, pin: homePin() };
          voices.set(name.toLowerCase(), def);
          if (v[2] !== undefined) {
            const ref = v[2].trim();
            const base = findSpeaker(env.speakers, ref);
            if (!base) throw new Error(`No library voice “${ref}”. Voices: ${env.speakers.map((s) => s.label).join(', ')}.`);
            def.base = base;
          }
          // Studio's order whatever the written one: preset, values, accents, mode.
          const items = parseItems(v[3] ?? '', spec, steering, true);
          if (items.preset) {
            if (items.mode === 'reference') throw new Error('A preset sets mode embedding.');
            const p = presetToValues(spec.ui.presets[items.preset]);
            def.values = p.values;
            def.pin = pinForMix(p.accents, spec.ui.accents, def.pin.tab);
            def.mode = 'embedding';
          }
          if (Object.keys(items.values).length) {
            const r = resolveValues(spec, items.values, 'design');
            warnOn(line)(r.warnings);
            for (const id of Object.keys(items.values)) {
              if (r.values[id]) def.values[id] = r.values[id];
              else delete def.values[id];
            }
          }
          if (Object.keys(items.accents).length) {
            const r = resolveAccents(spec, items.accents);
            warnOn(line)(r.warnings);
            def.pin = r.pin;
          }
          if (items.mode) def.mode = items.mode;
          break;
        }
      }
    });
  });

  const preferred = set.mode ?? 'embedding';
  // Each voice against its base: a library voice's now, step 1's when it has one (after the clone, else).
  for (const def of voices.values()) {
    const base = def.fromSelected ? env.selected : def.base;
    if (base && !errors.some((e) => e.line === def.line)) at(def.line, () => (def.speaks = voiceMode(spec, def, base, preferred)));
  }
  const laci = set.laci ?? env.laci;
  const ctx = capsContext(env.caps, { tagStrength: set.tags ?? DEFAULT_SETTINGS.tagStrength, decoder: set.decoder ?? null, laci });
  if (set.tags !== undefined && !ctx.tags) warnOn(seen.get('tags')!)(['This checkpoint doesn’t read nonverbal tags: @tags does nothing.']);

  // Pass 2: speech lines, after every @voice is known (a voice may be defined after its first line).
  const candidates = [
    ...[...voices.values()].map((voice) => ({ name: voice.name, voice, speaker: null })),
    ...env.speakers.map((speaker) => ({ name: speaker.label, voice: null, speaker })),
    ...env.speakers.map((speaker) => ({ name: speaker.id, voice: null, speaker })),
  ].sort((x, y) => y.name.length - x.name.length);
  const speech: Speech[] = [];
  for (const raw of lines) {
    at(raw.line, () => {
      const warn = warnOn(raw.line);
      let rest = raw.text;
      let voice: VoiceDef | null = null;
      let speaker: Speaker | null = null;
      let braces: string | undefined;
      if (!raw.escaped) {
        const hit = candidates.find((c) => rest.slice(0, c.name.length).toLowerCase() === c.name.toLowerCase() && PREFIX.test(rest.slice(c.name.length)));
        if (hit) {
          const m = PREFIX.exec(rest.slice(hit.name.length))!;
          ({ voice, speaker } = hit);
          braces = m[1];
          rest = rest.slice(hit.name.length + m[0].length);
        } else {
          const m = /^\{([^{}]*)\}\s*:/.exec(rest);
          if (m) {
            braces = m[1];
            rest = rest.slice(m[0].length);
          } else {
            const w = /^([\p{L}\p{N}_'’-]+)\s*:\s/u.exec(rest);
            if (w) warn([`“${w[1]}:” is spoken: no voice is named ${w[1]}.`]);
          }
        }
      }
      const delivery = braces === undefined ? null : wholeLine(spec, parseItems(braces, spec, steering, false), warn);
      // Spans: `[words]{items}` → the words, painted; other brackets stay text.
      let said = '';
      let from = 0;
      const placed: Speech['placed'] = [];
      for (const m of rest.matchAll(SPAN)) {
        said += rest.slice(from, m.index);
        const words = m[1];
        if (!words.trim()) throw new Error('A [ ]{…} span needs words.');
        const r = resolveValues(spec, parseItems(m[2], spec, steering, false).values, 'delivery');
        warn(r.warnings);
        const start = said.length + words.length - words.trimStart().length;
        placed.push({ start, end: start + words.trim().length, values: r.values });
        said += words;
        from = m.index + m[0].length;
      }
      said += rest.slice(from);
      const lead = said.length - said.trimStart().length;
      const spoken = said.trim();
      if (!spoken) throw new Error('Nothing to say on this line.');
      speech.push({ line: raw.line, voice, speaker, text: spoken, delivery, placed: placed.map((p) => ({ ...p, start: p.start - lead, end: p.end - lead })) });
    });
  }
  if (!lines.length) errors.push({ line: 0, message: 'Nothing to say: write a line of text.' });
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
      at(s.line, () => {
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
        const tags = tagProblem(s.text);
        if (tags) warnOn(s.line)([tags]);
        if (laci !== 'off' && !env.caps.laci_languages.includes(lang) && !noHeads.has(lang)) {
          noHeads.add(lang);
          warnOn(s.line)([`No alignment heads for ${langLabel(lang)}: its lines run without LACI.`]);
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

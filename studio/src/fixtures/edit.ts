/**
 * Client-side stand-in for the speech-editing endpoints (`/v2/edit/transcribe`, `/v2/edit/align`,
 * `/v2/edit`), following the contract's shapes and 400s. Clips the offline synthesizer made are remembered with
 * their exact word timings (a perfect aligner); any other clip (uploads, recordings, server audio) is aligned
 * from its energy envelope. Fills are synthesized with the clip's voice when it is known, spliced in with short
 * crossfades, verified with a simulated ASR that sometimes mishears a word.
 */
import { ApiError } from '../lib/http';
import { type Alignment, type ControlsSpec, type Dose, type EditCaps, type EditRequest, type EditResponse, type EditTake, type EditWord } from '../api';
import { resample } from '../audio/mix';
import { contextWindows, diffOpcodes, normalizeTokens, textWords, tokenOwners, wordDiff } from '../edit/regions';
import { base64ToBytes, bytesToBase64 } from '../lib/files';
import { parseWav, wavBytes } from '../lib/wav';
import { ipaGloss } from '../lib/pron';
import { EDIT_LANGUAGES } from '../lib/languages';
import { hash32, rng } from '../lib/math';
import { syntheticEmbedding } from './embeddings';
import { synthesizeWords, type SynthWord } from './synth';

const SR = 44100;

export interface Known {
  text: string;
  words: SynthWord[];
  embedding: Float32Array;
  controls?: Dose;
}

/** Clips the fixture made, by duration in ms (a session never makes two of the same length to the ms). */
const known = new Map<number, Known>();
/** Alignments handed out, by duration in ms: /v2/edit plans its fills from them. */
const aligned = new Map<number, EditWord[]>();

const keyOf = (seconds: number) => Math.round(seconds * 1000);

function lookup<T>(m: Map<number, T>, seconds: number): T | undefined {
  const k = keyOf(seconds);
  return m.get(k) ?? m.get(k - 1) ?? m.get(k + 1);
}

// A pronunciation group as sent; the recognizer writes some spelling of it ("shivon").
const SENT_GROUP = /\ue000([^\ue000\ue001]*)\ue001/g;
const spoken = (t: string) => t.replace(SENT_GROUP, (_, ipa: string) => ipaGloss(ipa));

export function remember(pcm: Float32Array, sr: number, info: Known) {
  // A recognizer writes a pronunciation group as some spelling of it ("shivon"), never the IPA.
  const heard = (t: string) => spoken(t).replace(/[\ue000\ue001]/g, '');
  known.set(keyOf(pcm.length / sr), { ...info, text: heard(info.text), words: info.words.map((w) => ({ ...w, word: heard(w.word) })).filter((w) => w.word) });
}

function decode(b64: string): { pcm: Float32Array; sr: number; duration: number } {
  let wav: { pcm: Float32Array; sr: number };
  try {
    wav = parseWav(base64ToBytes(b64));
  } catch {
    throw new ApiError(400, 'audio_base64 could not be decoded (the fixture reads WAV only).');
  }
  return { ...wav, duration: wav.pcm.length / wav.sr };
}

function wait(ms: number, signal?: AbortSignal) {
  const { promise, resolve, reject } = Promise.withResolvers<void>();
  if (signal?.aborted) reject(new DOMException('Aborted', 'AbortError'));
  const t = setTimeout(resolve, ms);
  signal?.addEventListener('abort', () => {
    clearTimeout(t);
    reject(new DOMException('Aborted', 'AbortError'));
  });
  return promise;
}

const asWords = (ws: SynthWord[], conf: (i: number) => number): EditWord[] =>
  ws.map((w, i) => ({ word: w.word, start: round3(w.start), end: round3(w.end), confidence: conf(i), tokens: normalizeTokens(w.word) }));

const round3 = (x: number) => Math.round(x * 1000) / 1000;

/** Spread the words of `text` over the speech in `pcm`, snapping boundaries to energy dips. */
function energyAlign(pcm: Float32Array, sr: number, text: string): EditWord[] {
  const hop = Math.round(sr * 0.01);
  const n = Math.floor(pcm.length / hop);
  const rms = new Float32Array(n);
  let peak = 1e-6;
  for (let f = 0; f < n; f++) {
    let s = 0;
    for (let i = f * hop; i < (f + 1) * hop; i++) s += pcm[i] * pcm[i];
    rms[f] = Math.sqrt(s / hop);
    peak = Math.max(peak, rms[f]);
  }
  const thr = peak * 0.12;
  let first = 0;
  while (first < n - 1 && rms[first] < thr) first++;
  let last = n - 1;
  while (last > first && rms[last] < thr) last--;
  const raw = textWords(text);
  if (!raw.length) return [];
  const weights = raw.map((w) => normalizeTokens(w).join('').length + 0.6);
  const total = weights.reduce((a, b) => a + b, 0);
  const bounds = [first];
  let acc = 0;
  for (let i = 0; i < raw.length - 1; i++) {
    acc += weights[i];
    const guess = Math.round(first + ((last - first) * acc) / total);
    let best = guess;
    for (let f = Math.max(first + 1, guess - 8); f <= Math.min(last - 1, guess + 8); f++) if (rms[f] < rms[best]) best = f;
    bounds.push(Math.max(best, bounds[bounds.length - 1] + 3));
  }
  bounds.push(last + 1);
  return raw.map((word, i) => {
    let s = bounds[i];
    let e = bounds[i + 1];
    while (s < e - 2 && rms[s] < thr * 0.6) s++;
    while (e > s + 2 && rms[e - 1] < thr * 0.6) e--;
    let voiced = 0;
    for (let f = s; f < e; f++) voiced += rms[f] > thr ? 1 : 0;
    return { word, start: round3(s * 0.01), end: round3(e * 0.01), confidence: Math.round((0.55 + 0.4 * (voiced / Math.max(1, e - s))) * 100) / 100, tokens: normalizeTokens(word) };
  });
}

const STOCK =
  'so this is a quick recording to try the editor and see how it copes when a few words change in the middle of a sentence ' +
  'then I keep talking for a little while longer so there is enough audio to work with and maybe a second idea at the end';

const MISHEAR: Record<string, string> = { a: 'the', the: 'a', to: 'too', for: 'four', I: 'eye', when: 'then', how: 'now', read: 'red', line: 'lime', short: 'sort', sound: 'found' };

const languageOk = (language: string) => {
  if (!EDIT_LANGUAGES.includes(language)) throw new ApiError(400, `Speech editing does not serve language '${language}' (supported: ${EDIT_LANGUAGES.join(', ')}).`);
};

/** A server without speech recognition (`edit_transcribe: false`) refuses what needs it, before any work. */
const noAsr = (fix: string) => new ApiError(503, `Speech recognition is not available on this server: ${fix}`);

/** The server's speech recognition (transcription, verify): it loads on first use. Alignment never needs it. */
let loaded = false;
async function asr(signal?: AbortSignal) {
  if (loaded) return;
  await wait(2500, signal);
  loaded = true;
}

export async function transcribe(caps: EditCaps, b64: string, language = 'en_us', signal?: AbortSignal): Promise<Alignment> {
  if (!caps.edit_transcribe) throw noAsr('send the transcript to /v2/edit/align instead');
  languageOk(language);
  await asr();
  const { pcm, sr, duration } = decode(b64);
  if (duration > 150) throw new ApiError(400, `Alignment supports clips up to 150 s (this one is ${duration.toFixed(1)} s).`);
  await wait(500 + duration * 40, signal);
  const k = lookup(known, duration);
  let words: EditWord[];
  if (k) {
    // Our own clip: exact timings, with one plausible mishearing at low confidence to correct.
    const r = rng(hash32(k.text));
    const wrong = Math.floor(r() * k.words.length);
    words = asWords(k.words, (i) => (i === wrong ? 0.34 : 0.88 + r() * 0.11)).map((w, i) => {
      if (i !== wrong) return w;
      const bare = w.word.replace(/[^\p{L}']/gu, '');
      const alt = MISHEAR[bare] ?? MISHEAR[bare.toLowerCase()] ?? bare.slice(0, Math.max(1, bare.length - 1));
      return { ...w, word: w.word.replace(bare, alt), tokens: normalizeTokens(alt) };
    });
  } else {
    const count = Math.max(2, Math.round(duration * 2.6));
    const stock = STOCK.split(' ');
    words = energyAlign(pcm, sr, Array.from({ length: count }, (_, i) => stock[i % stock.length]).join(' '));
  }
  const text = words.map((w) => w.word).join(' ');
  aligned.set(keyOf(duration), words);
  return { text, duration: round3(duration), words };
}

export async function align(b64: string, text: string, language = 'en_us', signal?: AbortSignal): Promise<Alignment> {
  languageOk(language);
  const { pcm, sr, duration } = decode(b64);
  if (duration > 150) throw new ApiError(400, `Alignment supports clips up to 150 s (this one is ${duration.toFixed(1)} s).`);
  if (!normalizeTokens(text).length) throw new ApiError(400, 'text has no words to align.');
  await wait(300 + duration * 25, signal);
  const k = lookup(known, duration);
  const raw = textWords(text.trim());
  let words: EditWord[];
  if (k && k.words.length === raw.length) {
    words = asWords(k.words, () => 0.95).map((w, i) => ({ ...w, word: raw[i], tokens: normalizeTokens(raw[i]) }));
  } else words = energyAlign(pcm, sr, text.trim());
  aligned.set(keyOf(duration), words);
  return { text: text.trim(), duration: round3(duration), words };
}

function validate(spec: ControlsSpec, caps: EditCaps, req: EditRequest, duration: number) {
  if (req.original_words) {
    if (!req.original_text) throw new ApiError(400, 'original_words requires original_text.');
    const spelled = req.original_words.flatMap((w) => w.tokens ?? normalizeTokens(w.word));
    if (spelled.join(' ') !== normalizeTokens(req.original_text).join(' ')) throw new ApiError(400, 'original_words do not spell original_text.');
    req.original_words.forEach((w, i) => {
      if (!(Number.isFinite(w.start) && Number.isFinite(w.end) && w.start < w.end)) throw new ApiError(400, `original_words[${i}] needs start < end.`);
      if (w.start < -1e-3 || w.end > duration + 1e-3) throw new ApiError(400, `original_words[${i}] is outside the clip.`);
      if (i && w.start < req.original_words![i - 1].end - 1e-6) throw new ApiError(400, `original_words[${i}] overlaps the word before it.`);
    });
  }
  if (!caps.speech_editing) throw new ApiError(400, 'This checkpoint cannot edit speech (no span-end token).');
  if (!req.target_text?.trim()) throw new ApiError(400, 'target_text is empty.');
  const spans = req.spans;
  if (!spans?.length) throw new ApiError(400, 'The fixture needs explicit spans.');
  if (spans.length > caps.edit_max_spans) throw new ApiError(400, `At most ${caps.edit_max_spans} spans per edit (got ${spans.length}).`);
  if (duration > caps.edit_max_seconds) throw new ApiError(400, `Edits support clips up to ${caps.edit_max_seconds} s (this one is ${duration.toFixed(1)} s).`);
  spans.forEach(([s, e], i) => {
    if (!(s >= 0 && e > s && e <= duration + 1e-3)) throw new ApiError(400, `span ${i} [${s}, ${e}] is outside the audio (0..${duration.toFixed(2)} s).`);
    if (i && s < spans[i - 1][1]) throw new ApiError(400, 'spans must be ascending and non-overlapping.');
  });
  if (req.region_controls && req.region_controls.length !== spans.length) throw new ApiError(400, `region_controls has ${req.region_controls.length} entries for ${spans.length} spans.`);
  const n = req.num_takes ?? 1;
  if (n < 1 || n > 4) throw new ApiError(400, 'num_takes must be 1-4.');
  for (const d of [req.controls, ...(req.region_controls ?? [])]) {
    for (const id of Object.keys(d ?? {})) {
      const dir = spec.directions[id];
      if (!dir) throw new ApiError(400, `Unknown control direction '${id}'.`);
      if (dir.space === 'raw' && req.speaker_embedding === false) throw new ApiError(400, `Control ${id} moves the speaker embedding and needs speaker_embedding.`);
    }
  }
}

function sumDoses(...ds: (Dose | null | undefined)[]): Dose {
  const out: Dose = {};
  for (const d of ds) for (const [k, v] of Object.entries(d ?? {})) out[k] = (out[k] ?? 0) + v;
  return out;
}

/** Raw words of the target that fill span [s, e]: between the nearest unchanged words on either side. */
function fillText(words: EditWord[], target: string, s: number, e: number): string {
  const tokens = words.flatMap((w) => w.tokens.map(() => w));
  const { equalMap } = diffOpcodes(words.flatMap((w) => w.tokens), normalizeTokens(target));
  const owners = tokenOwners(target);
  let jL = -1;
  let jR = owners.length;
  tokens.forEach((w, i) => {
    const mid = (w.start + w.end) / 2;
    if (mid < s && equalMap[i] >= 0) jL = equalMap[i];
    if (mid > e && equalMap[i] >= 0 && jR === owners.length) jR = equalMap[i];
  });
  const picked: string[] = [];
  let lastOwner = -1;
  for (let j = jL + 1; j < jR; j++) {
    if (owners[j][0] === lastOwner) continue;
    lastOwner = owners[j][0];
    picked.push(target.slice(owners[j][0], owners[j][1]));
  }
  return picked.join(' ');
}

function crossfadeConcat(parts: Float32Array[], fade: number): Float32Array {
  let out: Float32Array = new Float32Array(0);
  for (const p of parts) {
    if (!out.length) {
      out = p.slice();
      continue;
    }
    const f = Math.min(fade, out.length, p.length);
    const next = new Float32Array(out.length + p.length - f);
    next.set(out.subarray(0, out.length - f));
    for (let i = 0; i < f; i++) {
      const a = i / f;
      next[out.length - f + i] = out[out.length - f + i] * (1 - a) + p[i] * a;
    }
    next.set(p.subarray(f), out.length);
    out = next;
  }
  return out;
}

/** verify.groups: every group of the target in order, with what was heard where a fill said it. */
function groupsHeard(target: string, heard: { ipa: string; heard: string | null }[]) {
  let k = 0;
  return [...target.matchAll(SENT_GROUP)].map((m) => (heard[k]?.ipa === m[1] ? heard[k++] : { ipa: m[1], heard: null }));
}

export async function edit(spec: ControlsSpec, caps: EditCaps, req: EditRequest, signal?: AbortSignal): Promise<EditResponse> {
  if (req.verify && !caps.edit_transcribe) throw noAsr('set verify to false');
  languageOk(req.language ?? 'en_us');
  if (req.verify) await asr(signal);
  const src = decode(req.audio_base64);
  validate(spec, caps, req, src.duration);
  const fps = caps.edit_frame_rate;
  const spans = (req.spans ?? []).map(([s, e]) => [Math.floor(s * fps) / fps, Math.min(src.duration, Math.ceil(e * fps) / fps)] as [number, number]);
  // Corrected timings win over the alignment, as on the server.
  const words: EditWord[] = req.original_words
    ? req.original_words.map((w) => ({ word: w.word, start: w.start, end: w.end, confidence: 1, tokens: w.tokens ?? normalizeTokens(w.word) }))
    : (lookup(aligned, src.duration) ?? (await align(req.audio_base64, req.original_text || req.target_text)).words);
  // Long clips are edited on word-snapped windows around the regions (what the model hears), merged when they touch.
  const contexts: [number, number][] =
    src.duration > (req.max_context_seconds ?? 16)
      ? contextWindows(words, spans, req.context_seconds ?? 4, src.duration).map(([a, b]) => [round3(a), round3(b)])
      : [];
  const k = lookup(known, src.duration);
  const embedding = k?.embedding ?? syntheticEmbedding(spec, `edit-${keyOf(src.duration)}`, '');
  const pcm = resample(src.pcm, src.sr, SR);
  const n = req.num_takes ?? 1;
  const base = req.seed ?? Math.floor(Math.random() * 100000);
  await wait(700 + 320 * n + spans.length * 150, signal);
  const fills = spans.map(([s, e]) => fillText(words, req.target_text, s, e));
  // Expected fill length: the region's seconds × target / original length of its words (as the server does).
  // A pronunciation group counts its IPA symbols.
  const letters = (t: string) => normalizeTokens(t).join('').replace(/[\ue000\ue001]/g, '').length;
  const expected = spans.map(([s, e], i) => {
    const orig = words.filter((w) => (w.start + w.end) / 2 >= s && (w.start + w.end) / 2 < e).map((w) => w.word).join(' ');
    return round3((e - s) * (letters(fills[i]) / Math.max(1, letters(orig) || letters(fills[i]))));
  });
  const targetTokens = normalizeTokens(req.target_text);
  const takes: EditTake[] = [];
  for (let t = 0; t < n; t++) {
    const seed = base + t;
    const r = rng(hash32(`take${seed}`));
    const parts: Float32Array[] = [];
    const edited: [number, number][] = [];
    const fillSeconds: number[] = [];
    const stopped: boolean[] = [];
    const outWords: EditWord[] = [];
    const heardGroups: { ipa: string; at: number }[] = [];
    let cursor = 0;
    let outLen = 0;
    const fade = Math.round(SR * 0.012);
    spans.forEach(([s, e], i) => {
      const a = Math.round(s * SR);
      const b = Math.round(e * SR);
      const keep = pcm.subarray(cursor, a);
      const shift = outLen / SR - cursor / SR;
      for (const w of words) {
        const mid = (w.start + w.end) / 2;
        if (mid >= cursor / SR && mid < s) outWords.push({ ...w, start: round3(Math.max(cursor / SR, w.start) + shift), end: round3(Math.min(s, w.end) + shift) });
      }
      parts.push(keep);
      outLen += keep.length;
      const text = spoken(fills[i]);
      let fill: Float32Array;
      let fw: SynthWord[] = [];
      const runaway = seed % 13 === 7 && i === 0;
      // seed % 13 == 5: the model skips the new words (a near-empty fill), as seen live on short replacements.
      const dropped = seed % 13 === 5 && i === 0 && !!text;
      const fillGroups = [...fills[i].matchAll(SENT_GROUP)].map((m) => m[1]);
      if (dropped) fill = new Float32Array(Math.round(SR * 0.07));
      else if (text) {
        const syn = synthesizeWords(spec, { text: runaway ? `${text} ${text} ${text}` : text, embedding, controls: sumDoses(k?.controls, req.controls, req.region_controls?.[i]), seed });
        // Trim the synthesizer's lead-in / tail so the fill sits tight between the kept audio.
        const lead = Math.max(0, Math.round(((syn.words[0]?.start ?? 0.1) - 0.05) * SR));
        const tail = Math.min(syn.pcm.length, Math.round(((syn.words[syn.words.length - 1]?.end ?? 0) + 0.06) * SR));
        fill = syn.pcm.slice(lead, Math.max(lead + 1, tail));
        fw = syn.words.map((w) => ({ ...w, start: w.start - lead / SR, end: w.end - lead / SR }));
      } else fill = new Float32Array(Math.round(SR * (0.08 + 0.06 * r())));
      const at = (outLen - fade) / SR;
      for (const w of fw) outWords.push({ word: w.word, start: round3(at + w.start), end: round3(at + w.end), confidence: 0.9, tokens: normalizeTokens(w.word) });
      // Where each group was said: the fill word spelling its gloss (recognized below, mishearings included).
      let from = outWords.length - fw.length;
      for (const ipa of fillGroups) {
        const gloss = ipaGloss(ipa);
        const idx = outWords.findIndex((w, j) => j >= from && w.word.replace(/[^\p{L}']/gu, '') === gloss);
        heardGroups.push({ ipa, at: dropped ? -1 : idx });
        if (idx >= 0) from = idx + 1;
      }
      parts.push(fill);
      outLen += fill.length - fade;
      edited.push([round3(Math.max(0, at)), round3(at + fill.length / SR)]);
      fillSeconds.push(round3(fill.length / SR));
      stopped.push(!runaway);
      cursor = b;
    });
    const shift = outLen / SR - cursor / SR;
    for (const w of words) {
      const mid = (w.start + w.end) / 2;
      if (mid >= cursor / SR) outWords.push({ ...w, start: round3(Math.max(cursor / SR, w.start) + shift), end: round3(w.end + shift) });
    }
    parts.push(pcm.subarray(cursor));
    const out = crossfadeConcat(parts, fade);
    // Simulated ASR: every third seed mishears one word of a fill.
    const hyp = outWords.map((w) => w.word);
    const fillWordIdx = outWords.map((w, i) => (edited.some(([s, e]) => w.start >= s - 1e-3 && w.end <= e + 1e-3) ? i : -1)).filter((i) => i >= 0);
    if (fillWordIdx.length && hash32(`asr${seed}`) % 3 === 0) {
      const i = fillWordIdx[Math.floor(r() * fillWordIdx.length)];
      hyp[i] = MISHEAR[hyp[i].replace(/[^\p{L}']/gu, '')] ?? 'uh';
    } else if (fillWordIdx.length && seed % 13 === 3) {
      // A spelling slip ("reely" for "really"): still heard.
      const i = fillWordIdx[fillWordIdx.length - 1];
      hyp[i] = hyp[i].replace(/[aeiou](?=[^aeiou]*$)/i, (v) => (v.toLowerCase() === 'e' ? 'a' : 'e'));
    }
    const asr = hyp.join(' ').replace(/[^\p{L}\p{N}'\s]/gu, '').toLowerCase();
    const hypTokens = normalizeTokens(asr);
    const wrong = wordDiff(targetTokens, hypTokens).filter(Boolean).length;
    const wer = Math.round((Math.max(wrong, Math.abs(targetTokens.length - hypTokens.length)) / Math.max(1, targetTokens.length)) * 1000) / 1000;
    remember(out, SR, { text: spoken(req.target_text.trim()), words: outWords, embedding, controls: k?.controls });
    aligned.set(keyOf(out.length / SR), outWords);
    takes.push({
      seed,
      audio_base64: bytesToBase64(wavBytes(out, SR).buffer),
      duration: round3(out.length / SR),
      edited_regions: edited,
      fill_seconds: fillSeconds,
      stopped,
      verify: req.verify
        ? { asr_text: asr, wer, cer: Math.round(wer * 600) / 1000, ...(req.target_text.includes('\ue000') ? { groups: groupsHeard(req.target_text, heardGroups.map((h) => ({ ipa: h.ipa, heard: h.at >= 0 ? hyp[h.at].replace(/[^\p{L}\p{N}']/gu, '').toLowerCase() || null : null }))) } : {}) }
        : null,
      words: req.verify ? outWords : null,
    });
  }
  return { sample_rate: SR, original_text: req.original_text ?? words.map((w) => w.word).join(' '), target_text: req.target_text.trim(), words, spans, contexts: contexts.length ? contexts : null, expected_seconds: req.original_text ? expected : null, takes };
}

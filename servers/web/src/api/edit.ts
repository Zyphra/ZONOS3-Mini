// POST /v2/edit/transcribe, /v2/edit/align and /v2/edit of the in-browser server: zonos3.tts.edit_runtime.EditRuntime
// without speech recognition. Alignment runs FuseAlign (src/aligner) on clips of at most ALIGN_MAX_SECONDS; an edit
// generates each region on the speech-edit backbone (span-end stop, codes only; src/edit/fill.ts), splices the fills
// into the clip's codes and decodes the whole clip once. Transcripts (/v2/edit/transcribe, an edit without
// original_text that needs the clip's words) and verify answer 503 "Speech recognition is not available on this
// server: <fix>" before any model work.
import languages from '../../../../model/src/zonos3/registries/languages.json';
import { ALIGN_MAX_SECONDS, resampleTo16k } from '../aligner/host';
import { b64DecodeStrict, b64Encode } from '../b64';
import { regionDoses, type Doses } from '../edit/doses';
import { decodeCodes, fillPrompt, runFill, type FillJob } from '../edit/fill';
import { spliceEdit, validateEditSpans } from '../edit/frames';
import { pyFixed, pySplit, pyStrip } from '../edit/py';
import { alignTokens, canonicalMarkup, targetTokens, tokens } from '../edit/text';
import {
  alignedWords, diffSpans, editWords, expectedFillSeconds, planEditWindows, secondsToSpans, wholeClipWindow, windowSpans,
  type EditWindow, type EditWord, type Span,
} from '../edit/words';
import { EDIT_MAX_SECONDS, MAX_FRAMES, SAMPLE_RATE, SAMPLING } from '../generation';
import { NCB, type Prompt } from '../prompt';
import { stripNonverbalMarkup } from '../request/markup';
import { resolveControls, STEER_FRAMES_PER_CHAR, textChars } from '../request/steer';
import { defaultSampling, type Sampling } from '../sampler';
import { wavBytes } from '../wav';
import { editMaxSpans } from './capabilities';
import type { EditModels, Engine } from './engine';
import { as400, CONTEXT_CAPACITY, resolveConditionBuckets, resolveLanguage } from './generate';
import { bool, fail, HttpError, isSet, num, parseBody, str, type EngineRequest, type Sink, sendJson } from './http';

type Body = Record<string, unknown>;

/** zonos3.tts.edit_align.AsrUnavailable: the detail of every 503 a transcript or verify gets here, and its fixes. */
const ASR_UNAVAILABLE = 'Speech recognition is not available on this server: ';
const SEND_ORIGINAL_TEXT = "send the clip's transcript as original_text";
const SEND_TO_ALIGN = 'send the transcript to /v2/edit/align instead';
const VERIFY_FALSE = 'set verify to false';
/** The codec hop (samples per frame at SAMPLE_RATE) and its frame rate. */
const HOP = 2940;
const FRAME_RATE = SAMPLE_RATE / HOP;

/** Server language code -> the Whisper-style code its words are tokenized by (edit_align.EDIT_LANGUAGES). */
const SPEECH_CODE = new Map(languages.server.map(l => [l.code, l.speech]));

function asrUnavailable(fix: string): never {
  fail(503, ASR_UNAVAILABLE + fix);
}

/** EditRuntime.require_editing: the export's speech-edit adapter and aligner. */
function requireEditing(e: Engine): EditModels {
  if (!e.edit || editMaxSpans(e) <= 0)
    fail(503, 'Speech editing is not available on this server: the model export has no speech-edit adapter (edit) ' +
      'or FuseAlign aligner (aligner).');
  return e.edit;
}

/** The upload of `audio_base64` as the client decoded it (44.1 kHz mono, at most EDIT_MAX_SECONDS) and its length:
 *  EditRuntime._write_audio (400 for bad base64, an undecodable file or a clip over EDIT_MAX_SECONDS). */
function clipAudio(body: Body, req: EngineRequest): { pcm: Float32Array; duration: number } {
  const raw = str(body, 'audio_base64') ?? '';
  if (pyStrip(raw).startsWith('data:') && !raw.includes(','))
    fail(400, 'Invalid data URL for audio_base64.');
  if (!b64DecodeStrict(raw)) fail(400, 'audio_base64 must be valid base64-encoded data.');
  const pcm = req.pcm;
  if (!pcm || !pcm.length) fail(400, 'could not decode the audio (any container this browser decodes)');
  const duration = req.pcmSeconds ?? pcm.length / SAMPLE_RATE;
  if (duration > EDIT_MAX_SECONDS)
    fail(400, `Audio is ${pyFixed(duration, 1)} s long; at most ${pyFixed(EDIT_MAX_SECONDS, 0)} s are supported here.`);
  return { pcm, duration: pcm.length / SAMPLE_RATE };
}

/** EditRuntime._aligned_words: the words of `text` in the clip with their edit tokens (FuseAlign over align_tokens;
 *  one forward, so at most ALIGN_MAX_SECONDS of audio). */
async function alignText(edit: EditModels, pcm: Float32Array, text: string, lang: string, hint: string): Promise<EditWord[]> {
  const duration = pcm.length / SAMPLE_RATE;
  if (duration > ALIGN_MAX_SECONDS)
    fail(400, `Audio is ${pyFixed(duration, 1)} s long; this server aligns at most ${ALIGN_MAX_SECONDS} s ` +
      `(edit_align_max_seconds)${hint}.`);
  const words = alignTokens(text, lang);
  const aligned = words.length ? await edit.align(resampleTo16k(pcm, SAMPLE_RATE), words) : [];
  return as400(() => alignedWords(aligned, words.map(tokens)));
}

function publicWords(words: EditWord[]): unknown[] {
  return words.map(w => ({ word: w.word, start: w.start, end: w.end, confidence: w.confidence, tokens: w.tokens }));
}

/** POST /v2/edit/transcribe: needs speech recognition, which this server has none of. */
export function transcribeRoute(e: Engine, req: EngineRequest): never {
  const body = parseBody(req, 'EditTranscribeRequest');
  requireEditing(e);
  resolveLanguage(body);
  asrUnavailable(SEND_TO_ALIGN);
}

/** POST /v2/edit/align: FuseAlign word timings, presences (`confidence`) and edit tokens of `text` in the clip. */
export async function alignRoute(e: Engine, req: EngineRequest, sink: Sink): Promise<void> {
  const body = parseBody(req, 'EditAlignRequest');
  const edit = requireEditing(e);
  const text = pyStrip(str(body, 'text')!);
  if (!text) fail(400, 'text is empty.');
  const lang = SPEECH_CODE.get(resolveLanguage(body))!;
  const { pcm, duration } = clipAudio(body, req);
  const words = await alignText(edit, pcm, text, lang, '');
  sendJson(sink, { text, duration, words: publicWords(words) });
}

/** EditEngine.check: the target's pronunciation groups (this checkpoint has no phoneme tokens), every region's doses
 *  against the steering bundle and the condition buckets / values. Returns the buckets and the dosed ids that move
 *  the speaker embedding. */
function checkEdit(e: Engine, body: Body, targetText: string, doses: Doses[] | null): { buckets: number[]; raw: string[] } {
  if (targetText.includes('\uE000') || targetText.includes('\uE001'))
    fail(400, 'This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups');
  const raw = new Set<string>();
  if (doses) {
    const steer = e.steer;
    if (!steer)
      fail(400, 'This server has no steering bundle (place steer.gguf next to the backbone or pass --steer); ' +
        '`controls` are unavailable.');
    for (const region of doses) {
      as400(() => resolveControls(steer, region, [], 0, STEER_FRAMES_PER_CHAR, new Float32Array(e.hp.speakerDim)));
      for (const id of Object.keys(region)) if (!steer.dirs[steer.index.get(id)!].hidden) raw.add(id);
    }
  }
  return { buckets: resolveConditionBuckets(e, body), raw: [...raw].sort() };
}

/** The request's sampling (registry defaults), with the take's seed. */
function editSampling(e: Engine, body: Body): Sampling {
  const sp = defaultSampling();
  sp.temperature = num(body, 'temperature', SAMPLING.temperature);
  sp.top_k = num(body, 'topk', SAMPLING.topk);
  sp.top_p = num(body, 'top_p', SAMPLING.top_p);
  sp.min_p = num(body, 'min_p', SAMPLING.min_p);
  sp.rep_penalty = num(body, 'repetition_penalty', SAMPLING.repetition_penalty);
  sp.rep_window = num(body, 'repetition_window', SAMPLING.repetition_window);
  sp.rep_codebooks = num(body, 'repetition_codebooks', SAMPLING.repetition_codebooks);
  if (sp.rep_codebooks < 0) sp.rep_codebooks = e.hp.nCodebooks;
  sp.greedy = sp.temperature <= 0;
  return sp;
}

/** POST /v2/edit: regenerate the edited regions of recorded speech; unedited codec frames are kept exactly
 *  (EditRuntime.edit). */
export async function editRoute(e: Engine, req: EngineRequest, sink: Sink, aborted: () => boolean): Promise<void> {
  const body = parseBody(req, 'EditRequest');
  const edit = requireEditing(e);
  const maxSpans = editMaxSpans(e);
  if (num(body, 'tag_cfg_scale', 1) !== 1)
    fail(400, 'tag_cfg_scale is not supported by /v2/edit (speech-edit requests run without tag CFG); use 1.0.');
  // Nonverbal markup is text for the model only: markers (canonical, no inner whitespace) ride in the prompt text and
  // never reach FuseAlign (word diff, alignment; targetTokens).
  const targetText = canonicalMarkup(pyStrip(str(body, 'target_text')!));
  const lang = SPEECH_CODE.get(resolveLanguage(body))!;
  if (!targetText) fail(400, 'target_text is empty.');
  const givenWords = isSet(body, 'original_words')
    ? body.original_words as { word: string; start: number; end: number; tokens?: string[] | null; confidence?: number | null }[]
    : null;
  const givenText = str(body, 'original_text');
  if (givenWords && !pyStrip(givenText ?? '')) fail(400, 'original_words needs the original_text they spell.');
  const requestSpans = isSet(body, 'spans') ? body.spans as number[][] : null;
  // Without original_text (or original_words) the clip's words, when read (no explicit spans, or a clip to window),
  // would come from a transcript: refused before any model work.
  const untranscribed = givenWords === null && !givenText;
  if (bool(body, 'verify')) asrUnavailable(VERIFY_FALSE);
  if (untranscribed && requestSpans === null) asrUnavailable(SEND_ORIGINAL_TEXT);
  const doses = as400(() => regionDoses(
    isSet(body, 'controls') ? body.controls as Doses : null,
    isSet(body, 'region_controls') ? body.region_controls as (Doses | null)[] : null, requestSpans));
  const speakerRow = e.hp.speakerDim > 0;
  const useRow = isSet(body, 'speaker_embedding') ? bool(body, 'speaker_embedding')! : speakerRow;
  if (useRow && !speakerRow) fail(400, 'speaker_embedding needs a checkpoint with a speaker-embedding row.');
  const { buckets, raw } = checkEdit(e, body, targetText, doses);
  if (raw.length && !useRow)
    fail(400, `Controls ${raw.join(', ')} move the speaker embedding and need the speaker row (speaker_embedding true ` +
      'or unset, on a speaker-embedding checkpoint).');
  const { pcm, duration } = clipAudio(body, req);
  const maxContextSeconds = num(body, 'max_context_seconds', 16);
  if (untranscribed && requestSpans !== null && duration > maxContextSeconds) asrUnavailable(SEND_ORIGINAL_TEXT);
  // The clip's frames as the reference encoder makes them (one per started hop). Longer clips are windowed and every
  // fill gets a minimum length; both read word timings of the original (the caller's original_words, else a FuseAlign
  // alignment of original_text, checked before the encoders run). Without original_text, a short clip with explicit
  // spans is not aligned: its regions' own lengths are expected.
  const totalFrames = Math.ceil(pcm.length / HOP);
  const windowed = totalFrames / FRAME_RATE > maxContextSeconds;
  const targetWords = pySplit(targetText);
  const normalizedTarget = targetTokens(targetText);
  const minFillRatio = num(body, 'min_fill_ratio', 0.7);
  let originalText = givenText;
  if (originalText) originalText = stripNonverbalMarkup(originalText);
  let words: EditWord[] | null = null;
  if (givenWords) {
    const wordTokens = givenWords.map(w => w.tokens ?? tokens(w.word));
    words = as400(() => editWords(givenWords, wordTokens, tokens(originalText!), duration));
  } else if (requestSpans === null || windowed || (minFillRatio > 0 && originalText)) {
    if (!originalText) asrUnavailable(SEND_ORIGINAL_TEXT);
    words = await alignText(edit, pcm, originalText, lang, '; send original_words for a longer clip');
  }

  let speakerEmbedding: Float32Array | null = null;
  if (useRow) {
    try {
      speakerEmbedding = await e.embedSpeaker(pcm);
    } catch {
      fail(400, 'speaker embedding computation failed');
    }
    if (speakerEmbedding.length !== e.hp.speakerDim || !speakerEmbedding.every(Number.isFinite))
      fail(400, 'speaker embedding computation failed');
  }
  let codes: Int32Array;
  try {
    codes = await e.encodeReference(pcm);
  } catch {
    fail(400, `reference audio encoding failed (maximum ${EDIT_MAX_SECONDS} seconds)`);
  }
  if (codes.length !== totalFrames * NCB) fail(500, 'reference audio encoding returned an unexpected frame count');
  const spans: Span[] = as400(() => {
    const out = requestSpans !== null
      ? secondsToSpans(requestSpans, totalFrames, FRAME_RATE)
      : diffSpans(words!, normalizedTarget[0], {
        totalFrames, frameRate: FRAME_RATE, marginSeconds: num(body, 'margin_seconds', 0.08), maxSpans,
      }).spans;
    validateEditSpans(out, totalFrames, maxSpans);
    return out;
  });
  const windows: EditWindow[] = windowed
    ? as400(() => planEditWindows(spans, words!, targetWords, normalizedTarget.slice(1), {
      totalFrames, frameRate: FRAME_RATE, contextSeconds: num(body, 'context_seconds', 4), targetText,
    }))
    : [wholeClipWindow(spans, totalFrames, targetText)];
  const expected = words !== null
    ? as400(() => expectedFillSeconds(spans, words!, targetWords, normalizedTarget.slice(1), FRAME_RATE))
    : spans.map(([, length]) => length / FRAME_RATE);
  // The span-end token is masked until a fill reaches min_fill_ratio of its expected length.
  const maxSpanSeconds = num(body, 'max_span_seconds', 30);
  const minFrames = expected.map(s => Math.min(Math.trunc(minFillRatio * s * FRAME_RATE), Math.trunc(maxSpanSeconds * FRAME_RATE)));
  const maxTokens = Math.min(Math.trunc(maxSpanSeconds * FRAME_RATE) + NCB + 1, MAX_FRAMES);
  // Without explicit spans the request's controls apply to every diffed region alike.
  const regionControls = spans.map((_, k) => doses === null ? null : doses[requestSpans !== null ? k : 0]);
  const baseSeed = isSet(body, 'seed') ? Number(body.seed)
    : crypto.getRandomValues(new Uint32Array(1))[0] % (2 ** 31 - 8);
  const sampling = editSampling(e, body);
  const bb = await edit.backbone();

  const takes: unknown[] = [];
  for (let k = 0; k < num(body, 'num_takes', 1); k++) {
    const fills: Int32Array[] = new Array(spans.length), stopped: (boolean | null)[] = new Array(spans.length).fill(null);
    for (const window of windows) {
      const windowCodes = codes.subarray(window.start * NCB, window.end * NCB);
      const relative = windowSpans(window, spans);
      const windowFills: Int32Array[] = [];
      for (const region of window.regions) {
        // The region's steering: its speaker-row embedding moved by the raw doses and the hidden delta of the others.
        const embedding = speakerEmbedding ? Float32Array.from(speakerEmbedding) : null;
        const controls = regionControls[region];
        const steer = controls && Object.keys(controls).length
          ? as400(() => resolveControls(e.steer!, controls, [], textChars(targetText), STEER_FRAMES_PER_CHAR, embedding)).hidden
          : null;
        const job: FillJob = {
          text: window.text, sampling: { ...sampling, seed: (baseSeed + k) >>> 0 }, maxFrames: maxTokens,
          minFrames: minFrames[region], buckets, speakerEmbedding: embedding, steer, windowCodes, spans: relative,
          fills: [...windowFills],
        };
        let prompt: Prompt;
        try {
          prompt = fillPrompt(e.tokenizer, edit.hp, job);
        } catch (err) {
          fail(400, `could not build the edit prompt: ${err instanceof Error ? err.message : String(err)}`);
        }
        if (prompt.nRows >= CONTEXT_CAPACITY) fail(400, `TTS input len ${prompt.nRows} exceeds ${CONTEXT_CAPACITY}`);
        job.maxFrames = Math.min(job.maxFrames, CONTEXT_CAPACITY - prompt.nRows);
        let out: { fill: Int32Array; ended: boolean };
        try {
          out = await runFill(bb, edit.hp, prompt, job, aborted);
        } catch (err) {
          if (err instanceof HttpError) throw err;
          console.error('edit fill:', err);
          fail(500, 'generation failed');
        }
        windowFills.push(out.fill);
        fills[region] = out.fill;
        stopped[region] = out.ended;
      }
    }
    const spliced = spliceEdit(codes, spans, fills, NCB);
    let edited: Float32Array;
    try {
      edited = await decodeCodes(e.codec, spliced, e.hp.audioPad);
    } catch (err) {
      console.error('edit decode:', err);
      fail(500, 'decoding the edited codes failed');
    }
    const editedRegions: [number, number][] = [];
    let shift = 0;
    spans.forEach(([start, length], j) => {
      const n = fills[j].length / NCB;
      editedRegions.push([(start + shift) / FRAME_RATE, (start + shift + n) / FRAME_RATE]);
      shift += n - length;
    });
    takes.push({
      seed: baseSeed + k, audio_base64: b64Encode(wavBytes(edited, SAMPLE_RATE)), duration: edited.length / SAMPLE_RATE,
      edited_regions: editedRegions, fill_seconds: fills.map(f => f.length / NCB / FRAME_RATE), stopped,
      verify: null, words: null,
    });
  }
  sendJson(sink, {
    sample_rate: SAMPLE_RATE, original_text: originalText, target_text: targetText,
    words: words === null ? null : publicWords(words),
    spans: spans.map(([start, length]) => [start / FRAME_RATE, (start + length) / FRAME_RATE]),
    // The audio the model saw per window (full-clip seconds); null when the clip was edited whole.
    contexts: windowed ? windows.map(w => [w.start / FRAME_RATE, w.end / FRAME_RATE]) : null,
    // Each region's expected fill length (seconds); its fills last at least min_fill_ratio of it.
    expected_seconds: expected, takes,
  });
}

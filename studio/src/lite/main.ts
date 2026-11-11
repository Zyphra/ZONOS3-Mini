/**
 * Lite UI (`/lite`): voice cloning from a speaker embedding, and nothing else. Plain DOM over the markup in
 * `lite.html`, styled by the studio's own tokens and primitives (`../styles/base.css`); the server contract is `./api.ts`.
 */
import '@fontsource-variable/inter/wght.css';
import '@fontsource/jetbrains-mono/latin-400.css';
import '@fontsource/jetbrains-mono/latin-500.css';
import './lite.css';
import { mountChrome, setLabel } from '../lib/chrome';
import { bytesToBase64, download, encodeNpy, slug } from '../lib/files';
import { levelMeter } from '../lib/level';
import { wavBytes } from '../lib/wav';
import { LACI_LABEL, LACI_MODES, laciFields, type LaciMode } from '../lib/laci';
import { langLabel } from '../lib/languages';
import { byLanguage, voiceServerLanguage } from '../lib/voices';
import { drawWave } from '../lib/wave';
import { onEncoderProgress, selectWebEngine, usingWebEngine, type Download } from '../lib/web-engine';
import { ApiError } from '../lib/http';
import {
  canRunWeb,
  connect,
  createSpeaker,
  generateStream,
  listSpeakers,
  SAMPLE_RATE,
  server,
  speakerEmbedding,
  speakerPreview,
  type Backend,
  type GenerateRequest,
  type Speaker,
} from './api';
import { StreamPlayer } from './player';
import { CLIP_MAX_S, clipLengthProblem, readiness, selectedVoice, type Clip, type Source, type VoiceState } from './voice';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const ui = {
  backend: $<HTMLButtonElement>('backend'),
  backendNote: $<HTMLParagraphElement>('backend-note'),
  studioLink: $<HTMLAnchorElement>('studio-link'),
  notice: $<HTMLParagraphElement>('notice'),
  download: $<HTMLDivElement>('download'),
  consent: $<HTMLElement>('consent'),
  consentYes: $<HTMLButtonElement>('consent-yes'),
  consentRetry: $<HTMLButtonElement>('consent-retry'),
  sections: document.querySelectorAll<HTMLElement>('main section:not(#consent)'),
  sources: document.querySelectorAll<HTMLInputElement>('input[name="source"]'),
  panels: { default: $('panel-default'), clip: $('panel-clip'), embedding: $('panel-embedding') },
  voiceSelect: $<HTMLSelectElement>('voice-select'),
  clipPick: $<HTMLButtonElement>('clip-pick'),
  clipPlay: $<HTMLButtonElement>('clip-play'),
  clipFile: $<HTMLInputElement>('clip-file'),
  record: $<HTMLButtonElement>('record'),
  recordLevel: $<HTMLSpanElement>('record-level'),
  clipState: $<HTMLSpanElement>('clip-state'),
  clipLabel: $<HTMLInputElement>('clip-label'),
  clipUpload: $<HTMLButtonElement>('clip-upload'),
  embeddingPick: $<HTMLButtonElement>('embedding-pick'),
  embeddingFile: $<HTMLInputElement>('embedding-file'),
  embeddingState: $<HTMLSpanElement>('embedding-state'),
  embeddingLabel: $<HTMLInputElement>('embedding-label'),
  embeddingUpload: $<HTMLButtonElement>('embedding-upload'),
  voiceError: $<HTMLParagraphElement>('voice-error'),
  voice: $<HTMLDivElement>('voice'),
  voiceLabel: $<HTMLElement>('voice-label'),
  voiceMeta: $<HTMLSpanElement>('voice-meta'),
  preview: $<HTMLButtonElement>('preview'),
  downloadEmbedding: $<HTMLButtonElement>('download-embedding'),
  form: $<HTMLFormElement>('generate-form'),
  text: $<HTMLTextAreaElement>('text'),
  language: $<HTMLSelectElement>('language'),
  seed: $<HTMLInputElement>('seed'),
  laciField: $<HTMLDivElement>('laci-field'),
  laci: $<HTMLSelectElement>('laci'),
  laciNote: $<HTMLSpanElement>('laci-note'),
  generate: $<HTMLButtonElement>('generate'),
  stop: $<HTMLButtonElement>('stop'),
  status: $<HTMLSpanElement>('generate-status'),
  why: $<HTMLSpanElement>('generate-why'),
  seedUse: $<HTMLButtonElement>('seed-use'),
  error: $<HTMLParagraphElement>('generate-error'),
  result: $<HTMLDivElement>('result'),
  wave: $<HTMLDivElement>('wave'),
  play: $<HTMLButtonElement>('play'),
  resultDownload: $<HTMLAnchorElement>('result-download'),
  resultMeta: $<HTMLSpanElement>('result-meta'),
  takes: $<HTMLOListElement>('takes'),
};

mountChrome(() => {
  drawn = -1;
  animate();
});
let voices: Speaker[] = [];
/** The library voice picked in the select. */
let library: Speaker | null = null;
/** Which source the reader is on; Generate speaks only with a voice of that source. */
let source: Source = 'default';
/** The clip to clone: a chosen file or a finished recording. */
let clip: Clip | null = null;
/** The voice the embedding file was opened into. */
let fileVoice: Speaker | null = null;
/** The voice the language select last followed: picking another voice sets it to that voice's language. */
let languageOf: string | null = null;
/** Languages with LACI alignment heads on the server (`/v2/capabilities.laci_languages`). */
let laciLanguages: readonly string[] = [];
let generation: AbortController | null = null;
const player = new StreamPlayer(SAMPLE_RATE);
const preview = new Audio();
let previewUrl = '';

const state = (): VoiceState => ({ source, library, clip, fileVoice, text: ui.text.value, generating: !!generation });

// ── Waveform ──────────────────────────────────────────────────────────────────────────────────────

/** The take so far: `audio[0, samples)`, grown by doubling while it streams; drawn and played from here. */
let audio = new Float32Array(SAMPLE_RATE * 8);
let samples = 0;
/** Samples drawn on the canvas (-1: redraw); the playhead loop redraws when the stream has grown. */
let drawn = -1;
let frame = 0;
const waveCanvas = ui.wave.querySelector('canvas')!;
const waveHead = ui.wave.querySelector<HTMLElement>('.wave-head')!;

function append(pcm: Float32Array) {
  if (samples + pcm.length > audio.length) {
    const grown = new Float32Array(Math.max(audio.length * 2, samples + pcm.length));
    grown.set(audio.subarray(0, samples));
    audio = grown;
  }
  audio.set(pcm, samples);
  samples += pcm.length;
}

/** One frame: the waveform when the take has grown (or the theme or width changed), the playhead, the play button. */
function redraw() {
  frame = 0;
  if (drawn !== samples) {
    // The studio's preview look: orange bars stretched over what has arrived; `color` is the token's value.
    drawWave(waveCanvas, audio.subarray(0, samples), ui.wave.clientWidth, 72, getComputedStyle(ui.wave).color);
    drawn = samples;
  }
  const at = player.position();
  const playing = at !== null && at * SAMPLE_RATE < samples;
  waveHead.hidden = !playing;
  if (playing) {
    const pct = Math.min(100, ((at * SAMPLE_RATE) / samples) * 100);
    waveHead.style.left = `${pct}%`;
    ui.wave.setAttribute('aria-valuenow', String(Math.round(pct)));
    ui.wave.setAttribute('aria-valuetext', `${at.toFixed(1)} of ${(samples / SAMPLE_RATE).toFixed(1)} s`);
  }
  if (ui.play.dataset.icon !== (playing ? 'stop' : 'play')) {
    ui.play.dataset.icon = playing ? 'stop' : 'play';
    setLabel(ui.play, '');
    ui.play.title = playing ? 'Stop' : 'Play';
    ui.play.setAttribute('aria-label', ui.play.title);
  }
  if (playing || generation) frame = requestAnimationFrame(redraw);
}

function animate() {
  if (!frame) frame = requestAnimationFrame(redraw);
}

new ResizeObserver(() => {
  drawn = -1;
  animate();
}).observe(ui.wave);

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

function show(el: HTMLElement, text: string | null) {
  el.textContent = text ?? '';
  el.hidden = !text;
}

const seconds = (samples: number) => `${(samples / SAMPLE_RATE).toFixed(1)} s`;

/** The in-browser server's model download as a bar (`.dl` in styles/base.css); null hides it. */
function showDownload(d: Download | null) {
  ui.download.hidden = !d;
  if (!d) return;
  const pct = Math.round(d.fraction * 100);
  ui.download.setAttribute('aria-valuenow', String(pct));
  ui.download.setAttribute('aria-valuetext', `${d.label}: ${pct}%, ${d.detail}`);
  ui.download.setAttribute('aria-label', d.label);
  ui.download.querySelector('.dl-label')!.textContent = d.label;
  ui.download.querySelector('.dl-pct')!.textContent = `${pct}%`;
  ui.download.querySelector<HTMLElement>('.dl-fill')!.style.width = `${d.fraction * 100}%`;
  ui.download.querySelector('.dl-detail')!.textContent = d.detail;
}

/** Disables every control: the server cannot clone from an embedding (or did not answer). */
function disableAll(text: string) {
  show(ui.notice, text);
  ui.notice.classList.add('bad');
  for (const el of document.querySelectorAll<HTMLInputElement | HTMLButtonElement | HTMLSelectElement | HTMLTextAreaElement>(
    'main section:not(#consent) input, main section:not(#consent) button, main section:not(#consent) select, main section:not(#consent) textarea',
  )) {
    el.disabled = true;
  }
}

// ── Voices ────────────────────────────────────────────────────────────────────────────────────────

function renderVoiceSelect() {
  ui.voiceSelect.replaceChildren();
  const groups: [string, Speaker[]][] = [
    ...byLanguage(voices.filter((s) => s.scope === 'default')).map((g): [string, Speaker[]] => [g.label ?? 'Library', g.voices]),
    ['Your voices (this tab)', voices.filter((s) => s.scope === 'session')],
  ];
  for (const [label, list] of groups) {
    if (!list.length) continue;
    const group = document.createElement('optgroup');
    group.label = label;
    for (const s of list) group.append(new Option(s.label, s.id, false, s.id === library?.id));
    ui.voiceSelect.append(group);
  }
  if (!voices.length) ui.voiceSelect.append(new Option('No voice can be used from an embedding', '', true, true));
  ui.voiceSelect.disabled = !voices.length;
}

async function refreshVoices(select?: Speaker) {
  voices = (await listSpeakers()).filter((s) => s.speaker_modes.includes('embedding'));
  if (select && !voices.some((s) => s.id === select.id)) voices.push(select);
  const first = byLanguage(voices.filter((s) => s.scope === 'default'))[0]?.voices[0];
  library = (select?.scope === 'default' ? select : null) ?? voices.find((s) => s.id === library?.id) ?? first ?? voices[0] ?? null;
  renderVoiceSelect();
  showVoice();
}

/** The Selected box: the voice of the chosen source, or nothing until that source has one. */
function showVoice() {
  const s = selectedVoice(state());
  preview.pause();
  ui.voice.hidden = !s;
  if (s) {
    if (source === 'default') ui.voiceSelect.value = s.id;
    if (s.id !== languageOf) {
      languageOf = s.id;
      const lang = voiceServerLanguage(s, ui.language.value);
      if ([...ui.language.options].some((o) => o.value === lang)) ui.language.value = lang;
      updateLaciNote();
    }
    ui.voiceLabel.textContent = s.label;
    const scope = s.scope === 'default' ? 'library voice' : 'this tab';
    const from = s.source_type === 'audio' ? 'from a clip' : 'from a voice file';
    ui.voiceMeta.textContent = `(${scope}, ${from})`;
    ui.voiceMeta.title = s.embedding_dim ? `${s.embedding_dim}-dimensional voice embedding` : '';
    ui.preview.hidden = !s.has_preview;
    setLabel(ui.preview, 'Play preview', 'play');
  }
  updateGenerate();
}

function showSource(next: Source) {
  source = next;
  for (const [name, panel] of Object.entries(ui.panels)) panel.hidden = name !== next;
  show(ui.voiceError, null);
  showVoice();
}

/** Runs a voice action with its button busy; failures show inline under the voice section. */
async function voiceAction(button: HTMLButtonElement, busy: string, run: () => Promise<void>) {
  const idle = button.textContent!;
  button.disabled = true;
  button.classList.add('is-busy');
  setLabel(button, busy);
  show(ui.voiceError, null);
  try {
    await run();
  } catch (e) {
    show(ui.voiceError, message(e));
  } finally {
    setLabel(button, idle);
    button.classList.remove('is-busy');
    button.disabled = false;
    showDownload(null);
  }
}

const stem = (name: string) => name.replace(/\.[^.]+$/, '');

let clipUrl = '';
const clipAudio = new Audio();

function setClip(next: Clip | null, text: string) {
  clip = next;
  const problem = next ? clipLengthProblem(next.samples, SAMPLE_RATE) : null;
  const short = !!next && next.samples > 0 && next.samples < SAMPLE_RATE * 5 && !problem;
  ui.clipState.textContent = problem ?? (short ? `${text} · under 5 s: a longer clip clones better` : text);
  ui.clipState.classList.toggle('is-warn', short || !!problem);
  if (problem) clip = null;
  ui.clipUpload.disabled = !clip || !!clip.speaker;
  if (clip && !ui.clipLabel.value.trim()) ui.clipLabel.placeholder = stem(clip.name);
  clipAudio.pause();
  if (clipUrl) URL.revokeObjectURL(clipUrl);
  clipUrl = next ? URL.createObjectURL(next.blob) : '';
  ui.clipPlay.hidden = !next;
  setLabel(ui.clipPlay, 'Play', 'play');
  showVoice();
}

ui.clipPlay.addEventListener('click', () => {
  if (!clipAudio.paused) return clipAudio.pause();
  clipAudio.src = clipUrl;
  void clipAudio.play();
});
clipAudio.addEventListener('play', () => setLabel(ui.clipPlay, 'Stop', 'stop'));
clipAudio.addEventListener('pause', () => setLabel(ui.clipPlay, 'Play', 'play'));
clipAudio.addEventListener('ended', () => setLabel(ui.clipPlay, 'Play', 'play'));

/** A clip as a 16-bit WAV with its length (every server reads WAV; not every one reads the browser's webm/ogg). */
async function asWav(blob: Blob, name: string): Promise<{ blob: Blob; name: string; samples: number }> {
  try {
    const decoded = await new OfflineAudioContext(1, 1, SAMPLE_RATE).decodeAudioData(await blob.arrayBuffer());
    const mono = new Float32Array(decoded.length);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const data = decoded.getChannelData(c);
      for (let i = 0; i < mono.length; i++) mono[i] += data[i] / decoded.numberOfChannels;
    }
    return { blob: new Blob([wavBytes(mono, decoded.sampleRate)], { type: 'audio/wav' }), name: `${stem(name)}.wav`, samples: mono.length };
  } catch {
    return { blob, name, samples: 0 };
  }
}

/** Clone the current clip into a voice of this tab (Generate does this by itself when needed). */
async function cloneClip(): Promise<Speaker> {
  if (!clip) throw new Error('Record or upload a clip first');
  const s = await createSpeaker({
    label: ui.clipLabel.value.trim() || stem(clip.name),
    speaker_audio_base64: bytesToBase64(await clip.blob.arrayBuffer()),
    speaker_audio_name: clip.name,
  });
  clip = { ...clip, speaker: s };
  await refreshVoices(s);
  ui.clipUpload.disabled = true;
  ui.clipState.textContent = `Cloned “${s.label}”`;
  ui.clipState.classList.remove('is-warn');
  return s;
}

let recorder: { stop: () => void } | null = null;

async function toggleRecording() {
  if (recorder) return recorder.stop();
  show(ui.voiceError, null);
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    const denied = e instanceof DOMException && (e.name === 'NotAllowedError' || e.name === 'SecurityError');
    return show(ui.voiceError, denied ? 'Allow the microphone in the browser’s site settings' : `Microphone unavailable: ${message(e)}`);
  }
  const level = ui.recordLevel.firstElementChild as HTMLElement;
  const meter = levelMeter(stream, (v) => (level.style.width = `${Math.round(v * 100)}%`));
  ui.recordLevel.hidden = false;
  const rec = new MediaRecorder(stream);
  const parts: Blob[] = [];
  const started = performance.now();
  const clock = () => {
    const s = Math.floor((performance.now() - started) / 1000);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')} / 0:${CLIP_MAX_S}`;
  };
  const tick = setInterval(() => (ui.clipState.textContent = `Recording… ${clock()}`), 250);
  const limit = setTimeout(() => recorder?.stop(), CLIP_MAX_S * 1000);
  rec.ondataavailable = (ev) => parts.push(ev.data);
  rec.onstop = async () => {
    clearInterval(tick);
    clearTimeout(limit);
    meter.stop();
    ui.recordLevel.hidden = true;
    stream.getTracks().forEach((t) => t.stop());
    recorder = null;
    ui.record.classList.remove('is-recording');
    setLabel(ui.record, 'Record');
    ui.record.setAttribute('aria-pressed', 'false');
    const wav = await asWav(new Blob(parts, { type: rec.mimeType }), 'recording');
    ui.clipFile.value = '';
    setClip({ ...wav, speaker: null }, `Recording ready${wav.samples ? ` (${seconds(wav.samples)})` : ''}`);
  };
  recorder = { stop: () => rec.state === 'recording' && rec.stop() };
  rec.start();
  ui.record.classList.add('is-recording');
  ui.record.replaceChildren(Object.assign(document.createElement('span'), { className: 'rec-dot' }), 'Stop recording');
  ui.record.setAttribute('aria-pressed', 'true');
  ui.clipState.textContent = `Recording… 0:00 / 0:${CLIP_MAX_S}`;
}

// ── Generation ────────────────────────────────────────────────────────────────────────────────────

function updateGenerate() {
  const r = readiness(state());
  ui.generate.disabled = !r.ok;
  show(ui.why, r.ok ? null : r.why || null);
}
ui.text.addEventListener('input', updateGenerate);
ui.text.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    void generate();
  }
});

/** Earlier takes, newest first (the newest is the one in the waveform). */
interface TakeRecord {
  pcm: Float32Array;
  seed: string | null;
  text: string;
  url: string;
  label: string;
}
const MAX_TAKES = 5;
const takes: TakeRecord[] = [];

function renderTakes() {
  const older = takes.slice(1);
  ui.takes.hidden = !older.length;
  ui.takes.replaceChildren(
    ...older.map((t, i) => {
      const li = document.createElement('li');
      li.className = 'take-row';
      const play = Object.assign(document.createElement('button'), { type: 'button', className: 'icon-btn sm', title: 'Play this take' });
      play.dataset.icon = 'play';
      setLabel(play, '', 'play', 14);
      play.setAttribute('aria-label', `Play take ${takes.length - 1 - i}`);
      play.addEventListener('click', () => {
        preview.pause();
        void player.play(t.pcm, 0).then(animate);
      });
      const meta = Object.assign(document.createElement('span'), { className: 'meta', textContent: `${t.label} · ${seconds(t.pcm.length)} · ${t.text.length > 48 ? `${t.text.slice(0, 48)}…` : t.text}` });
      meta.title = t.seed ? `seed ${t.seed}` : '';
      const dl = Object.assign(document.createElement('a'), { className: 'btn sm ghost', href: t.url, download: `${slug(t.label)}${t.seed ? `-seed${t.seed}` : ''}.wav`, textContent: 'Download' });
      li.append(play, meta, dl);
      return li;
    }),
  );
}

function keepTake(pcm: Float32Array, seed: string | null, text: string, label: string) {
  const url = URL.createObjectURL(new Blob([wavBytes(pcm, SAMPLE_RATE)], { type: 'audio/wav' }));
  takes.unshift({ pcm, seed, text, url, label });
  for (const gone of takes.splice(MAX_TAKES)) URL.revokeObjectURL(gone.url);
  renderTakes();
  return url;
}

async function generate() {
  const r = readiness(state());
  if (!r.ok) return updateGenerate();
  const text = ui.text.value.trim();
  const ctrl = new AbortController();
  generation = ctrl;
  updateGenerate();
  ui.stop.hidden = false;
  show(ui.error, null);
  preview.pause();
  let voice = selectedVoice(state());
  try {
    if (!voice) {
      // Clip mode with a clip not yet cloned: Generate clones it first (the server does the embedding).
      ui.status.textContent = 'Cloning the clip…';
      voice = await cloneClip();
    }
  } catch (e) {
    generation = null;
    ui.stop.hidden = true;
    ui.status.textContent = '';
    show(ui.error, message(e));
    updateGenerate();
    return;
  }
  const body: GenerateRequest = {
    text,
    language: ui.language.value,
    speaker_mode: 'embedding',
    speaker_embedding_id: voice.id,
    stream: true,
    format: 'pcm',
  };
  Object.assign(body, laciFields(ui.laci.value as LaciMode, laciLanguages, body.language ?? 'en_us'));
  if (ui.seed.value !== '') body.seed = Number(ui.seed.value);
  ui.status.textContent = 'Generating…';
  samples = 0;
  drawn = -1;
  ui.result.hidden = false;
  ui.wave.classList.add('is-busy');
  ui.resultDownload.hidden = true;
  ui.resultMeta.textContent = '';
  animate();
  const t0 = performance.now();
  let firstAudio = 0;
  try {
    await player.begin();
    const { seed } = await generateStream(
      body,
      (pcm) => {
        firstAudio ||= performance.now();
        append(pcm);
        player.push(pcm);
        ui.status.textContent = `Playing while generating… ${seconds(samples)}`;
      },
      ctrl.signal,
    );
    player.flush();
    if (!samples) throw new ApiError(200, 'The server returned no audio.');
    const url = keepTake(audio.slice(0, samples), seed, text, voice.label);
    ui.resultDownload.href = url;
    ui.resultDownload.download = `${slug(voice.label)}${seed ? `-seed${seed}` : ''}.wav`;
    ui.resultDownload.hidden = false;
    ui.resultMeta.textContent = seconds(samples);
    ui.resultMeta.title = `seed ${seed ?? 'unknown'} · first audio after ${((firstAudio - t0) / 1000).toFixed(2)} s`;
    ui.seedUse.hidden = seed == null;
    ui.result.dataset.samples = String(samples);
    ui.result.dataset.seed = seed ?? '';
    ui.status.textContent = `Done: ${seconds(samples)} of audio.`;
  } catch (e) {
    player.stop();
    if (ctrl.signal.aborted) ui.status.textContent = 'Stopped.';
    else {
      ui.status.textContent = '';
      show(ui.error, message(e));
    }
    ui.result.hidden = !samples;
  } finally {
    generation = null;
    ui.wave.classList.remove('is-busy');
    ui.stop.hidden = true;
    updateGenerate();
    showDownload(null);
    animate();
  }
}

/** Plays the take from `fraction` of its length; while it plays, the play button stops it (generation goes on). */
async function playTake(fraction: number) {
  if (!samples || generation) return;
  preview.pause();
  await player.play(audio.subarray(0, samples), (fraction * samples) / SAMPLE_RATE);
  animate();
}

function togglePlayback() {
  if (player.position() === null) return void playTake(0);
  player.stop();
  animate();
}

ui.play.addEventListener('click', togglePlayback);
ui.wave.addEventListener('pointerdown', (e) => {
  const r = ui.wave.getBoundingClientRect();
  void playTake(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)));
});
ui.wave.addEventListener('keydown', (e) => {
  if (!samples) return;
  if (e.key === ' ') {
    e.preventDefault();
    togglePlayback();
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    e.preventDefault();
    const length = samples / SAMPLE_RATE;
    const at = Math.max(0, Math.min(length, (player.position() ?? 0) + (e.key === 'ArrowLeft' ? -1 : 1)));
    void playTake(at / length);
  }
});

// ── Wiring ────────────────────────────────────────────────────────────────────────────────────────

for (const input of ui.sources) input.addEventListener('change', () => input.checked && showSource(input.value as Source));

ui.voiceSelect.addEventListener('change', () => {
  library = voices.find((s) => s.id === ui.voiceSelect.value) ?? null;
  showVoice();
});

/** LACI on in a language without alignment heads: say that its takes run without it. */
function updateLaciNote() {
  show(ui.laciNote, ui.laci.value !== 'off' && !laciLanguages.includes(ui.language.value) ? `No alignment heads for ${langLabel(ui.language.value)}: takes run without LACI.` : null);
}
ui.laci.addEventListener('change', updateLaciNote);
ui.language.addEventListener('change', updateLaciNote);

ui.clipPick.addEventListener('click', () => ui.clipFile.click());
ui.clipFile.addEventListener('change', async () => {
  const file = ui.clipFile.files?.[0];
  if (!file) return setClip(null, '');
  ui.clipState.textContent = 'Reading…';
  const wav = await asWav(file, file.name);
  setClip({ ...wav, speaker: null }, `${file.name}${wav.samples ? ` (${seconds(wav.samples)})` : ''}`);
});
ui.record.addEventListener('click', () => void toggleRecording());
ui.clipUpload.addEventListener('click', () =>
  voiceAction(ui.clipUpload, 'Cloning…', async () => {
    await cloneClip();
    ui.clipLabel.value = '';
  }),
);

ui.embeddingPick.addEventListener('click', () => ui.embeddingFile.click());
const EMBEDDING_MAX_BYTES = 2e6;
ui.embeddingFile.addEventListener('change', () => {
  const file = ui.embeddingFile.files?.[0];
  show(ui.voiceError, null);
  const bad = file && (!/\.np[yz]$/i.test(file.name) || file.size > EMBEDDING_MAX_BYTES) ? 'Couldn’t read this voice file: it must be a .npy or .npz under 2 MB' : null;
  if (bad) {
    ui.embeddingFile.value = '';
    show(ui.voiceError, bad);
  }
  const ok = file && !bad;
  ui.embeddingUpload.disabled = !ok;
  ui.embeddingState.textContent = ok ? file.name : '';
  if (ok) ui.embeddingLabel.placeholder = stem(file.name);
});
ui.embeddingUpload.addEventListener('click', () =>
  voiceAction(ui.embeddingUpload, 'Opening…', async () => {
    const file = ui.embeddingFile.files?.[0];
    if (!file) return;
    let s: Speaker;
    try {
      s = await createSpeaker({
        label: ui.embeddingLabel.value.trim() || stem(file.name),
        speaker_embedding_base64: bytesToBase64(await file.arrayBuffer()),
        speaker_embedding_name: file.name,
      });
    } catch (e) {
      throw e instanceof ApiError && e.status === 400 ? new Error(`Couldn’t read this voice file: ${e.detail}`) : e;
    }
    fileVoice = s;
    await refreshVoices(s);
    ui.embeddingFile.value = '';
    ui.embeddingLabel.value = '';
    ui.embeddingState.textContent = `Opened “${s.label}”`;
    ui.embeddingUpload.disabled = true;
  }),
);

ui.preview.addEventListener('click', () => {
  const s = selectedVoice(state());
  if (!s) return;
  if (!preview.paused) return preview.pause();
  const id = s.id;
  // voiceAction restores the idle label when it settles; the playback state names the button after that.
  void voiceAction(ui.preview, 'Loading…', async () => {
    if (preview.dataset.id !== id) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(await speakerPreview(id));
      preview.src = previewUrl;
      preview.dataset.id = id;
    }
    await preview.play();
  }).then(() => preview.paused || playing(true));
});
/** The preview button while the preview plays: Stop, in orange. */
function playing(on: boolean) {
  setLabel(ui.preview, on ? 'Stop preview' : 'Play preview', on ? 'stop' : 'play');
  ui.preview.classList.toggle('is-playing', on);
}
preview.addEventListener('play', () => playing(true));
preview.addEventListener('pause', () => playing(false));

ui.downloadEmbedding.addEventListener('click', () => {
  const s = selectedVoice(state());
  if (!s) return;
  void voiceAction(ui.downloadEmbedding, 'Downloading…', async () => {
    download(encodeNpy(await speakerEmbedding(s.id)), `${slug(s.label)}.npy`, 'application/octet-stream');
  });
});

ui.form.addEventListener('submit', (ev) => {
  ev.preventDefault();
  void generate();
});
// Stop aborts the generation (what was heard so far stays); the play button only affects playback.
ui.stop.addEventListener('click', () => {
  generation?.abort();
  player.stop();
});
ui.seedUse.addEventListener('click', () => {
  ui.seed.value = ui.result.dataset.seed ?? '';
  ui.seed.closest('details')?.setAttribute('open', '');
  ui.seed.focus();
});

// ── Start ─────────────────────────────────────────────────────────────────────────────────────────

/** While the backend boots: "Connecting…", then the model download bar; the cards stay inert until it answers. */
function booting(on: boolean) {
  show(ui.notice, on ? 'Connecting…' : null);
  if (!on) showDownload(null);
  for (const section of ui.sections) section.inert = on;
}

/** No server: ask before the in-browser engine downloads (once per page load, like the studio). */
function askWeb(): Promise<boolean> {
  show(ui.notice, null);
  ui.consent.hidden = false;
  ui.consentYes.focus();
  const { promise, resolve } = Promise.withResolvers<boolean>();
  ui.consentYes.addEventListener('click', () => resolve(true), { once: true });
  ui.consentRetry.addEventListener('click', () => location.reload(), { once: true });
  return promise.finally(() => {
    ui.consent.hidden = true;
    show(ui.notice, 'Starting the in-browser engine…');
  });
}

/** The connection failed: the reason, Try again, and Run in this browser when this browser could. */
async function failed(e: unknown) {
  booting(false);
  const web = usingWebEngine();
  disableAll(web ? `The in-browser engine couldn’t start: ${message(e)}` : `Can’t reach the server at ${server || location.origin}: ${message(e)}`);
  const retry = Object.assign(document.createElement('button'), { type: 'button', className: 'btn sm', textContent: 'Try again' });
  retry.addEventListener('click', () => location.reload());
  ui.notice.append(' ', retry);
  if (!web && (await canRunWeb())) {
    const run = Object.assign(document.createElement('button'), { type: 'button', className: 'btn sm primary', textContent: 'Run in this browser' });
    run.addEventListener('click', () => {
      selectWebEngine();
      const u = new URL(location.href);
      u.searchParams.set('api', 'web');
      location.href = u.toString();
    });
    ui.notice.append(' ', run);
  }
}

async function start() {
  if (server) ui.studioLink.search = `?server=${encodeURIComponent(server)}`;
  updateGenerate();
  booting(true);
  let backend: Backend;
  try {
    backend = await connect({
      download: (d) => {
        show(ui.notice, null);
        showDownload(d);
      },
      confirmWeb: askWeb,
    });
  } catch (e) {
    await failed(e);
    return;
  }
  booting(false);
  const { kind, caps } = backend;
  ui.backend.hidden = false;
  ui.backend.querySelector('.pill-text')!.textContent = kind === 'web' ? 'In this browser' : 'Server';
  const explanation = kind === 'web' ? 'The voice engine runs in this tab on your graphics card; its models are cached in the browser.' : `Connected to ${server || location.origin}: that server renders every voice.`;
  ui.backend.title = explanation;
  ui.backend.classList.add(`pill-${kind}`);
  ui.backend.addEventListener('click', () => {
    const open = ui.backendNote.hidden;
    show(ui.backendNote, open ? explanation : null);
    ui.backend.setAttribute('aria-expanded', String(open));
  });
  if (kind === 'web') {
    ui.studioLink.search = server ? `?api=web&server=${encodeURIComponent(server)}` : '?api=web';
    void onEncoderProgress((d) => showDownload(d.fraction >= 1 ? null : d));
  }
  if (!caps.speaker_modes.includes('embedding')) {
    return disableAll('This server can’t clone from a voice file: its model has no voice-embedding input.');
  }
  const languages = caps.text_norm_languages.length ? caps.text_norm_languages : ['en_us'];
  for (const code of languages) ui.language.append(new Option(langLabel(code), code, false, code === 'en_us'));
  laciLanguages = caps.laci_languages;
  if (laciLanguages.length) {
    for (const m of LACI_MODES) ui.laci.append(new Option(LACI_LABEL[m], m, false, m === 'off'));
    ui.laciField.hidden = false;
  }
  if (!caps.speaker_audio_upload) {
    ui.clipUpload.disabled = true;
    ui.clipState.textContent = 'This server doesn’t take clip uploads.';
    ui.clipFile.disabled = ui.clipPick.disabled = ui.record.disabled = true;
  }
  if (typeof MediaRecorder === 'undefined' || !navigator.mediaDevices?.getUserMedia) ui.record.hidden = true;
  try {
    await refreshVoices();
  } catch (e) {
    show(ui.voiceError, `Couldn’t list voices: ${message(e)}`);
  }
}

void start();

/**
 * Lite UI (`/lite.html`): voice cloning from a speaker embedding, and nothing else. Plain DOM over the markup in
 * `lite.html`; the server contract is `./api.ts`.
 */
import './lite.css';
import { bytesToBase64, download, encodeNpy, encodeWav, slug } from '../lib/files';
import { langLabel } from '../lib/languages';
import { onEncoderProgress, usingWebEngine } from '../lib/web-engine';
import {
  ApiError,
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

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const ui = {
  backend: $<HTMLSpanElement>('backend'),
  studioLink: $<HTMLAnchorElement>('studio-link'),
  notice: $<HTMLParagraphElement>('notice'),
  sections: document.querySelectorAll<HTMLElement>('main section'),
  sources: document.querySelectorAll<HTMLInputElement>('input[name="source"]'),
  panels: { default: $('panel-default'), clip: $('panel-clip'), embedding: $('panel-embedding') },
  voiceSelect: $<HTMLSelectElement>('voice-select'),
  clipFile: $<HTMLInputElement>('clip-file'),
  record: $<HTMLButtonElement>('record'),
  clipState: $<HTMLSpanElement>('clip-state'),
  clipLabel: $<HTMLInputElement>('clip-label'),
  clipUpload: $<HTMLButtonElement>('clip-upload'),
  embeddingFile: $<HTMLInputElement>('embedding-file'),
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
  generate: $<HTMLButtonElement>('generate'),
  stop: $<HTMLButtonElement>('stop'),
  status: $<HTMLParagraphElement>('generate-status'),
  error: $<HTMLParagraphElement>('generate-error'),
  result: $<HTMLDivElement>('result'),
  resultAudio: $<HTMLAudioElement>('result-audio'),
  resultDownload: $<HTMLAnchorElement>('result-download'),
  resultMeta: $<HTMLSpanElement>('result-meta'),
};

let voices: Speaker[] = [];
let voice: Speaker | null = null;
/** The clip to clone: a chosen file or a finished recording. */
let clip: { blob: Blob; name: string } | null = null;
let generation: AbortController | null = null;
const player = new StreamPlayer(SAMPLE_RATE);
const preview = new Audio();
let previewUrl = '';
let resultUrl = '';

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

function show(el: HTMLElement, text: string | null) {
  el.textContent = text ?? '';
  el.hidden = !text;
}

const seconds = (samples: number) => `${(samples / SAMPLE_RATE).toFixed(2)} s`;

/** Disables every control: the server cannot clone from an embedding (or did not answer). */
function disableAll(text: string) {
  show(ui.notice, text);
  ui.notice.classList.add('bad');
  for (const el of document.querySelectorAll<HTMLInputElement | HTMLButtonElement | HTMLSelectElement | HTMLTextAreaElement>(
    'main input, main button, main select, main textarea',
  )) {
    el.disabled = true;
  }
}

// ── Voices ────────────────────────────────────────────────────────────────────────────────────────

function renderVoiceSelect() {
  ui.voiceSelect.replaceChildren();
  const groups: [string, Speaker[]][] = [
    ['Default voices', voices.filter((s) => s.scope === 'default')],
    ['Your voices (this tab)', voices.filter((s) => s.scope === 'session')],
  ];
  for (const [label, list] of groups) {
    if (!list.length) continue;
    const group = document.createElement('optgroup');
    group.label = label;
    for (const s of list) group.append(new Option(s.label, s.id, false, s.id === voice?.id));
    ui.voiceSelect.append(group);
  }
  if (!voices.length) ui.voiceSelect.append(new Option('No voice can be used from an embedding', '', true, true));
  ui.voiceSelect.disabled = !voices.length;
}

async function refreshVoices(select?: Speaker) {
  voices = (await listSpeakers()).filter((s) => s.speaker_modes.includes('embedding'));
  if (select && !voices.some((s) => s.id === select.id)) voices.push(select);
  renderVoiceSelect();
  setVoice(select ?? voices.find((s) => s.id === voice?.id) ?? voices[0] ?? null);
}

function setVoice(s: Speaker | null) {
  voice = s;
  preview.pause();
  ui.voice.hidden = !s;
  if (s) {
    ui.voiceSelect.value = s.id;
    ui.voiceLabel.textContent = s.label;
    const scope = s.scope === 'default' ? 'default voice' : 'this tab';
    const from = s.source_type === 'audio' ? 'from a clip' : 'from an embedding';
    ui.voiceMeta.textContent = `(${scope}, ${from}${s.embedding_dim ? `, ${s.embedding_dim}-d` : ''})`;
    ui.preview.hidden = !s.has_preview;
    ui.preview.textContent = 'Play preview';
  }
  updateGenerate();
}

function showSource(source: string) {
  for (const [name, panel] of Object.entries(ui.panels)) panel.hidden = name !== source;
  show(ui.voiceError, null);
  if (source === 'default') {
    const s = voices.find((v) => v.id === ui.voiceSelect.value);
    if (s) setVoice(s);
  }
}

/** Runs a voice action with its button busy; failures show inline under the voice section. */
async function voiceAction(button: HTMLButtonElement, busy: string, run: () => Promise<void>) {
  const idle = button.textContent;
  button.disabled = true;
  button.textContent = busy;
  show(ui.voiceError, null);
  try {
    await run();
  } catch (e) {
    show(ui.voiceError, message(e));
  } finally {
    button.textContent = idle;
    button.disabled = false;
    encoderLoaded();
  }
}

/**
 * Ends the encoder readout: the in-browser server loads its speaker encoder on a clip voice's first use (generate or
 * embedding download), its progress on the notice line meanwhile.
 */
function encoderLoaded() {
  if (!ui.notice.classList.contains('bad')) show(ui.notice, null);
}

const stem = (name: string) => name.replace(/\.[^.]+$/, '');

function setClip(next: typeof clip, state: string) {
  clip = next;
  ui.clipState.textContent = state;
  ui.clipUpload.disabled = !clip;
  if (clip && !ui.clipLabel.value.trim()) ui.clipLabel.placeholder = stem(clip.name);
}

/** A recording as a 16-bit WAV (every server reads WAV; not every one reads the browser's webm/ogg). */
async function recordingAsWav(blob: Blob): Promise<{ blob: Blob; name: string; samples: number }> {
  try {
    const decoded = await new OfflineAudioContext(1, 1, SAMPLE_RATE).decodeAudioData(await blob.arrayBuffer());
    const mono = new Float32Array(decoded.length);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const data = decoded.getChannelData(c);
      for (let i = 0; i < mono.length; i++) mono[i] += data[i] / decoded.numberOfChannels;
    }
    return { blob: new Blob([encodeWav(mono, decoded.sampleRate)], { type: 'audio/wav' }), name: 'recording.wav', samples: mono.length };
  } catch {
    return { blob, name: `recording.${blob.type.includes('ogg') ? 'ogg' : 'webm'}`, samples: 0 };
  }
}

let recorder: { stop: () => void } | null = null;

async function toggleRecording() {
  if (recorder) return recorder.stop();
  show(ui.voiceError, null);
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    return show(ui.voiceError, `Microphone unavailable: ${message(e)}`);
  }
  const rec = new MediaRecorder(stream);
  const parts: Blob[] = [];
  const started = performance.now();
  const tick = setInterval(() => (ui.clipState.textContent = `Recording… ${((performance.now() - started) / 1000).toFixed(0)} s`), 250);
  const limit = setTimeout(() => recorder?.stop(), 30_000);
  rec.ondataavailable = (ev) => parts.push(ev.data);
  rec.onstop = async () => {
    clearInterval(tick);
    clearTimeout(limit);
    stream.getTracks().forEach((t) => t.stop());
    recorder = null;
    ui.record.textContent = 'Record';
    ui.record.setAttribute('aria-pressed', 'false');
    const wav = await recordingAsWav(new Blob(parts, { type: rec.mimeType }));
    ui.clipFile.value = '';
    setClip(wav, `Recording ready${wav.samples ? ` (${seconds(wav.samples)})` : ''}`);
  };
  recorder = { stop: () => rec.state === 'recording' && rec.stop() };
  rec.start();
  ui.record.textContent = 'Stop recording';
  ui.record.setAttribute('aria-pressed', 'true');
  ui.clipState.textContent = 'Recording… 0 s';
}

// ── Generation ────────────────────────────────────────────────────────────────────────────────────

function updateGenerate() {
  ui.generate.disabled = !voice || !!generation;
}

function concat(chunks: Float32Array[], total: number): Float32Array {
  const pcm = new Float32Array(total);
  let at = 0;
  for (const c of chunks) {
    pcm.set(c, at);
    at += c.length;
  }
  return pcm;
}

async function generate() {
  if (!voice || generation) return;
  const text = ui.text.value.trim();
  if (!text) return;
  const body: GenerateRequest = {
    text,
    language: ui.language.value,
    speaker_mode: 'embedding',
    speaker_embedding_id: voice.id,
    stream: true,
    format: 'pcm',
  };
  if (ui.seed.value !== '') body.seed = Number(ui.seed.value);

  const ctrl = new AbortController();
  generation = ctrl;
  updateGenerate();
  ui.stop.hidden = false;
  show(ui.error, null);
  ui.result.hidden = true;
  ui.resultAudio.pause();
  preview.pause();
  ui.status.textContent = 'Generating…';
  const chunks: Float32Array[] = [];
  let total = 0;
  const t0 = performance.now();
  let firstAudio = 0;
  try {
    await player.begin();
    const { seed } = await generateStream(
      body,
      (pcm) => {
        firstAudio ||= performance.now();
        chunks.push(pcm);
        total += pcm.length;
        player.push(pcm);
        ui.status.textContent = `Playing while generating… ${seconds(total)}`;
      },
      ctrl.signal,
    );
    player.flush();
    if (!total) throw new ApiError(200, 'The server returned no audio.');
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = URL.createObjectURL(new Blob([encodeWav(concat(chunks, total), SAMPLE_RATE)], { type: 'audio/wav' }));
    ui.resultAudio.src = resultUrl;
    ui.resultDownload.href = resultUrl;
    ui.resultDownload.download = `${slug(voice.label)}${seed ? `-seed${seed}` : ''}.wav`;
    ui.resultMeta.textContent = `${seconds(total)} · seed ${seed ?? 'unknown'} · first audio after ${((firstAudio - t0) / 1000).toFixed(2)} s`;
    ui.result.dataset.samples = String(total);
    ui.result.dataset.seed = seed ?? '';
    ui.result.hidden = false;
    ui.status.textContent = `Done: ${seconds(total)} of audio.`;
  } catch (e) {
    player.stop();
    if (ctrl.signal.aborted) ui.status.textContent = 'Stopped.';
    else {
      ui.status.textContent = '';
      show(ui.error, message(e));
    }
  } finally {
    generation = null;
    ui.stop.hidden = true;
    updateGenerate();
    encoderLoaded();
  }
}

// ── Wiring ────────────────────────────────────────────────────────────────────────────────────────

for (const input of ui.sources) input.addEventListener('change', () => input.checked && showSource(input.value));

ui.voiceSelect.addEventListener('change', () => setVoice(voices.find((s) => s.id === ui.voiceSelect.value) ?? null));

ui.clipFile.addEventListener('change', () => {
  const file = ui.clipFile.files?.[0];
  setClip(file ? { blob: file, name: file.name } : null, '');
});
ui.record.addEventListener('click', () => void toggleRecording());
ui.clipUpload.addEventListener('click', () =>
  voiceAction(ui.clipUpload, 'Cloning…', async () => {
    if (!clip) return;
    const s = await createSpeaker({
      label: ui.clipLabel.value.trim() || stem(clip.name),
      speaker_audio_base64: bytesToBase64(await clip.blob.arrayBuffer()),
      speaker_audio_name: clip.name,
    });
    await refreshVoices(s);
  }),
);

ui.embeddingFile.addEventListener('change', () => {
  const file = ui.embeddingFile.files?.[0];
  ui.embeddingUpload.disabled = !file;
  if (file) ui.embeddingLabel.placeholder = stem(file.name);
});
ui.embeddingUpload.addEventListener('click', () =>
  voiceAction(ui.embeddingUpload, 'Loading…', async () => {
    const file = ui.embeddingFile.files?.[0];
    if (!file) return;
    const s = await createSpeaker({
      label: ui.embeddingLabel.value.trim() || stem(file.name),
      speaker_embedding_base64: bytesToBase64(await file.arrayBuffer()),
      speaker_embedding_name: file.name,
    });
    await refreshVoices(s);
  }),
);

ui.preview.addEventListener('click', () => {
  if (!voice) return;
  if (!preview.paused) return preview.pause();
  const id = voice.id;
  // voiceAction restores the idle label when it settles; the playback state names the button after that.
  void voiceAction(ui.preview, 'Loading…', async () => {
    if (preview.dataset.id !== id) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(await speakerPreview(id));
      preview.src = previewUrl;
      preview.dataset.id = id;
    }
    await preview.play();
  }).then(() => (ui.preview.textContent = preview.paused ? 'Play preview' : 'Stop preview'));
});
preview.addEventListener('play', () => (ui.preview.textContent = 'Stop preview'));
preview.addEventListener('pause', () => (ui.preview.textContent = 'Play preview'));

ui.downloadEmbedding.addEventListener('click', () => {
  const s = voice;
  if (!s) return;
  void voiceAction(ui.downloadEmbedding, 'Downloading…', async () => {
    download(encodeNpy(await speakerEmbedding(s.id)), `${slug(s.label)}.npy`, 'application/octet-stream');
  });
});

ui.form.addEventListener('submit', (ev) => {
  ev.preventDefault();
  void generate();
});
ui.stop.addEventListener('click', () => {
  generation?.abort();
  player.stop();
});

// ── Start ─────────────────────────────────────────────────────────────────────────────────────────

/** The backend's readout under the header while it boots; the cards stay inert until it answers. */
function booting(text: string | null) {
  show(ui.notice, text);
  for (const section of ui.sections) section.inert = text !== null;
}

async function start() {
  if (server) ui.studioLink.search = `?server=${encodeURIComponent(server)}`;
  updateGenerate();
  booting('Connecting…');
  let backend: Backend;
  try {
    backend = await connect(booting);
  } catch (e) {
    booting(null);
    return disableAll(
      usingWebEngine()
        ? `The in-browser server could not start: ${message(e)}`
        : `Cannot reach the server${server ? ` at ${server}` : ''}: ${message(e)}`,
    );
  }
  booting(null);
  const { kind, caps } = backend;
  if (kind === 'web') {
    ui.backend.hidden = false;
    ui.studioLink.search = '?api=web';
    void onEncoderProgress((text) => show(ui.notice, text));
  }
  if (!caps.speaker_modes.includes('embedding')) {
    return disableAll(
      'This server cannot clone from a speaker embedding: its checkpoint does not take speaker embeddings ' +
        `(speaker modes: ${caps.speaker_modes.join(', ') || 'none'}). Use a server whose checkpoint has an embedding speaker input.`,
    );
  }
  const languages = caps.text_norm_languages.length ? caps.text_norm_languages : ['en_us'];
  for (const code of languages) ui.language.append(new Option(langLabel(code), code, false, code === 'en_us'));
  if (!caps.speaker_audio_upload) {
    ui.clipUpload.disabled = true;
    ui.clipState.textContent = 'This server does not take clip uploads.';
    ui.clipFile.disabled = ui.record.disabled = true;
  }
  if (typeof MediaRecorder === 'undefined' || !navigator.mediaDevices?.getUserMedia) ui.record.hidden = true;
  try {
    await refreshVoices();
  } catch (e) {
    show(ui.voiceError, `Could not list voices: ${message(e)}`);
  }
}

void start();

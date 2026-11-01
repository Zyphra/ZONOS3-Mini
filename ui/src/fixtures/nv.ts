/**
 * Nonverbal tags on the client-side synth, so the studio's NV path can be heard offline. Point tags splice in a
 * sound between the words (laugh, breath, sigh, panting; cough barely, as measured on the NV model); span tags shape
 * their words (shouting louder and brighter, crying with a tremble), mildly, as the model does. Tag guidance
 * (`tag_cfg_scale`) makes the sounds stronger.
 */
import type { ControlSpan } from '../api';
import { parseTags, type TagName } from '../lib/tags';
import type { SynthWord } from './synth';

export interface Prepared {
  /** The text without markup (what the babble speaks). */
  text: string;
  spans: ControlSpan[];
  points: { at: number; name: TagName }[];
  regions: { start: number; end: number; name: TagName }[];
  tagged: boolean;
}

/** Remove the markup; painted spans, point tags and span tags move to offsets on the spoken text. */
export function prepare(text: string, spans: ControlSpan[] | undefined): Prepared {
  const { tokens, spans: tagSpans } = parseTags(text);
  if (!tokens.length) return { text, spans: spans ?? [], points: [], regions: [], tagged: false };
  // Spoken offset of each sent offset.
  const map = new Int32Array(text.length + 1);
  let out = '';
  let k = 0;
  for (let i = 0; i < text.length; i++) {
    map[i] = out.length;
    while (k < tokens.length && i >= tokens[k].end) k++;
    if (!(k < tokens.length && i >= tokens[k].start)) out += text[i];
  }
  map[text.length] = out.length;
  return {
    text: out,
    spans: (spans ?? []).map((s) => ({ ...s, start: map[s.start], end: map[s.end] })).filter((s) => s.end > s.start),
    points: tokens.filter((t) => t.kind === 'point').map((t) => ({ at: map[t.start], name: t.name })),
    regions: tagSpans.map((s) => ({ start: map[s.open.end], end: map[s.close.start], name: s.name })),
    tagged: true,
  };
}

function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}

/** One nonverbal sound; `k` is its strength (tag guidance). */
function sound(name: TagName, sr: number, k: number, seed: number): Float32Array {
  const r = rng(seed);
  const make = (sec: number) => new Float32Array(Math.round(sec * sr));
  const noiseBurst = (buf: Float32Array, at: number, len: number, amp: number, cutoff: number, f0 = 0) => {
    let y = 0;
    const a = Math.exp((-2 * Math.PI * cutoff) / sr);
    const n0 = Math.round(at * sr);
    const n = Math.round(len * sr);
    for (let i = 0; i < n && n0 + i < buf.length; i++) {
      const env = Math.sin((Math.PI * i) / n) ** 2;
      y = a * y + (1 - a) * r();
      const voice = f0 ? 0.6 * Math.sin((2 * Math.PI * f0 * i) / sr) * (0.6 + 0.4 * Math.sin((2 * Math.PI * 9 * i) / sr)) : 0;
      buf[n0 + i] += amp * env * (y * 3 + voice);
    }
  };
  switch (name) {
    case 'laugh': {
      const n = 3 + Math.round(k);
      const b = make(0.16 * n + 0.15);
      for (let j = 0; j < n; j++) noiseBurst(b, 0.05 + j * 0.16, 0.11, 0.12 * Math.min(k, 3) * (1 - j * 0.08), 2500, 230 - j * 8);
      return b;
    }
    case 'breath': {
      const b = make(0.55);
      noiseBurst(b, 0.04, 0.45, 0.07 * Math.min(k, 3), 1400);
      return b;
    }
    case 'sigh': {
      const b = make(0.85);
      noiseBurst(b, 0.04, 0.75, 0.08 * Math.min(k, 3), 900, 140);
      return b;
    }
    case 'panting': {
      const b = make(0.95);
      for (let j = 0; j < 5; j++) noiseBurst(b, 0.03 + j * 0.18, 0.13, 0.08 * Math.min(k, 3), 1800);
      return b;
    }
    case 'cough': {
      // Measured ≈ no effect on the NV model: a barely-there catch.
      const b = make(0.14);
      noiseBurst(b, 0.02, 0.08, 0.015, 1200);
      return b;
    }
    default:
      // shouting / crying as points: the model hardly reacts; a short pause.
      return make(0.08);
  }
}

/** Splice the point-tag sounds in and shape the span-tag words; word timings follow. */
export function voice(pcm: Float32Array, sr: number, words: SynthWord[], p: Prepared, scale: number, seed: number): { pcm: Float32Array; words: SynthWord[] } {
  if (!p.tagged) return { pcm, words };
  const k = Math.max(1, scale);
  // Character start of each word the synth timed (words with letters, in order).
  const starts = [...p.text.matchAll(/[^\s]+/g)].filter((m) => /[\p{L}']/u.test(m[0])).map((m) => m.index ?? 0);
  const ends = [...p.text.matchAll(/[^\s]+/g)].filter((m) => /[\p{L}']/u.test(m[0])).map((m) => (m.index ?? 0) + m[0].length);
  const timeAt = (at: number) => {
    let t = 0.05;
    for (let i = 0; i < words.length && i < starts.length; i++) if (starts[i] < at) t = words[i].end;
    return t;
  };
  // Span tags: gain and tremble over their words (weak, as measured).
  const out0 = pcm.slice();
  for (const reg of p.regions) {
    const idx = starts.map((s, i) => (s >= reg.start && ends[i] <= reg.end + 1 ? i : -1)).filter((i) => i >= 0 && i < words.length);
    if (!idx.length) continue;
    const a = Math.floor(words[idx[0]].start * sr);
    const b = Math.min(out0.length, Math.ceil(words[idx[idx.length - 1]].end * sr));
    for (let i = a; i < b; i++) {
      if (reg.name === 'shouting') out0[i] = Math.tanh(out0[i] * (1 + 0.35 * k)) * 0.9;
      else if (reg.name === 'crying') out0[i] *= 1 + 0.25 * Math.min(k, 3) * Math.sin((2 * Math.PI * 6 * i) / sr);
    }
  }
  // Point tags, in time order: insert each sound, shifting what follows.
  const cuts = p.points.map((pt, j) => ({ t: timeAt(pt.at), s: sound(pt.name, sr, k, seed + j * 97) })).sort((x, y) => x.t - y.t);
  if (!cuts.length) return { pcm: out0, words };
  const extra = cuts.reduce((n, c) => n + c.s.length, 0);
  const out = new Float32Array(out0.length + extra);
  let src = 0;
  let dst = 0;
  let shift = 0;
  const moved = words.map((w) => ({ ...w }));
  for (const c of cuts) {
    const at = Math.min(out0.length, Math.max(src, Math.round(c.t * sr)));
    out.set(out0.subarray(src, at), dst);
    dst += at - src;
    src = at;
    out.set(c.s, dst);
    dst += c.s.length;
    const dt = c.s.length / sr;
    for (const w of moved) if (w.start >= c.t - 1e-6 + shift) {
      w.start += dt;
      w.end += dt;
    }
    shift += dt;
  }
  out.set(out0.subarray(src), dst);
  return { pcm: out, words: moved };
}

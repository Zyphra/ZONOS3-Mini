/**
 * Offline stand-in for /tts/generate: a small source-filter babbler (no words, just speech-like syllables)
 * driven exactly by what a request carries. Raw doses steer the embedding with the contract's formula and the
 * voice is read back from its projections; hidden doses (global and per character span) shape the delivery.
 * Only used when no server answers, so every Studio control stays audible.
 */
import type { ControlSpan, ControlsSpec, Dose } from '../api';
import { steerEmbedding } from '../state/engine';
import { clamp, hash32, rng } from '../lib/math';

const SR = 44100;
const VOWELS: Record<string, [number, number, number]> = {
  a: [730, 1090, 2440],
  e: [530, 1840, 2480],
  i: [300, 2250, 3000],
  o: [570, 840, 2410],
  u: [320, 900, 2240],
};

interface Syl {
  start: number;
  end: number;
  vowel: string;
  onset: boolean;
  pause: number;
  stress: number;
}

function syllables(text: string, seed: number): Syl[] {
  const out: Syl[] = [];
  const r = rng(seed);
  for (const m of text.matchAll(/[\p{L}']+|[,;:.!?]/gu)) {
    const tok = m[0];
    const at = m.index ?? 0;
    if (/^[,;:.!?]$/.test(tok)) {
      const last = out[out.length - 1];
      if (last) last.pause += tok === ',' ? 0.16 : tok === ';' || tok === ':' ? 0.22 : 0.38;
      continue;
    }
    const groups = [...tok.toLowerCase().matchAll(/[aeiouy]+/g)];
    const n = Math.max(1, groups.length);
    for (let k = 0; k < n; k++) {
      const v = groups[k]?.[0]?.[0] ?? 'a';
      out.push({
        start: at + Math.floor((tok.length * k) / n),
        end: at + Math.floor((tok.length * (k + 1)) / n),
        vowel: v === 'y' ? 'i' : v,
        onset: k > 0 || !/^[aeiouy]/i.test(tok),
        pause: k === n - 1 ? 0.035 : 0,
        stress: k === 0 ? 1 : 0.55 + 0.3 * r(),
      });
    }
  }
  return out;
}

const sig = (x: number) => 1 / (1 + Math.exp(-x));

/** Absolute coordinate of e on a raw direction: e·v / |v|². */
function coord(spec: ControlsSpec, e: Float32Array, id: string): number {
  const v = spec.directions[id]?.vector;
  if (!v) return 0;
  let s = 0;
  let n = 0;
  for (let i = 0; i < v.length; i++) {
    s += e[i] * v[i];
    n += v[i] * v[i];
  }
  return s / (n || 1);
}

export interface SynthInput {
  text: string;
  embedding: Float32Array;
  controls?: Dose;
  spans?: ControlSpan[];
  seed: number;
}

/** A word of the synthesized text and when it is spoken (seconds). */
export interface SynthWord {
  word: string;
  start: number;
  end: number;
}

export function synthesize(spec: ControlsSpec, inp: SynthInput): Float32Array {
  return synthesizeWords(spec, inp).pcm;
}

/** The babble plus exact word timings (what a perfect forced aligner would return). */
export function synthesizeWords(spec: ControlsSpec, inp: SynthInput): { pcm: Float32Array; words: SynthWord[] } {
  const doses = inp.controls ?? {};
  const e = steerEmbedding(spec, inp.embedding, doses);
  const c = (id: string) => coord(spec, e, id);
  const g = c('identity.gender');
  const age = c('identity.age');
  const pitch = c('pitch.pitch');
  const range = c('pitch.pitch_range');
  const deep = c('texture.deep');
  const shrill = c('texture.shrill');
  const husky = c('texture.husky');
  const raspy = c('texture.raspy');
  const soft = c('texture.soft');
  const auth = c('texture.authoritative');
  const clean = c('room.clean');
  // Accents: each nudges the vowel space its own (deterministic) way; L1 accents also even out the rhythm.
  let f1k = 1;
  let f2k = 1;
  let evenRhythm = 0;
  for (const [id, d] of Object.entries(spec.directions)) {
    if (d.space !== 'raw' || !/^(accents_|mandarin\.)/.test(id)) continue;
    const a = c(id);
    if (Math.abs(a) < 0.05) continue;
    const h = hash32(id);
    f1k *= 1 + a * (((h & 0xff) / 255) - 0.5) * 0.12;
    f2k *= 1 + a * ((((h >> 8) & 0xff) / 255) - 0.5) * 0.14;
    if (!id.startsWith('accents_en')) evenRhythm += Math.abs(a) * 0.3;
  }

  const fem = sig(g * 1.8);
  const f0Base = 108 * Math.pow(2, fem * 0.9 + pitch * 0.11 + shrill * 0.06 - deep * 0.05 - Math.max(age, 0) * 0.04 + Math.max(-age, 0) * 0.03);
  const formant = (1 + 0.17 * fem - 0.02 * deep + 0.02 * shrill - 0.015 * age) * 1;
  const contourBase = 2.6 * (1 + 0.3 * range);
  const breathBase = 0.04 + 0.07 * Math.max(husky, 0) + 0.02 * Math.max(soft, 0);
  const rasp = 0.06 * Math.max(raspy, 0) + 0.01 * Math.max(age, 0);
  const tiltBase = 0.9 + 0.05 * Math.max(soft, 0) - 0.03 * Math.max(auth, 0);
  const gainBaseDb = -1.9 * Math.max(soft, 0) + 0.5 * Math.max(auth, 0);

  const sylls = syllables(inp.text, hash32(inp.text));
  const r = rng(inp.seed * 7919 + 13);
  const hidden = (at: number): Dose => {
    const h: Dose = {};
    for (const [id, v] of Object.entries(doses)) if (spec.directions[id]?.space !== 'raw') h[id] = v;
    for (const s of inp.spans ?? []) {
      if (at >= s.start && at < s.end) for (const [id, v] of Object.entries(s.controls)) h[id] = (h[id] ?? 0) + v;
    }
    return h;
  };

  // Plan the syllables in time.
  interface Plan { t0: number; dur: number; f0: number; slope: number; f: [number, number, number]; voiced: number; gain: number; tilt: number; breath: number; onset: boolean }
  const plan: Plan[] = [];
  let t = 0.12;
  const meanDur = 0.17 * (1 + 0.03 * Math.max(auth, 0));
  // Named moods beyond angry / happy / sad move valence and arousal by their measured landmark (pad −1…1 at
  // their default dose ≈ ±6 axis dose at the pad edge).
  const landmarks = (spec.ui.moods ?? []).flatMap((m) => {
    const ctl = spec.ui.controls.find((c) => c.id === m.control);
    if (!ctl || ['angry', 'happy', 'sad'].includes(m.control) || !m.dose) return [];
    return Object.entries(ctl.map['+']).map(([dir, w]) => ({ dir, val: (m.valence * 6 * w) / m.dose, aro: (m.arousal * 6 * w) / m.dose }));
  });
  sylls.forEach((s, i) => {
    const h = hidden((s.start + s.end) >> 1);
    const q = (id: string) => h[id] ?? 0;
    const moodVal = landmarks.reduce((a, l) => a + l.val * q(l.dir), 0);
    const moodAro = landmarks.reduce((a, l) => a + l.aro * q(l.dir), 0);
    const tempo = q('delivery.tempo') + 0.06 * (q('mood.arousal') + moodAro) - 0.08 * q('emotions.sad');
    const rate = tempo >= 0 ? 1 + 0.3 * tempo : 1 / (1 + 0.45 * -tempo);
    const conv = q('delivery.conversational');
    const dis = Math.max(q('disfluency.disfluent'), 0);
    const expr = q('expressive.expressive');
    const whisper = Math.max(q('delivery.whisper'), 0);
    const loud = Math.max(q('delivery.loud'), 0);
    const val = q('mood.valence') + moodVal;
    const aro = q('mood.arousal') + moodAro;
    const angry = Math.max(q('emotions.angry'), 0);
    const happy = Math.max(q('emotions.happy'), 0);
    const sad = Math.max(q('emotions.sad'), 0);
    let dur = (meanDur / rate) * (1 + 0.012 * expr) * (s.stress > 0.9 ? 1.12 : 0.92);
    dur *= 1 + (r() - 0.5) * (0.12 + 0.03 * Math.max(conv, 0)) * (1 - clamp(evenRhythm, 0, 0.8));
    if (dis > 0 && r() < 0.03 * dis) dur *= 1.4 + 0.05 * dis;
    const pos = i / Math.max(1, sylls.length - 1);
    const depth = contourBase * (1 + 0.06 * expr + 0.15 * aro + 0.15 * happy - 0.12 * sad - 0.05 * Math.max(conv, 0));
    const st = depth * ((s.stress - 0.7) * 1.4 + (r() - 0.5) * 0.6) - 2.2 * pos + 0.8 * loud + 0.25 * expr + 0.3 * val + 1.2 * happy - sad + 0.5 * angry;
    const f0 = f0Base * Math.pow(2, st / 12);
    const vf = VOWELS[s.vowel] ?? VOWELS.a;
    const fk = formant * (1 + 0.015 * val + 0.02 * loud);
    plan.push({
      t0: t,
      dur,
      f0,
      slope: (r() - 0.5) * 0.08 - 0.04 * pos,
      f: [vf[0] * fk * f1k, vf[1] * fk * f2k, vf[2] * fk],
      voiced: clamp(1 - 0.3 * whisper, 0.04, 1),
      gain: Math.pow(10, (gainBaseDb + 3.5 * loud - 4.5 * whisper + 0.6 * conv + 1 * aro + 2 * angry - 1.5 * sad) / 20),
      tilt: clamp(tiltBase - 0.06 * loud - 0.05 * angry + 0.04 * whisper, 0.55, 0.995),
      breath: breathBase + 0.12 * whisper + 0.03 * angry,
      onset: s.onset,
    });
    t += dur + s.pause * (1 / Math.sqrt(rate)) + (dis > 0 && r() < 0.02 * dis ? 0.18 + 0.2 * r() : 0);
  });
  const total = Math.ceil((t + 0.25) * SR);
  const out = new Float32Array(total);
  const words: SynthWord[] = [];
  for (const m of inp.text.matchAll(/[^\s]+/g)) {
    const from = m.index ?? 0;
    const to = from + m[0].length;
    const own = sylls.map((sy, k) => (sy.start >= from && sy.start < to ? k : -1)).filter((k) => k >= 0);
    if (!own.length) continue;
    const first = plan[own[0]];
    const last = plan[own[own.length - 1]];
    words.push({ word: m[0], start: first.t0, end: last.t0 + last.dur });
  }

  // Render: polyBLEP saw → spectral tilt → three parallel formant resonators, plus aspiration and onset noise.
  let phase = 0;
  const ys = [new Float64Array(2), new Float64Array(2), new Float64Array(2)];
  let tiltY = 0;
  let curF = plan[0]?.f.slice() ?? [500, 1500, 2500];
  for (const p of plan) {
    const n0 = Math.floor(p.t0 * SR);
    const n = Math.floor(p.dur * SR);
    const coef = [0, 0, 0].map(() => ({ b1: 0, b2: 0, a0: 0 }));
    for (let k = 0; k < n && n0 + k < total; k++) {
      if ((k & 31) === 0) {
        for (let j = 0; j < 3; j++) {
          curF[j] += (p.f[j] - curF[j]) * 0.25;
          const bw = 60 + j * 40;
          const rr = Math.exp((-Math.PI * bw) / SR);
          coef[j].b1 = 2 * rr * Math.cos((2 * Math.PI * curF[j]) / SR);
          coef[j].b2 = -rr * rr;
          coef[j].a0 = 1 - rr;
        }
      }
      const tt = k / n;
      const jitter = 1 + rasp * (r() - 0.5) * 0.6;
      const f0 = p.f0 * (1 + p.slope * tt) * jitter;
      const dt = f0 / SR;
      phase += dt;
      if (phase >= 1) phase -= 1;
      let saw = 2 * phase - 1;
      if (phase < dt) {
        const x = phase / dt;
        saw -= x + x - x * x - 1;
      } else if (phase > 1 - dt) {
        const x = (phase - 1) / dt;
        saw -= x * x + x + x + 1;
      }
      const sub = rasp > 0.02 && phase < 0.5 && (Math.floor(k * dt) & 1) ? 1 - rasp * 2 : 1;
      tiltY = p.tilt * tiltY + (1 - p.tilt) * saw * sub;
      const noise = r() * 2 - 1;
      const env = Math.min(1, k / (0.018 * SR)) * Math.min(1, (n - k) / (0.04 * SR));
      const src = (tiltY * 6 * p.voiced + noise * (p.breath + (1 - p.voiced) * 0.5)) * env;
      let y = 0;
      const gains = [1, 0.55, 0.28];
      for (let j = 0; j < 3; j++) {
        const yy = coef[j].a0 * src + coef[j].b1 * ys[j][0] + coef[j].b2 * ys[j][1];
        ys[j][1] = ys[j][0];
        ys[j][0] = yy;
        y += yy * gains[j];
      }
      if (p.onset && k < 0.03 * SR) y += noise * 0.05 * (1 - k / (0.03 * SR));
      out[n0 + k] += y * p.gain;
    }
  }

  // Room: phone band-limit + noise floor below 0, quieter floor above (28 dB at −2 … 58 dB at +4).
  let peak = 1e-6;
  for (let i = 0; i < total; i++) peak = Math.max(peak, Math.abs(out[i]));
  const norm = 0.32 / peak;
  const snrDb = clean >= 0 ? 40 + 4.5 * clean : 40 + 6 * clean;
  const noiseAmp = 0.32 * Math.pow(10, -snrDb / 20);
  const phone = clamp(-clean / 2, 0, 1);
  let hp = 0;
  let hpPrev = 0;
  let lp = 0;
  const hpA = Math.exp((-2 * Math.PI * 320) / SR);
  const lpA = Math.exp((-2 * Math.PI * 3400) / SR);
  const nr = rng(inp.seed + 1);
  for (let i = 0; i < total; i++) {
    let x = out[i] * norm;
    if (phone > 0) {
      hp = hpA * (hp + x - hpPrev);
      hpPrev = x;
      lp = lpA * lp + (1 - lpA) * hp;
      x = x * (1 - phone) + Math.tanh(lp * 2.2) * 0.6 * phone;
    }
    out[i] = Math.tanh((x + (nr() * 2 - 1) * noiseAmp) * 1.4) / 1.4;
  }
  return { pcm: out, words };
}

// Emotion / accent directions (emotion.gguf, accent-directions.gguf): port of servers/ggml/src/zonos3-emotion.cpp.
//
// Types: EmotionCalibration (per-speaker doses), Directions {space raw|lda|proj|attnres_source, sourceIndex, dim,
// inputDim, names, axes, named, axis, tiersJson, presetsJson, refBaseNorm, calibrated, calibration, lda*},
// EmotionSliders = [name, weight][] in request order. Functions throw Error with the C++ message where it returns false.
import { GGUF_TYPE, type Gguf } from '../gguf';

const f32 = Math.fround;

export class EmotionCalibration {
  bySpeaker = new Map<string, Map<string, number>>();
  defaults = new Map<string, number>();
  globalDefault = 1.0;

  private lookup(speaker: string, key: string): number | undefined {
    if (speaker) {
      const e = this.bySpeaker.get(speaker)?.get(key);
      if (e !== undefined) return e;
    }
    return this.defaults.get(key);
  }
  strength(speaker: string, emotion: string): number {
    return this.lookup(speaker, emotion) ?? this.globalDefault;
  }
  poleStrength(speaker: string, emotion: string, weight: number): number {
    if (weight < 0) {
      const s = this.lookup(speaker, emotion + '-');
      if (s !== undefined) return s;
    }
    return this.strength(speaker, emotion);
  }
  disabled(speaker: string, emotions: string[]): string[] {
    const out: string[] = [];
    for (const e of emotions) {
      if (this.strength(speaker, e) === 0) out.push(e);
      else if (this.lookup(speaker, e + '-') === 0) out.push(e + '-');
    }
    return out;
  }
  static parse(jsonText: string): EmotionCalibration {
    const out = new EmotionCalibration();
    const num = (v: unknown): number => {
      if (typeof v !== 'number') throw new Error('type must be number');
      return v;
    };
    try {
      const data: unknown = JSON.parse(jsonText);
      if (typeof data !== 'object' || data === null) throw new Error('calibration is not an object');
      const d = data as Record<string, unknown>;
      if (d.by_speaker !== undefined)
        for (const [sp, es] of Object.entries(d.by_speaker as Record<string, Record<string, unknown>>)) {
          const m = new Map<string, number>();
          for (const [k, v] of Object.entries(es)) m.set(k, num(v));
          out.bySpeaker.set(sp, m);
        }
      if (d.default !== undefined)
        for (const [k, v] of Object.entries(d.default as Record<string, unknown>)) out.defaults.set(k, num(v));
      out.globalDefault = d.global_default === undefined ? 1.0 : num(d.global_default);
    } catch (e) {
      throw new Error(`invalid emotion calibration: ${e instanceof Error ? e.message : String(e)}`);
    }
    return out;
  }
}

export class Directions {
  space = 'raw';
  sourceIndex = -1;
  dim = 0;
  inputDim = 0;
  names: string[] = [];
  axes: string[] = [];
  named: Float32Array[] = [];
  axis: Float32Array[] = [];
  tiersJson = '{}';
  presetsJson = '{}';
  refBaseNorm = NaN;
  calibrated = false;
  calibration = new EmotionCalibration();
  ldaDim = 0;
  ldaWeight: Float32Array = new Float32Array(0);
  ldaBias: Float32Array = new Float32Array(0);
  ldaPinv: Float32Array = new Float32Array(0);

  hidden(): boolean { return this.space === 'proj' || this.space === 'attnres_source'; }
  empty(): boolean { return this.names.length === 0 && this.axes.length === 0; }
  find(name: string): Float32Array | null {
    const i = this.names.indexOf(name);
    if (i >= 0) return this.named[i];
    const j = this.axes.indexOf(name);
    return j >= 0 ? this.axis[j] : null;
  }
  controls(): string[] { return [...this.names, ...this.axes]; }
}

export type EmotionSliders = [string, number][];

export function loadDirections(g: Gguf, path = 'directions.gguf'): Directions {
  const d = new Directions();
  let err: string | null = null;
  const fail = (m: string) => { err ??= `${path}: ${m}`; };
  const key = (name: string, type: number): boolean => {
    if (g.types.get(name)?.type !== type) { fail(`missing or invalid ${name}`); return false; }
    return true;
  };
  const str = (name: string): string => (key(name, GGUF_TYPE.STRING) ? (g.kv.get(name) as string) : '');
  const optionalStr = (name: string): string => (g.kv.has(name) ? str(name) : '');
  const strArray = (name: string, out: string[]) => {
    if (!g.kv.has(name) || !key(name, GGUF_TYPE.ARRAY)) return;
    if (g.types.get(name)!.arrType !== GGUF_TYPE.STRING) { fail(`${name} is not a string array`); return; }
    out.push(...(g.kv.get(name) as string[]));
  };
  const vec = (name: string, n: number): Float32Array => {
    const t = g.tensors.get(name);
    if (!t || t.length !== n) { fail(`missing or invalid tensor ${name} (${n} f32 values)`); return new Float32Array(0); }
    return t;
  };
  if (str('general.architecture') !== 'zonos3_directions') fail('not a zonos3_directions GGUF');
  d.space = str('zonos3_directions.space');
  if (err === null && !['raw', 'lda', 'proj', 'attnres_source'].includes(d.space)) fail(`unknown direction space ${d.space}`);
  if (key('zonos3_directions.source_index', GGUF_TYPE.INT32)) d.sourceIndex = g.kv.get('zonos3_directions.source_index') as number;
  if (key('zonos3_directions.dim', GGUF_TYPE.UINT32)) d.dim = g.kv.get('zonos3_directions.dim') as number;
  if (key('zonos3_directions.input_dim', GGUF_TYPE.UINT32)) d.inputDim = g.kv.get('zonos3_directions.input_dim') as number;
  strArray('zonos3_directions.names', d.names);
  strArray('zonos3_directions.axes', d.axes);
  const calibration = optionalStr('zonos3_directions.calibration');
  d.tiersJson = optionalStr('zonos3_directions.tiers') || '{}';
  d.presetsJson = optionalStr('zonos3_directions.presets') || '{}';
  if (key('zonos3_directions.ref_base_norm', GGUF_TYPE.FLOAT32)) d.refBaseNorm = g.kv.get('zonos3_directions.ref_base_norm') as number;
  if (err === null && (d.dim <= 0 || d.inputDim <= 0)) fail('invalid dim / input_dim');
  if (err === null && d.space === 'attnres_source' && d.sourceIndex < 0) fail('attnres_source directions need source_index >= 0');
  if (err === null && calibration) {
    try { d.calibration = EmotionCalibration.parse(calibration); d.calibrated = true; } catch (e) { fail((e as Error).message); }
  }
  for (let i = 0; err === null && i < d.names.length; i++) d.named.push(vec(`dir.named.${i}`, d.dim));
  for (let i = 0; err === null && i < d.axes.length; i++) d.axis.push(vec(`dir.axis.${i}`, d.dim));
  if (err === null && d.space === 'lda') {
    d.ldaDim = d.dim;
    d.ldaWeight = vec('dir.lda.weight', d.ldaDim * d.inputDim);
    d.ldaBias = vec('dir.lda.bias', d.ldaDim);
    d.ldaPinv = vec('dir.lda.pinv', d.inputDim * d.ldaDim);
  }
  if (err === null && d.space === 'raw' && d.inputDim !== d.dim) fail('raw directions need input_dim == dim');
  if (err !== null) throw new Error(err);
  return d;
}

// Python's repr of a list of str.
const pyList = (items: string[]) => '[' + items.map((s) => `'${s}'`).join(', ') + ']';

// torch.linalg.vector_norm of an fp32 vector (double accumulation, fp32 result).
function norm(v: Float32Array): number {
  let s = 0;
  for (const x of v) s += x * x;
  return f32(Math.sqrt(s));
}

function rescaleTo(v: Float32Array, target: number): void {
  const cur = norm(v);
  if (cur > f32(1e-8)) {
    const k = f32(target / cur);
    for (let i = 0; i < v.length; i++) v[i] = f32(v[i] * k);
  }
}

// Calibrated mode: every weight w becomes w * poleStrength(speaker, name, w); likewise valence and arousal.
export function calibrate(d: Directions, speaker: string, sliders: EmotionSliders, va: { valence: number; arousal: number }): void {
  if (!d.calibrated) return;
  for (const s of sliders) s[1] = s[1] * d.calibration.poleStrength(speaker, s[0], s[1]);
  va.valence = va.valence * d.calibration.poleStrength(speaker, 'valence', va.valence);
  va.arousal = va.arousal * d.calibration.poleStrength(speaker, 'arousal', va.arousal);
}

// _combine_delta. Unknown names throw when `strict`, else are skipped.
export function combine(d: Directions, sliders: EmotionSliders, valence: number, arousal: number,
                        strict: boolean): { delta: Float32Array; requested: boolean } {
  const delta = new Float32Array(d.dim);
  let requested = false;
  const accumulate = (weight: number, v: Float32Array) => {
    const w = f32(weight);
    for (let i = 0; i < delta.length; i++) delta[i] += f32(w * v[i]);
    requested = true;
  };
  for (const [name, w] of sliders) {
    if (w === 0) continue;
    const v = d.find(name);
    if (!v) {
      const msg = `Unknown emotion direction '${name}'. Available: ${pyList(d.controls())}.`;
      if (strict) throw new Error(msg);
      console.warn(`zonos3: ${msg}`);
      continue;
    }
    accumulate(w, v);
  }
  for (const [name, w] of [['valence', valence], ['arousal', arousal]] as const) {
    if (w === 0) continue;
    const i = d.axes.indexOf(name);
    if (i < 0) {
      if (strict) throw new Error(`Emotion axis '${name}' is not available.`);
      continue;
    }
    accumulate(w, d.axis[i]);
  }
  return { delta, requested };
}

// apply_emotion: the embedding moved by raw/lda directions (a copy of `base` for hidden spaces / nothing requested).
export function applyEmotion(d: Directions, base: Float32Array, sliders: EmotionSliders, valence: number, arousal: number,
                             strength: number, preserveNorm: boolean, strict: boolean): Float32Array {
  const out = Float32Array.from(base);
  if (d.empty() || d.hidden()) return out;
  if (base.length !== d.inputDim)
    throw new Error(`Speaker embedding dim ${base.length} does not match emotion directions input dim ${d.inputDim}.`);
  const { delta, requested } = combine(d, sliders, valence, arousal, strict);
  if (!requested || strength === 0) return out;
  const s = f32(strength);
  if (d.space === 'lda') {
    const L = d.ldaDim, I = d.inputDim;
    const ldaBase = new Float32Array(L), ldaNew = new Float32Array(L);
    for (let r = 0; r < L; r++) {
      let acc = 0;
      for (let c = 0; c < I; c++) acc += d.ldaWeight[r * I + c] * base[c];
      ldaBase[r] = f32(acc) + d.ldaBias[r];
      ldaNew[r] = ldaBase[r] + f32(s * delta[r]);
    }
    if (preserveNorm) rescaleTo(ldaNew, norm(ldaBase));
    for (let r = 0; r < L; r++) ldaNew[r] -= d.ldaBias[r];
    for (let c = 0; c < I; c++) {
      let acc = 0;
      for (let r = 0; r < L; r++) acc += d.ldaPinv[c * L + r] * ldaNew[r];
      out[c] = acc;
    }
    return out;
  }
  for (let i = 0; i < out.length; i++) out[i] = base[i] + f32(s * delta[i]);
  if (preserveNorm) rescaleTo(out, norm(base));
  return out;
}

// emotion_hidden_delta: strength * Σ weight·direction for proj / attnres_source; null = none.
export function hiddenDelta(d: Directions, sliders: EmotionSliders, valence: number, arousal: number, strength: number,
                            strict: boolean): Float32Array | null {
  if (d.empty() || !d.hidden()) return null;
  const { delta, requested } = combine(d, sliders, valence, arousal, strict);
  if (!requested || strength === 0) return null;
  const s = f32(strength);
  const out = new Float32Array(delta.length);
  for (let i = 0; i < delta.length; i++) out[i] = s * delta[i];
  return out;
}

// sgl _apply_steering_accent direction part: a raw set moves `embedding` in place (norm kept), a hidden set returns
// strength * dose * direction (null otherwise).
export function accentSteering(d: Directions, accent: string, strength: number, speaker: string, useCalibration: boolean,
                               embedding: Float32Array | null): Float32Array | null {
  const v = d.find(accent);
  if (!v) throw new Error(`unknown accent direction '${accent}'`);
  const dose = d.calibrated && useCalibration ? d.calibration.strength(speaker, accent) : 1.0;
  if (strength === 0 || dose === 0) return null;
  if (d.space === 'raw') {
    if (!embedding) throw new Error(`Accent '${accent}' is a speaker-embedding direction and needs speaker_mode=embedding.`);
    embedding.set(applyEmotion(d, embedding, [[accent, strength * dose]], 0, 0, 1, true, true));
    return null;
  }
  const k = f32(strength * dose);
  const hidden = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) hidden[i] = k * v[i];
  return hidden;
}

// zonos3_steer_add_delta: adds `delta` to the slot of `source` in a [sources.length * hiddenSize] layout (null starts
// at zero). Returns the layout, or null when `source` is not among `sources` or the sizes mismatch.
export function addHiddenDelta(sources: number[], hiddenSize: number, source: number, delta: Float32Array,
                               layout: Float32Array | null): Float32Array | null {
  const k = sources.indexOf(source);
  if (k < 0 || delta.length !== hiddenSize) return null;
  const out = layout ?? new Float32Array(sources.length * hiddenSize);
  if (out.length !== sources.length * hiddenSize) return null;
  for (let i = 0; i < hiddenSize; i++) out[k * hiddenSize + i] += delta[i];
  return out;
}

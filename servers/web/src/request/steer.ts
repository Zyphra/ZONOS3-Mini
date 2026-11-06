// Steering directions (steer.gguf): port of servers/ggml/src/zonos3-steer.cpp. Raw-space controls move the speaker
// embedding, hidden-space controls add to Full AttnRes bank sources of generated frames.
//
// Types: Calibration (a set's calibration.json: per-speaker doses), SteerDirection {id, set, name, axis, hidden,
// source, vec}, SteerSet {name, hidden, source, calibration}, RateExtension, SteerBundle {name, speakerDim, hiddenSize,
// sets, dirs, timing (id -> [dose, factor][], id-sorted), timingJson, uiJson, index, sources, rateExtension},
// ControlSpan {start, end, controls}, SteerSpan {start, end, ramp, delta, charStart, charEnd}, Steering (resolved
// request: hidden [K*D] | null, spans, deltaAt). Errors are thrown as Error with the C++ message. Controls iterate in
// key order like std::map.
import { GGUF_TYPE, type Gguf } from '../gguf';

/** Why steering fields fail on a server without a bundle; each site adds its own suffix. */
export const NO_STEERING = 'This server has no steering bundle (place steer.gguf next to the backbone or pass --steer)';

export const STEER_RAMP_FRAMES = 3;
export const STEER_FRAMES_PER_CHAR = 1.0;

/** A set's calibration.json (zonos3.tts.directions.Calibration): doses keyed by direction name, `<name>-` = the
 *  negative pole's own dose. Exposed in /v2/controls and speakers' disabled_controls; never applied to controls. */
export class Calibration {
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
  strength(speaker: string, name: string): number {
    return this.lookup(speaker, name) ?? this.globalDefault;
  }
  /** The names whose dose is 0 for `speaker`; a negative pole disabled on its own is listed as `<name>-`. */
  disabled(speaker: string, names: string[]): string[] {
    const out: string[] = [];
    for (const n of names) {
      if (this.strength(speaker, n) === 0) out.push(n);
      else if (this.lookup(speaker, n + '-') === 0) out.push(n + '-');
    }
    return out;
  }
  /** The /v2/controls ControlCalibration object. */
  document(): unknown {
    const by: Record<string, Record<string, number>> = {};
    for (const [sp, m] of this.bySpeaker) by[sp] = Object.fromEntries(m);
    return { default: Object.fromEntries(this.defaults), by_speaker: by, global_default: this.globalDefault };
  }
  static parse(jsonText: string): Calibration {
    const out = new Calibration();
    const isNumObj = (v: unknown): v is Record<string, number> =>
      isObj(v) && Object.values(v).every((x) => typeof x === 'number');
    const data: unknown = JSON.parse(jsonText);
    if (!isObj(data)) throw new Error('calibration is not an object');
    if (data.by_speaker !== undefined) {
      if (!isObj(data.by_speaker)) throw new Error('by_speaker is not an object');
      for (const [sp, doses] of Object.entries(data.by_speaker)) {
        if (!isNumObj(doses)) throw new Error(`by_speaker.${sp} is not a {name: dose} object`);
        out.bySpeaker.set(sp, new Map(Object.entries(doses)));
      }
    }
    if (data.default !== undefined) {
      if (!isNumObj(data.default)) throw new Error('default is not a {name: dose} object');
      out.defaults = new Map(Object.entries(data.default));
    }
    if (data.global_default !== undefined) {
      if (typeof data.global_default !== 'number') throw new Error('global_default is not a number');
      out.globalDefault = data.global_default;
    }
    return out;
  }
}

export interface SteerDirection { id: string; set: string; name: string; axis: boolean; hidden: boolean; source: number; vec: Float32Array }
export interface SteerSet { name: string; hidden: boolean; source: number; calibration: Calibration | null }
export interface RateExtension {
  direction: string; rowEdges: number[]; below: number[]; above: number[];
  doses: Map<string, Map<number, number>>; referenceSides: string[];
}
export interface SteerBundle {
  name: string; speakerDim: number; hiddenSize: number;
  sets: SteerSet[]; dirs: SteerDirection[];
  timing: Map<string, [number, number][]>;
  timingJson: string; uiJson: string;
  index: Map<string, number>;
  sources: number[];
  rateExtension: RateExtension | null;
}
export interface SteerModel { nEmbd: number; nLayer: number; speakerDim: number }
export interface ControlSpan { start: number; end: number; controls: Record<string, number> }
export interface SteerSpan { start: number; end: number; ramp: number; delta: Float32Array; charStart: number; charEnd: number }

const f32 = Math.fround;
const sortedKeys = (o: Record<string, number> | Map<string, unknown>): string[] =>
  (o instanceof Map ? [...o.keys()] : Object.keys(o)).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

export class Steering {
  sources: number[] = [];
  hiddenSize = 0;
  hidden: Float32Array | null = null;   // constant delta [sources.length * hiddenSize]; null = none
  spans: SteerSpan[] = [];               // sorted, non-overlapping
  hiddenActive(): boolean { return this.hidden !== null || this.spans.length > 0; }
  // out.v = the delta at `frames`, or null when it is zero
  deltaAt(frames: number, out: { v: Float32Array | null }): void {
    if (!this.hiddenActive()) { out.v = null; return; }
    const n = this.sources.length * this.hiddenSize;
    const v = this.hidden ? Float32Array.from(this.hidden) : new Float32Array(n);
    let any = this.hidden !== null;
    const f = f32(frames);
    for (const s of this.spans) {
      const r = f32(Math.max(s.ramp, 1));
      const up = f32(f32(f - f32(s.start - r)) / r), down = f32(f32(f32(s.end + r) - f) / r);
      const w = Math.min(Math.max(Math.min(up, down), 0), 1);
      if (!(w > 0)) continue;
      any = true;
      for (let i = 0; i < n; i++) v[i] += f32(w * s.delta[i]);
    }
    out.v = any ? v : null;
  }
}

export function steerIndex(b: SteerBundle): void {
  b.index = new Map();
  b.dirs.forEach((d, i) => b.index.set(d.id, i));
  const src = new Set<number>();
  for (const d of b.dirs) if (d.hidden) src.add(d.source);
  b.sources = [...src].sort((x, y) => x - y);
}

// numpy.interp: clamped linear interpolation over x-sorted points.
function interp(t: [number, number][], x: number): number {
  if (t.length === 0) return 1.0;
  if (x <= t[0][0]) return t[0][1];
  if (x >= t[t.length - 1][0]) return t[t.length - 1][1];
  let hi = 1;
  while (t[hi][0] < x) hi++;
  const a = t[hi - 1], b = t[hi];
  if (b[0] === a[0]) return b[1];
  return a[1] + (b[1] - a[1]) * ((x - a[0]) / (b[0] - a[0]));
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
function numArray(v: unknown, what: string): number[] {
  if (!Array.isArray(v) || !v.every((x) => typeof x === 'number')) throw new Error(`${what} is not a number array`);
  return v;
}

export function loadSteer(g: Gguf, model: SteerModel, path = 'steer.gguf'): SteerBundle {
  const invalid = (why: string) =>
    new Error(`'${path}' is not a valid zonos3 steering GGUF (${why}); reconvert with tools/convert_gguf.py steer`);
  const typeOf = (k: string) => g.types.get(k);
  const getStr = (k: string): string | null => (typeOf(k)?.type === GGUF_TYPE.STRING ? (g.kv.get(k) as string) : null);
  const getU32 = (k: string): number | null => (typeOf(k)?.type === GGUF_TYPE.UINT32 ? (g.kv.get(k) as number) : null);
  const getStrs = (k: string): string[] | null => {
    const t = typeOf(k);
    return t?.type === GGUF_TYPE.ARRAY && t.arrType === GGUF_TYPE.STRING ? (g.kv.get(k) as string[]) : null;
  };
  if (getStr('general.architecture') !== 'zonos3_steer') throw invalid('general.architecture is not zonos3_steer');
  const st = typeOf('zonos3_steer.set.sources');
  if (st?.type !== GGUF_TYPE.ARRAY || st.arrType !== GGUF_TYPE.INT32) throw invalid('missing zonos3_steer.set.sources');
  const setSources = g.kv.get('zonos3_steer.set.sources') as number[];
  const name = getStr('zonos3_steer.name'), speakerDim = getU32('zonos3_steer.speaker_dim');
  const hiddenSize = getU32('zonos3_steer.hidden_size');
  const setNames = getStrs('zonos3_steer.set.names'), setSpaces = getStrs('zonos3_steer.set.spaces');
  const setCalibrations = getStrs('zonos3_steer.set.calibrations');
  const ids = getStrs('zonos3_steer.direction.ids'), kinds = getStrs('zonos3_steer.direction.kinds');
  const timingJson = getStr('zonos3_steer.timing'), uiJson = getStr('zonos3_steer.ui');
  if (name === null || speakerDim === null || hiddenSize === null || !setNames || !setSpaces || !setCalibrations ||
      !ids || !kinds || timingJson === null || uiJson === null) throw invalid('missing or mistyped zonos3_steer.* metadata');
  if (setNames.length === 0 || setNames.length !== setSpaces.length || setNames.length !== setSources.length ||
      setNames.length !== setCalibrations.length || ids.length === 0 || ids.length !== kinds.length ||
      g.tensors.size !== ids.length)
    throw invalid('inconsistent metadata array sizes');

  const b: SteerBundle = {
    name, speakerDim, hiddenSize, sets: [], dirs: [], timing: new Map(), timingJson, uiJson,
    index: new Map(), sources: [], rateExtension: null,
  };
  const setIndex = new Map<string, number>();
  let anyRaw = false, anyHidden = false;
  for (let i = 0; i < setNames.length; i++) {
    const s: SteerSet = { name: setNames[i], hidden: false, source: -1, calibration: null };
    if (s.name === '' || s.name.includes('.') || setIndex.has(s.name)) throw invalid(`bad set name '${s.name}'`);
    if (setSpaces[i] === 'attnres_source') {
      s.hidden = true;
      s.source = setSources[i];
      if (s.source < 0) throw invalid(`set '${s.name}' has a negative source`);
      anyHidden = true;
    } else if (setSpaces[i] === 'raw') {
      anyRaw = true;
    } else {
      throw invalid(`set '${s.name}' has unknown space '${setSpaces[i]}'`);
    }
    if (setCalibrations[i]) {
      try {
        s.calibration = Calibration.parse(setCalibrations[i]);
      } catch (e) {
        throw invalid(`set '${s.name}' has a bad calibration: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    setIndex.set(s.name, b.sets.length);
    b.sets.push(s);
  }
  if ((anyRaw && speakerDim === 0) || (anyHidden && hiddenSize === 0)) throw invalid('a set space has no dimension');

  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    const dot = id.indexOf('.');
    if (dot < 0 || dot + 1 >= id.length) throw invalid(`bad direction id '${id}'`);
    const set = id.slice(0, dot), dname = id.slice(dot + 1);
    const si = setIndex.get(set);
    if (si === undefined) throw invalid(`direction '${id}' names an unknown set`);
    if (kinds[i] !== 'named' && kinds[i] !== 'axis') throw invalid(`direction '${id}' has unknown kind`);
    const hidden = b.sets[si].hidden;
    const tname = `steer.dir.${i}`;
    const t = g.tensors.get(tname), shape = g.shapes.get(tname);
    const dim = hidden ? hiddenSize : speakerDim;
    if (!t || !shape || shape.length !== 1 || shape[0] !== dim)
      throw invalid(`tensor ${tname} is missing or is not F32 [${dim}]`);
    for (const v of t) if (!Number.isFinite(v)) throw invalid(`tensor ${tname} has non-finite values`);
    b.dirs.push({ id, set, name: dname, axis: kinds[i] === 'axis', hidden, source: b.sets[si].source, vec: t });
  }
  steerIndex(b);
  if (b.index.size !== b.dirs.length) throw invalid('duplicate direction ids');

  try {
    const tj: unknown = JSON.parse(timingJson);
    if (!isObj(tj)) throw new Error('timing is not an object');
    const timing = new Map<string, [number, number][]>();
    for (const [k, v] of Object.entries(tj)) {
      if (!b.index.has(k)) throw invalid(`timing for unknown direction '${k}'`);
      if (!Array.isArray(v) || v.length === 0) throw new Error('empty timing table');
      const table: [number, number][] = [];
      for (const p of v) {
        if (!Array.isArray(p) || p.length !== 2 || typeof p[0] !== 'number' || typeof p[1] !== 'number')
          throw new Error('bad timing pair');
        table.push([p[0], p[1]]);
      }
      table.sort((l, r) => l[0] - r[0]);  // Array.prototype.sort is stable
      timing.set(k, table);
    }
    for (const k of sortedKeys(timing)) b.timing.set(k, timing.get(k)!);
    if (!isObj(JSON.parse(uiJson))) throw new Error('ui is not an object');
  } catch (e) {
    if (e instanceof Error && e.message.startsWith(`'${path}' is not a valid`)) throw e;
    throw invalid(`bad timing/ui JSON: ${e instanceof Error ? e.message : String(e)}`);
  }

  const rateJson = getStr('zonos3_steer.rate_extension');
  if (rateJson) {
    let ext: RateExtension | null = null;
    try {
      const r: unknown = JSON.parse(rateJson);
      if (isObj(r) && Object.keys(r).length > 0) {
        if (typeof r.direction !== 'string') throw new Error('direction is not a string');
        const direction = r.direction;
        const d = b.index.get(direction);
        if (d === undefined || !b.dirs[d].hidden)
          throw invalid(`rate_extension direction '${direction}' is not a hidden direction of the bundle`);
        const doses = new Map<string, Map<number, number>>();
        if (!isObj(r.languages)) throw new Error('languages is not an object');
        for (const [lang, bd] of Object.entries(r.languages)) {
          if (!isObj(bd)) throw new Error('bad language doses');
          const m = new Map<number, number>();
          for (const [k, v] of Object.entries(bd)) {
            if (typeof v !== 'number') throw new Error('bad dose');
            m.set(parseInt(k, 10), v);
          }
          doses.set(lang, m);
        }
        const sides = r.reference_sides;
        ext = {
          direction, rowEdges: numArray(r.row_edges, 'row_edges'), below: numArray(r.below, 'below'),
          above: numArray(r.above, 'above'), doses,
          referenceSides: Array.isArray(sides) ? sides.map(String) : ['slower', 'faster'],
        };
      }
    } catch (e) {
      if (e instanceof Error && e.message.startsWith(`'${path}' is not a valid`)) throw e;
      throw invalid(`bad rate_extension JSON: ${e instanceof Error ? e.message : String(e)}`);
    }
    b.rateExtension = ext;
  }

  if (b.hiddenSize && b.hiddenSize !== model.nEmbd)
    throw new Error(`steering hidden directions have ${b.hiddenSize} values; the backbone hidden size is ${model.nEmbd}`);
  if (b.speakerDim && b.speakerDim !== model.speakerDim)
    throw new Error(`steering speaker directions have ${b.speakerDim} values; the backbone speaker embedding has ${model.speakerDim}`);
  for (const s of b.sources)
    if (s > 2 * model.nLayer)
      throw new Error(`steering source ${s} is outside the backbone's Full AttnRes bank (0..${2 * model.nLayer})`);
  return b;
}

// The JSON object of zonos3_steer_controls_json (same key order).
export function controlsDocument(b: SteerBundle, speakerDim: number): unknown {
  const sets: Record<string, unknown> = {};
  for (const s of b.sets) {
    const e: Record<string, unknown> = {
      space: s.hidden ? 'attnres_source' : 'raw',
      modes: s.hidden ? ['embedding', 'reference'] : ['embedding'],
    };
    if (s.hidden) e.source_index = s.source;
    if (s.calibration) e.calibration = s.calibration.document();
    sets[s.name] = e;
  }
  const dirs: Record<string, unknown> = {};
  for (const d of b.dirs) {
    let n2 = 0;
    for (const v of d.vec) n2 += v * v;
    const e: Record<string, unknown> = {
      set: d.set, space: d.hidden ? 'attnres_source' : 'raw', kind: d.axis ? 'axis' : 'named', norm: Math.sqrt(n2),
    };
    if (!d.hidden) e.vector = Array.from(d.vec);
    dirs[d.id] = e;
  }
  return {
    bundle: b.name, embedding_dim: speakerDim, sets, directions: dirs,
    timing: JSON.parse(b.timingJson), ui: JSON.parse(b.uiJson),
  };
}

/** steering.disabled_controls: per calibrated set, in direction order, the control ids (`<set>.<name>` or
 *  `<set>.<name>-`) whose calibrated dose for `speakerId` is 0; [] without a bundle. */
export function disabledControls(b: SteerBundle | null, speakerId: string): string[] {
  if (!b) return [];
  const out: string[] = [];
  for (const s of b.sets) {
    if (!s.calibration) continue;
    const names = b.dirs.filter((d) => d.set === s.name).map((d) => d.name);
    for (const n of s.calibration.disabled(speakerId, names)) out.push(`${s.name}.${n}`);
  }
  return out;
}

export function steerDurationFactor(b: SteerBundle, doses: Map<string, number>): number {
  let factor = 1.0;
  for (const [id, table] of b.timing) {
    const v = doses.get(id);
    if (v === undefined || v === 0) continue;
    factor *= interp(table, v);
  }
  return factor;
}

// Unicode code points.
export function textChars(s: string): number {
  let n = 0;
  for (const _ of s) n++;
  return n;
}

// numpy round / std::nearbyint: halves to even.
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// zonos3_steer_resolve; `embedding` (embedding-mode speaker or null) is moved in place by raw controls.
export function resolveControls(b: SteerBundle, controls: Record<string, number>, spans: ControlSpan[],
                                textChars: number, rate: number, embedding: Float32Array | null): Steering {
  const out = new Steering();
  out.sources = b.sources.slice();
  out.hiddenSize = b.hiddenSize;

  const doses = new Map<string, number>();
  for (const id of sortedKeys(controls)) if (controls[id] !== 0) doses.set(id, controls[id]);
  const unknown = (id: string) => new Error(`Unknown control '${id}'; see /v2/controls.`);
  for (const id of doses.keys()) if (!b.index.has(id)) throw unknown(id);
  for (const s of spans) for (const id of sortedKeys(s.controls)) if (!b.index.has(id)) throw unknown(id);
  for (const [id, v] of doses) if (!Number.isFinite(v)) throw new Error(`Control '${id}' must be finite.`);

  const K = b.sources.length, D = b.hiddenSize;
  const slotOf = (source: number) => { let i = 0; while (i < K && b.sources[i] < source) i++; return i; };
  const hiddenVector = (ds: Map<string, number>): Float32Array | null => {
    let v: Float32Array | null = null;
    for (const [id, dose] of ds) {
      if (dose === 0) continue;
      const d = b.dirs[b.index.get(id)!];
      if (!d.hidden) continue;
      v ??= new Float32Array(K * D);
      const o = slotOf(d.source) * D, f = f32(dose);
      for (let i = 0; i < D; i++) v[o + i] += f32(f * d.vec[i]);
    }
    return v;
  };

  const raw: string[] = [];
  for (const id of doses.keys()) if (!b.dirs[b.index.get(id)!].hidden) raw.push(id);
  if (raw.length) {
    if (!embedding) throw new Error(`Controls ${raw.join(', ')} move the speaker embedding and need speaker_mode=embedding.`);
    const base = embedding;
    const delta = new Float32Array(base.length);
    for (const id of raw) {
      const d = b.dirs[b.index.get(id)!];
      if (d.vec.length !== base.length)
        throw new Error(`Control '${id}' has dim ${d.vec.length}, the speaker embedding has ${base.length}.`);
      const f = f32(doses.get(id)!);
      for (let i = 0; i < base.length; i++) delta[i] += f32(f * d.vec[i]);
    }
    let nb = 0, nm = 0;
    for (let i = 0; i < base.length; i++) {
      delta[i] += base[i];
      nb += base[i] * base[i];
      nm += delta[i] * delta[i];
    }
    const scale = f32(Math.sqrt(nb) / Math.max(Math.sqrt(nm), 1e-8));
    for (let i = 0; i < base.length; i++) base[i] = delta[i] * scale;
  }

  out.hidden = hiddenVector(doses);
  if (spans.length === 0) return out;
  const r0 = (rate > 0 ? rate : STEER_FRAMES_PER_CHAR) * steerDurationFactor(b, doses);
  if (!(r0 > 0)) throw new Error('control_span_rate must be positive.');

  const cleaned: { start: number; end: number; doses: Map<string, number> }[] = [];
  for (const s of spans) {
    const c = { start: s.start, end: s.end, doses: new Map<string, number>() };
    for (const id of sortedKeys(s.controls)) if (s.controls[id] !== 0) c.doses.set(id, s.controls[id]);
    if (!(0 <= s.start && s.start < s.end && s.end <= textChars))
      throw new Error(`control span [${s.start}, ${s.end}) is outside the text (0..${textChars}).`);
    for (const [id, v] of c.doses) {
      if (!Number.isFinite(v)) throw new Error(`Control '${id}' must be finite.`);
      if (!b.dirs[b.index.get(id)!].hidden)
        throw new Error(`Control '${id}' moves the speaker embedding and cannot vary inside a request; ` +
          'only hidden-space controls take spans.');
    }
    cleaned.push(c);
  }
  cleaned.sort((l, r) => (l.start !== r.start ? l.start - r.start : l.end - r.end));
  for (let i = 1; i < cleaned.length; i++)
    if (cleaned[i].start < cleaned[i - 1].end)
      throw new Error(`control spans [${cleaned[i - 1].start}, ${cleaned[i - 1].end}) and [${cleaned[i].start}, ...) overlap.`);
  const frameAt = new Float64Array(textChars + 1);
  const perChar = new Float64Array(textChars).fill(r0);
  for (const c of cleaned) {
    const r = r0 * steerDurationFactor(b, c.doses);
    for (let i = c.start; i < c.end; i++) perChar[i] = r;
  }
  for (let i = 0; i < textChars; i++) frameAt[i + 1] = frameAt[i] + perChar[i];
  for (const c of cleaned) {
    const delta = hiddenVector(c.doses);
    if (!delta) continue;
    out.spans.push({
      start: roundHalfEven(frameAt[c.start]), end: roundHalfEven(frameAt[c.end]), ramp: STEER_RAMP_FRAMES,
      delta, charStart: c.start, charEnd: c.end,
    });
  }
  return out;
}

// LACI (Localized Attention-Constrained Inference): port of servers/ggml/src/zonos3-laci.cpp (model/src/zonos3/tts/
// laci.py). Duration targets (PaceMonitor / DurationTarget) serve plugin duration-target takes only and are not ported: a session here
// never carries a duration, so every pace path of the C++ session is absent.
//
// Types: LaciConfig, Failure, Head {layer, head}, HeadWindow {head, lo, hi}, Windows, ProbeControl {textLen, mask,
// stop}, Continuation, FailureRecord, LaciRecord (toJson = LaciSession.record() dict), RateBuckets, SessionParams,
// RowRange {begin, end}. Rows are stream indices: prefill rows 0..P-1, generated rows follow.
import type { SpanFollower } from './spanFollow';

export interface LaciConfig {
  binTokens: number; coverFrac: number; skipFrac: number; stallFactor: number; minStallFrames: number;
  tailFrames: number; rollbackMarginFrames: number; maxRetries: number; paceWindowBins: number; warmupBins: number;
  /** Once the text is done and stopBoostDelay more rows were sampled, the sampler adds stopBoost to the stop id's logit
   *  in every codebook before the sampling filters (laci.py LaciConfig); 0 turns it off. */
  stopBoost: number; stopBoostDelay: number;
}
export const defaultLaciConfig = (): LaciConfig => ({
  binTokens: 8, coverFrac: 0.5, skipFrac: 0.15, stallFactor: 3.0, minStallFrames: 45, tailFrames: 15,
  rollbackMarginFrames: 8, maxRetries: 8, paceWindowBins: 16, warmupBins: 2, stopBoost: 3.0, stopBoostDelay: 6,
});

export interface Failure { kind: string; detectRow: number; onsetRow: number; rollbackRow: number; bin: number; frontier: number }

export const COMMIT_UNBOUNDED = Number.MAX_SAFE_INTEGER;

function median(v: number[]): number {
  const s = v.slice().sort((a, b) => a - b);
  const n = s.length, i = n >> 1;
  return n % 2 === 1 ? s[i] : (s[i - 1] + s[i]) / 2;
}

function lowerBound(a: number[], x: number): number {
  let i = 0;
  while (i < a.length && a[i] < x) i++;
  return i;
}

export class Detector {
  nb: number;
  private obsRows: number[] = [];
  private obsPos: number[] = [];
  private bumps: number[] = [];
  private countsArr: number[] = [];
  private firstRow: (number | null)[] = [];
  private frontierV = 0;
  private lastAdvance: number | null = null;
  private done: number | null = null;
  private readRow: number | null = null;   // first row the reading was on the last text token (EOS)

  constructor(public textLen: number, public prior: number, public cfg: LaciConfig, public minRollbackRow: number) {
    if (textLen < 1) throw new Error(`text_len must be >= 1, got ${textLen}`);
    this.nb = Math.floor((textLen + cfg.binTokens - 1) / cfg.binTokens);
    this.reset();
  }

  frontier(): number { return this.frontierV; }
  counts(): readonly number[] { return this.countsArr; }
  lastAdvanceRow(): number | null { return this.lastAdvance; }
  doneRow(): number | null { return this.done; }

  private reset(): void {
    this.countsArr = new Array<number>(this.nb).fill(0);
    this.firstRow = new Array<number | null>(this.nb).fill(null);
    this.frontierV = 0;
    this.lastAdvance = null;
    this.done = null;
    this.readRow = null;
  }
  private tokens(b: number): number { return Math.min(this.cfg.binTokens, this.textLen - b * this.cfg.binTokens); }

  paceTok(): number {
    const lo = Math.max(0, this.frontierV - this.cfg.paceWindowBins);
    const rates: number[] = [];
    for (let x = lo; x < this.frontierV; x++) rates.push(this.countsArr[x] / this.tokens(x));
    if (rates.length < this.cfg.warmupBins) return this.prior;
    return median(rates);
  }
  private share(b: number): number { return this.paceTok() * this.tokens(b); }

  lastPosBefore(row: number): number {
    const i = lowerBound(this.obsRows, row);
    return i ? this.obsPos[i - 1] : 0;
  }
  stopRow(): number | null { return this.done === null ? null : this.done + this.cfg.tailFrames; }

  commitRow(nextRow: number): number {
    if (this.done !== null) return COMMIT_UNBOUNDED;
    let onset = Math.min(this.lastAdvance ?? nextRow, nextRow);
    for (let b = this.frontierV + 2; b < this.nb; b++) {
      const f = this.firstRow[b];
      if (f !== null) onset = Math.min(onset, f);
    }
    return this.rollback(onset);
  }
  private rollback(onset: number): number { return Math.max(onset - this.cfg.rollbackMarginFrames, this.minRollbackRow); }

  observe(row: number, pos: number): Failure | null {
    if (!(0 <= pos && pos < this.textLen)) throw new Error(`pos ${pos} outside text span [0, ${this.textLen})`);
    if (this.obsRows.length && row <= this.obsRows[this.obsRows.length - 1])
      throw new Error(`rows must increase: got ${row} after ${this.obsRows[this.obsRows.length - 1]}`);
    this.obsRows.push(row);
    this.obsPos.push(pos);
    return this.step(row, pos);
  }

  private step(row: number, pos: number): Failure | null {
    const cfg = this.cfg;
    const b = Math.floor(pos / cfg.binTokens);
    this.countsArr[b] += 1;
    if (this.firstRow[b] === null) this.firstRow[b] = row;
    if (this.lastAdvance === null) this.lastAdvance = row;

    if (b > this.frontierV && this.countsArr[b] >= cfg.coverFrac * this.share(b)) {
      for (let x = this.frontierV + 1; x < b; x++) {
        if (this.countsArr[x] < cfg.skipFrac * this.share(x)) {
          let onset: number | null = null;
          for (let y = x + 1; y < this.nb; y++) {
            const f = this.firstRow[y];
            if (f !== null && (onset === null || f < onset)) onset = f;
          }
          return { kind: 'skip', detectRow: row, onsetRow: onset!, rollbackRow: this.rollback(onset!), bin: x, frontier: b };
        }
      }
      this.frontierV = b;
      this.lastAdvance = row;
    }

    if (this.done === null) {
      const limit = Math.max(cfg.stallFactor * this.paceTok() * cfg.binTokens, cfg.minStallFrames);
      if (row - this.lastAdvance > limit) {
        const onset = this.lastAdvance;
        return {
          kind: 'hallucination', detectRow: row, onsetRow: onset, rollbackRow: this.rollback(onset),
          bin: this.frontierV + 1, frontier: this.frontierV,
        };
      }
    }

    // text done: the last bin read for its share, and the reading has reached the last text token (with two bins the
    // share alone, a pace prior, can call the text read a second before the end is spoken)
    if (this.readRow === null && pos >= this.textLen - 1) this.readRow = row;
    if (this.done === null && this.readRow !== null && this.frontierV >= this.nb - 2
        && this.countsArr[this.nb - 1] >= this.share(this.nb - 1))
      this.done = row;
    return null;
  }

  onEos(row: number): Failure | null {
    if (this.frontierV < this.nb - 2)
      return { kind: 'truncation', detectRow: row, onsetRow: row, rollbackRow: this.rollback(row), bin: this.frontierV + 1, frontier: this.frontierV };
    return null;
  }

  truncate(row: number): void {
    const keep = lowerBound(this.obsRows, row);
    const rows = this.obsRows.slice(0, keep), poss = this.obsPos.slice(0, keep);
    const bumps = this.bumps.filter((r) => r < row);
    this.obsRows = [];
    this.obsPos = [];
    this.reset();
    let pending = 0;
    for (let k = 0; k < rows.length; k++) {
      const r = rows[k];
      while (pending < bumps.length && r > bumps[pending]) {
        if (this.lastAdvance !== null) this.lastAdvance = Math.max(this.lastAdvance, bumps[pending]);
        pending++;
      }
      this.obsRows.push(r);
      this.obsPos.push(poss[k]);
      if (this.step(r, poss[k])) throw new Error('LACI detector: replay raised a failure');
    }
    if (this.lastAdvance !== null) this.lastAdvance = Math.max(this.lastAdvance, row);
    bumps.push(row);
    this.bumps = bumps;
  }
}

// ACI dynamic-programming attention centre, held monotone and limited to +1 per update.
export class AciCenter {
  c0: number;
  center: number;
  private d: number[] = [0];
  constructor(c0: number, public textLen: number) {
    this.c0 = Math.min(Math.max(c0, 0), textLen - 1);
    this.center = this.c0;
  }
  update(m: number): number {
    const d = this.d, c0 = this.c0;
    if (c0 + d.length < this.textLen) d.push(Infinity);
    for (let i = d.length - 1; i > 0; i--) {
      const e = m - (c0 + i);
      d[i] = Math.min(d[i], d[i - 1]) + e * e;
    }
    const e0 = m - c0;
    d[0] += e0 * e0;
    let best = 0;
    for (let i = 1; i < d.length; i++) if (d[i] < d[best]) best = i;
    this.center = Math.min(Math.max(c0 + best, this.center), this.center + 1);
    return this.center;
  }
}

export interface Head { layer: number; head: number }
export interface HeadWindow { head: Head; lo: number; hi: number }
export type Windows = HeadWindow[];

const intOf = (v: unknown, what: string): number => {
  if (typeof v !== 'number' || !Number.isInteger(v)) throw new Error(`${what} must be an integer`);
  return v;
};

export class LaciHeads {
  heads: Head[] = [];
  radius: [Head, number][] = [];   // dict order (first insertion, last value)
  framesPerToken = 0;

  static parse(json: string | unknown, path = 'heads_json'): LaciHeads {
    const spec: unknown = typeof json === 'string' ? JSON.parse(json) : json;
    if (typeof spec !== 'object' || spec === null) throw new Error(`${path}: no alignment heads`);
    const s = spec as Record<string, unknown>;
    const hs = s.heads;
    if (hs === undefined || hs === null || hs === false || (Array.isArray(hs) && hs.length === 0))
      throw new Error(`${path}: no alignment heads`);
    if (!Array.isArray(hs)) throw new Error(`${path}: heads is not an array`);
    const h = new LaciHeads();
    for (const e of hs as Record<string, unknown>[]) {
      const hd = { layer: intOf(e.layer, 'layer'), head: intOf(e.head, 'head') };
      h.heads.push(hd);
      const r = intOf(e.radius, 'radius');
      const found = h.radius.find(([x]) => x.layer === hd.layer && x.head === hd.head);
      if (found) found[1] = r; else h.radius.push([hd, r]);
    }
    if (typeof s.frames_per_token !== 'number') throw new Error(`${path}: frames_per_token must be a number`);
    h.framesPerToken = s.frames_per_token;
    return h;
  }
}

// A speaking-rate scale (condition-row `index`, `nBuckets` row buckets, edges of buckets first..last).
export class RateBuckets {
  constructor(public index: number, public edges: number[], public nBuckets: number, public first = 0) {}
  last(): number { return this.first + this.edges.length; }
  row(bucket: number): number { return bucket >= 0 && bucket < this.nBuckets ? bucket : this.nBuckets; }
}

export interface ProbeControl { textLen: number; mask: Windows | null; stop: boolean }

export interface Continuation {
  nRows: number;
  historyRows: number;
  seed: number | null;
  maxTokens: number;
  conditionValues: number[] | null;
  swapDelta: boolean;
  steeringDelta: Float32Array | null;
}

export interface FailureRecord {
  kind: string; detectRow: number; onsetRow: number; rollbackRow: number; clamped: boolean; bin: number;
  frontier: number; retrySeed: number | null; ignored: boolean; retryBucket: number | null;
}

export class LaciRecord {
  attempts = 0;
  capped = false;
  end = '';
  textLen = 0;
  nBins = 0;
  finalRows = 0;
  finalEosRow: number | null = null;
  generatedTotal = 0;
  paceTok: number | null = null;
  heads: Head[] = [];
  failures: FailureRecord[] = [];
  paceRetries = 0;
  rateBucket: number | null = null;

  toJson(): Record<string, unknown> {
    return {
      attempts: this.attempts, capped: this.capped, end: this.end, text_len: this.textLen, n_bins: this.nBins,
      final_rows: this.finalRows, final_eos_row: this.finalEosRow, generated_total: this.generatedTotal,
      pace_tok: this.paceTok, heads: this.heads.map((h) => [h.layer, h.head]),
      failures: this.failures.map((f) => ({
        kind: f.kind, detect_row: f.detectRow, onset_row: f.onsetRow, rollback_row: f.rollbackRow, clamped: f.clamped,
        bin: f.bin, frontier: f.frontier, retry_seed: f.retrySeed, ignored: f.ignored, retry_bucket: f.retryBucket,
      })),
      pace_retries: this.paceRetries, rate_bucket: this.rateBucket,
    };
  }
}

export interface SessionParams {
  heads: LaciHeads;
  cfg: LaciConfig;
  prefillRows: Int32Array[];          // rows that open the stream; may be empty
  seed: number | null;
  maxTokens: number;
  nCodebooks: number;
  hold: boolean;                      // a whole decode: nothing goes downstream before the result is final, so no rollback is ever clamped
  rates?: RateBuckets | null;
  conditionValues?: number[] | null;  // the request's condition-row ids
  bucket?: number | null;             // default: conditionValues[rates.index]
  rateDeltas?: (Float32Array | null)[] | null;  // per bucket rates.first..rates.last
  follower?: SpanFollower | null;     // not owned; rolled back with the stream
}

export interface RowRange { begin: number; end: number }

function failureRecord(f: Failure, retrySeed: number | null, ignored: boolean, resumeRow: number | null = null,
                       retryBucket: number | null = null): FailureRecord {
  const rollbackRow = resumeRow ?? f.rollbackRow;
  return {
    kind: f.kind, detectRow: f.detectRow, onsetRow: f.onsetRow, rollbackRow, clamped: rollbackRow !== f.rollbackRow,
    bin: f.bin, frontier: f.frontier, retrySeed, ignored, retryBucket,
  };
}

// One request under LACI: its stream, detector, retries and ACI probation.
export class LaciSession {
  readonly nPrefill: number;
  stream: Int32Array[];
  attemptStart: number;
  emitted = 0;
  attempts = 0;
  textDone = false;
  failures: FailureRecord[] = [];
  end: [string, number | null] | null = null;
  detector: Detector | null = null;
  private textLen = 0;
  private probation: [Failure, AciCenter] | null = null;
  private pendingFailure: Failure | null = null;
  private eosRow: number | null = null;
  private generatedTotal = 0;
  private ignored = new Set<string>();
  private conditionValues: number[] | null;
  private bucket: number | null;

  constructor(private p: SessionParams) {
    this.nPrefill = p.prefillRows.length;
    this.stream = p.prefillRows.slice();
    this.attemptStart = this.nPrefill;
    this.conditionValues = p.conditionValues ? p.conditionValues.slice() : null;
    this.bucket = p.bucket ?? null;
    if (this.bucket === null && p.rates && this.conditionValues) this.bucket = this.conditionValues[p.rates.index];
    if (p.rateDeltas) {
      const want = p.rates ? p.rates.last() - p.rates.first + 1 : 0;
      if (p.rateDeltas.length !== want) throw new Error(`rate_deltas: ${p.rateDeltas.length} for ${want} buckets`);
    }
  }

  rateBucket(): number | null { return !this.p.rates || !this.conditionValues ? null : this.bucket; }

  window(center: number): Windows {
    return this.p.heads.radius.map(([head, r]) => ({ head, lo: center - r + 1, hi: center + r - 1 }));
  }

  probeMask(): Windows | null { return this.probation ? this.window(this.probation[1].center) : null; }

  /** The logit the sampler adds to the stop id (every codebook) for the next row: cfg.stopBoost once the text is done
   *  and cfg.stopBoostDelay rows followed, else 0 (laci.py LaciSession.stop_boost). */
  stopBoost(): number {
    const done = this.detector?.doneRow() ?? null;
    if (done === null || this.stream.length < done + this.p.cfg.stopBoostDelay) return 0;
    return this.p.cfg.stopBoost;
  }

  onStep(probe: ProbeControl, step: number, pos: number, m: number): void {
    const row = this.attemptStart + step;
    this.p.follower?.observe(row - this.nPrefill, pos);
    if (!this.detector) {
      this.textLen = probe.textLen;
      this.detector = new Detector(probe.textLen, this.p.heads.framesPerToken, this.p.cfg, this.nPrefill);
    }
    const det = this.detector;
    if (this.probation) {
      const [failure, aci] = this.probation;
      probe.mask = this.window(aci.update(m));
      if (det.frontier() > failure.frontier) {
        this.probation = null;
        probe.mask = null;
      }
    }
    const f = det.observe(row, pos);
    if (f) {
      if (this.attempts < this.p.cfg.maxRetries) {
        this.pendingFailure = f;
        probe.stop = true;
        return;
      }
      this.ignore(f);
    }
    const stopRow = det.stopRow();
    if (stopRow !== null && row >= stopRow + this.p.nCodebooks + 2) {
      this.textDone = true;
      probe.stop = true;
    }
  }

  private ignore(f: Failure): void {
    const k = `${f.kind}\u0000${f.bin}\u0000${f.onsetRow}`;
    if (!this.ignored.has(k)) {
      this.ignored.add(k);
      this.failures.push(failureRecord(f, null, true));
    }
  }

  append(row: Int32Array, eosFrame: number | null): void {
    this.stream.push(row);
    this.generatedTotal += 1;
    if (eosFrame !== null) this.eosRow = this.attemptStart + eosFrame;
  }

  commitRow(): number {
    const n = this.stream.length;
    if (this.end) return n;
    if (this.p.hold) return this.nPrefill;
    let row: number;
    if (this.attempts >= this.p.cfg.maxRetries) {
      row = n;
    } else {
      const eos = this.eosRow ?? n - (this.p.nCodebooks - 1);
      row = Math.max(eos - this.p.cfg.rollbackMarginFrames, this.nPrefill);
      row = this.detector ? Math.min(row, this.detector.commitRow(n)) : this.nPrefill;
    }
    return Math.max(this.nPrefill, Math.min(row, n - 1));
  }

  // Generated rows that became final since the last drain (all remaining ones when `final`).
  drain(final = false): RowRange {
    const upto = final ? this.stream.length : this.commitRow();
    const start = this.nPrefill + this.emitted;
    const end = Math.max(start, Math.min(upto, this.stream.length));
    this.emitted += end - start;
    return { begin: start, end };
  }

  finishAttempt(): Continuation | null {
    let f = this.pendingFailure;
    if (!f && this.eosRow !== null && this.detector) {
      f = this.detector.onEos(this.eosRow);
      if (f && this.attempts >= this.p.cfg.maxRetries) {
        this.ignore(f);
        f = null;
      }
    }
    let resume = 0;
    if (f) resume = Math.max(f.rollbackRow, this.nPrefill + this.emitted);
    if (f && this.p.maxTokens - (resume - this.nPrefill) <= 0) {
      this.ignore(f);
      f = null;
    }
    if (!f) {
      if (this.textDone) this.end = ['text_done', this.detector!.stopRow()];
      else if (this.eosRow !== null) this.end = ['eos', this.eosRow];
      else this.end = ['max_tokens', null];
      return null;
    }
    this.attempts += 1;
    const seed = this.p.seed === null ? null : this.p.seed + 1000 * this.attempts;
    this.failures.push(failureRecord(f, seed, false, resume, null));
    this.stream.length = resume;
    this.detector!.truncate(resume);
    this.p.follower?.truncate(resume - this.nPrefill);
    this.probation = [f, new AciCenter(this.detector!.lastPosBefore(resume), this.textLen)];
    this.pendingFailure = null;
    this.textDone = false;
    this.eosRow = null;
    this.attemptStart = resume;

    let steeringDelta: Float32Array | null = null;
    if (this.p.rateDeltas && this.bucket !== null && this.p.rates) {
      const d = this.p.rateDeltas[this.bucket - this.p.rates.first];
      if (d && d.length) steeringDelta = d;
    }
    return {
      nRows: resume, historyRows: resume - this.nPrefill, seed, maxTokens: this.p.maxTokens - (resume - this.nPrefill),
      conditionValues: this.conditionValues ? this.conditionValues.slice() : null,
      swapDelta: !!this.p.rateDeltas, steeringDelta,
    };
  }

  // The final result's EOS as an aligned frame in generated rows.
  eosFrame(): number | null {
    return !this.end || this.end[1] === null ? null : this.end[1] - this.nPrefill;
  }

  record(): LaciRecord {
    if (!this.end) throw new Error('LaciSession::record before the result is final');
    const det = this.detector;
    const r = new LaciRecord();
    r.attempts = this.attempts;
    r.capped = this.attempts >= this.p.cfg.maxRetries;
    r.end = this.end[0];
    r.textLen = this.textLen;
    r.nBins = det ? det.nb : 0;
    r.finalRows = (this.end[1] ?? this.stream.length) - this.nPrefill;
    r.finalEosRow = this.end[1];
    r.generatedTotal = this.generatedTotal;
    if (det) r.paceTok = det.paceTok();
    r.heads = this.p.heads.heads;
    r.failures = this.failures;
    r.paceRetries = 0;
    r.rateBucket = this.rateBucket();
    return r;
  }
}

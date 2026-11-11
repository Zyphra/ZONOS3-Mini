// Replays servers/ggml/tests/fixtures/laci-replay through the TS LaciSession / SpanFollower exactly as
// servers/ggml/tests/laci-replay.cpp does and compares with the stored results.
import { readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { defaultLaciConfig, LaciHeads, LaciSession, RateBuckets, type ProbeControl, type Windows } from '../../src/request/laci';
import { defaultFollowConfig, SpanFollower, type FollowConfig, type Range } from '../../src/request/spanFollow';

const DIR = new URL('../../../ggml/tests/fixtures/laci-replay/', import.meta.url);
type J = Record<string, any>;

const followCfg = (j: J | undefined): FollowConfig => ({ ...defaultFollowConfig(), ...(j ?? {}) });
const makeFollower = (j: J) =>
  new SpanFollower(j.ranges as Range[], (j.deltas as number[][]).map((d) => Float32Array.from(d)), followCfg(j.cfg), j.origin ?? 0);
const maskJson = (m: Windows | null) => (m ? m.map((w) => [w.head.layer, w.head.head, w.lo, w.hi]) : null);

function runLaci(t: J): J {
  const textLen: number = t.text_len;
  const heads = LaciHeads.parse(t.heads_json);
  if (t.frames_per_token != null) heads.framesPerToken = t.frames_per_token;
  const cfg = defaultLaciConfig();
  const keys: [string, keyof typeof cfg][] = [['bin_tokens', 'binTokens'], ['cover_frac', 'coverFrac'],
    ['skip_frac', 'skipFrac'], ['stall_factor', 'stallFactor'], ['min_stall_frames', 'minStallFrames'],
    ['tail_frames', 'tailFrames'], ['rollback_margin_frames', 'rollbackMarginFrames'], ['max_retries', 'maxRetries'],
    ['pace_window_bins', 'paceWindowBins'], ['warmup_bins', 'warmupBins'], ['stop_boost', 'stopBoost'],
    ['stop_boost_delay', 'stopBoostDelay']];
  for (const [k, c] of keys) if (t.cfg?.[k] !== undefined) cfg[c] = t.cfg[k];
  const nPrefill: number = t.n_prefill ?? 0;
  const C: number = t.n_codebooks ?? 10;
  const follower = t.follower ? makeFollower(t.follower) : null;
  const r = t.rates;
  const s = new LaciSession({
    heads, cfg, prefillRows: Array.from({ length: nPrefill }, () => Int32Array.of(-1)),
    seed: 'seed' in t ? t.seed : 0, maxTokens: t.max_tokens ?? 4000, nCodebooks: C, hold: t.hold ?? false,
    rates: r ? new RateBuckets(r.index, r.edges, r.n_buckets, r.first ?? 0) : null,
    conditionValues: t.condition_values ?? null, bucket: t.bucket ?? null,
    rateDeltas: t.rate_deltas ? (t.rate_deltas as (number[] | null)[]).map((d) => (d ? Float32Array.from(d) : null)) : null,
    follower,
  });
  const tags = (b: number, e: number) => s.stream.slice(b, e).map((row) => row[0]);
  const attempts: J[] = t.attempts ?? [{ steps: t.steps, eos_at: t.eos_at ?? null, eos_lag: t.eos_lag ?? 0 }];
  const out: J[] = [];
  for (let k = 0; ; k++) {
    if (k >= (t.max_attempts ?? 64)) throw new Error('more than max_attempts attempts');
    const a = attempts[Math.min(k, attempts.length - 1)];
    const tag = k + 1;
    const eosAt: number | null = a.eos_at ?? null, eosLag: number = a.eos_lag ?? 0;
    const probe: ProbeControl = { textLen, mask: s.probeMask(), stop: false };
    const rec: J = { mask: maskJson(probe.mask) };
    const changes: unknown[] = [], drained: number[] = [], fweights: number[][] = [];
    s.append(Int32Array.of(tag), null);
    let d = s.drain();
    drained.push(...tags(d.begin, d.end));
    let countdown: number | null = null, stepsRun = 0, stopStep: number | null = null, boostStep: number | null = null;
    for (let step = 1; step <= a.steps.length; step++) {
      const [pos, m] = a.steps[step - 1];
      if (follower) fweights.push(follower.weights(s.attemptStart + step - s.nPrefill).slice());
      const before = JSON.stringify(maskJson(probe.mask));
      s.onStep(probe, step, pos, m);
      stepsRun = step;
      const after = maskJson(probe.mask);
      if (JSON.stringify(after) !== before) changes.push([step, after]);
      let eos: number | null = null;
      if (eosAt !== null && step >= eosAt) {
        eos = eosAt - eosLag;
        countdown ??= C + 1;
      }
      s.append(Int32Array.of(tag * 100000 + step), eos);
      if (boostStep === null && s.stopBoost() > 0) boostStep = step;   // the first step whose next row is boosted
      if (countdown !== null) countdown -= 1;
      if (probe.stop || countdown === 0) {
        if (probe.stop) stopStep = step;
        break;
      }
      d = s.drain();
      drained.push(...tags(d.begin, d.end));
    }
    rec.mask_changes = changes;
    rec.steps_run = stepsRun;
    rec.stop_step = stopStep;
    rec.boost_step = boostStep;
    rec.drained = drained;
    if (follower) rec.follower_weights = fweights;
    const cont = s.finishAttempt();
    rec.continuation = cont && {
      n_rows: cont.nRows, history_rows: cont.historyRows, seed: cont.seed, max_tokens: cont.maxTokens,
      condition_values: cont.conditionValues, swap_delta: cont.swapDelta,
      steering_delta: cont.steeringDelta ? Array.from(cont.steeringDelta) : null,
      probe_mask: maskJson(s.probeMask()),
    };
    out.push(rec);
    if (!cont) break;
  }
  const fd = s.drain(true);
  return {
    attempts: out, final_drain: tags(fd.begin, fd.end), eos_frame: s.eosFrame(), end: s.end,
    record: s.record().toJson(), stream: tags(s.nPrefill, s.stream.length),
  };
}

function runSpanFollow(t: J): J {
  const f = makeFollower(t);
  const results = (t.ops as J[]).map((op) => {
    switch (op[0]) {
      case 'observe': f.observe(op[1], op[2]); return f.position();
      case 'on_step': f.onStep(op[1], op[2]); return f.position();
      case 'weights': return f.weights(op[1]).slice();
      case 'delta': { const d = f.delta(op[1]); return d ? Array.from(d) : null; }
      case 'truncate': f.truncate(op[1]); return f.position();
      case 'position': return f.position();
      default: throw new Error(`unknown span_follow op ${op[0]}`);
    }
  });
  return { results };
}

// The C++ driver dumps float32 vectors through double: compare numbers at float32 precision.
function same(want: unknown, got: unknown, path = '$'): void {
  if (typeof want === 'number' && typeof got === 'number') {
    if (Math.fround(want) !== Math.fround(got)) throw new Error(`${path}: want ${want}, got ${got}`);
    return;
  }
  if (Array.isArray(want) && Array.isArray(got)) {
    if (want.length !== got.length) throw new Error(`${path}: length ${want.length} != ${got.length}`);
    want.forEach((w, i) => same(w, got[i], `${path}[${i}]`));
    return;
  }
  if (want && got && typeof want === 'object' && typeof got === 'object') {
    const wk = Object.keys(want).sort(), gk = Object.keys(got).sort();
    expect(gk, path).toEqual(wk);
    for (const k of wk) same((want as J)[k], (got as J)[k], `${path}.${k}`);
    return;
  }
  expect(got, path).toEqual(want);
}

// pace_miss carries a duration target (duration-target takes): PaceMonitor is not ported (plan step 6).
const cases = readdirSync(DIR).filter((f) => f.endsWith('.trace.json')).map((f) => f.slice(0, -'.trace.json'.length));

describe('laci-replay fixtures', () => {
  for (const name of cases) {
    const t: J = JSON.parse(readFileSync(new URL(`${name}.trace.json`, DIR), 'utf8'));
    const skip = t.duration_target != null;
    it.skipIf(skip)(name, () => {
      const want = JSON.parse(readFileSync(new URL(`${name}.expected.json`, DIR), 'utf8'));
      const mode = t.mode ?? 'laci';
      const got = mode === 'laci' ? runLaci(t) : mode === 'span_follow' ? runSpanFollow(t) : null;
      if (!got) throw new Error(`unsupported mode ${mode}`);
      same(want, got);
    });
  }
});

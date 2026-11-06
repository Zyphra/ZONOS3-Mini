// Benchmark page for harness/bench.mjs. ?mode=e2e: the engine worker pair through localFetch (time to first audio and
// real-time factor of streamed /v2/generate runs). ?mode=micro: the GPU classes on this thread (backbone decode step,
// codec chunk, host iSTFT + resampler). Progress lines go to POST /bench/log, the result JSON to POST /bench/result.
import { Backbone } from '../src/backbone';
import { Gpu, initGpu } from '../src/gpu/device';
import { loadModel, modelsBaseUrl } from '../src/gpu/weights';
import { IstftHead } from '../src/istft';
import { buildPrompt, modelHp } from '../src/prompt';
import { Resampler } from '../src/resample';
import { createTokenizer } from '../src/tokenizer';
import { ZacStream } from '../src/zac';

const TEXT = 'Hello! This is a cloned voice, speaking from its speaker embedding.';
const SAMPLE_RATE = 44100;

const logEl = document.getElementById('log')!;
const post = (path: string, body: string) => fetch(path, { method: 'POST', body }).then(() => {}, () => {});
const log = (s: string) => {
  logEl.textContent += s + '\n';
  console.log(s);
  void post('/bench/log', s);
};

const median = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

async function adapterName(): Promise<string> {
  try {
    const info = (await navigator.gpu.requestAdapter())?.info;
    return info ? [info.vendor, info.architecture, info.device].filter(Boolean).join('/') || '-' : '-';
  } catch {
    return '-';
  }
}

interface Run { case: string; i: number; ttfaMs: number; totalMs: number; audioS: number; rtf: number }

async function e2e(runs: number): Promise<Record<string, unknown>> {
  // Dynamic: src/client.ts spawns the engine workers when evaluated, which micro mode must not do.
  const { engine, localFetch } = await import('../src/client');
  const t0 = performance.now();
  await engine.init(() => {});
  const initMs = performance.now() - t0;
  log(`init ${Math.round(initMs)} ms`);
  const id = ((await (await fetch('/models/voices/voices.json')).json()) as { id: string }[])[0].id;
  const plain = {
    text: TEXT, language: 'en_us', speaker_mode: 'embedding', speaker_embedding_id: id, stream: true, format: 'pcm', seed: 42,
  };
  const cases: [string, Record<string, unknown>][] = [['plain', plain], ['cfg', { ...plain, speaker_cfg_scale: 1.5 }]];
  const once = async (req: Record<string, unknown>) => {
    const t = performance.now();
    const res = await localFetch('/v2/generate', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(req) });
    if (res.status !== 200) throw new Error(`generate: ${res.status} ${await res.text()}`);
    const reader = res.body!.getReader();
    let bytes = 0, ttfa = -1;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value.byteLength > 0 && ttfa < 0) ttfa = performance.now() - t;
      bytes += value.byteLength;
    }
    const totalMs = performance.now() - t;
    const audioS = bytes / 4 / SAMPLE_RATE;
    return { ttfaMs: ttfa, totalMs, audioS, rtf: totalMs / 1000 / audioS };
  };
  await once(plain);   // speaker encoder load + the voice's embedding
  log('warm-up done');
  const out: Run[] = [];
  for (const [name, req] of cases) {
    for (let i = 0; i < runs; i++) {
      const r = { case: name, i, ...(await once(req)) };
      log(`run ${name} #${i} ttfa ${Math.round(r.ttfaMs)} ms, total ${Math.round(r.totalMs)} ms, rtf ${r.rtf.toFixed(2)}`);
      out.push(r);
    }
  }
  return { mode: 'e2e', initMs, runs: out };
}

async function micro(): Promise<Record<string, unknown>> {
  const gpu = new Gpu(await initGpu(m => log('fatal: ' + m)));
  const base = modelsBaseUrl();
  const bbw = await loadModel(gpu, base, 'backbone');
  const zacw = await loadModel(gpu, base, 'zac');
  log('models loaded');
  const res: Record<string, unknown> = { mode: 'micro' };

  const bb = new Backbone(gpu, bbw, []);
  const prompt = buildPrompt(createTokenizer(), TEXT, { buckets: [], silencePrefill: 3 }, modelHp(bbw.meta));
  await bb.prefill(0, prompt.ids, prompt.cond);
  await bb.prefill(1, prompt.ids, prompt.cond);
  const n = prompt.nRows;
  const row = prompt.ids.subarray((n - 1) * 11, n * 11);
  const timed = async (warm: number, reps: number, f: () => Promise<unknown>) => {
    for (let i = 0; i < warm; i++) await f();
    const ts: number[] = [];
    for (let i = 0; i < reps; i++) {
      const t = performance.now();
      await f();
      ts.push(performance.now() - t);
    }
    return median(ts);
  };
  res.bb_step_ms = await timed(4, 32, () => bb.step(0, row, null, null));
  log(`bb_step_ms ${(res.bb_step_ms as number).toFixed(2)}`);
  res.bb_pair_ms = await timed(4, 32, () => bb.stepPair(row, null, null, null));
  log(`bb_pair_ms ${(res.bb_pair_ms as number).toFixed(2)}`);

  const zac = await ZacStream.create(gpu, zacw);
  res.gemm_plan = `${zac.gemmPlan.skinny}/${zac.gemmPlan.large}`;
  zac.reset();
  let p0 = 0;
  res.zac_chunk_ms = await timed(1, 8, async () => {
    await zac.chunk(new Int32Array(160), p0, 16);
    p0 += 16;
  });
  log(`zac_chunk_ms ${(res.zac_chunk_ms as number).toFixed(2)}`);

  const host = (k: string) => {
    const v = zacw.host.get(k);
    if (!v) throw new Error(`zac: missing ${k}`);
    return v;
  };
  const nFft = zacw.meta['head.n_fft'] as [number, number];
  const istft = new IstftHead({
    nFft, hop: Number(zacw.meta['head.hop']), windows: [host('head.window.0'), host('head.window.1')],
    mixer: host('head.mixer.weight'), maxLogMag: Number(zacw.meta['head.max_log_magnitude']),
  });
  const rs = new Resampler(host('resample.kernel'));
  const cols = 256;
  const rand = (len: number) => { const a = new Float32Array(len); for (let i = 0; i < len; i++) a[i] = Math.random() * 2 - 1; return a; };
  const ts: number[] = [];
  for (let i = 0; i < 8; i++) {
    const s0 = rand(cols * (nFft[0] + 2)), s1 = rand(cols * (nFft[1] + 2));
    const t = performance.now();
    rs.push(istft.push(s0, s1, cols));
    ts.push(performance.now() - t);
  }
  res.istft_ms = median(ts);
  log(`istft_ms ${(res.istft_ms as number).toFixed(2)}`);
  return res;
}

async function main(): Promise<void> {
  const q = new URLSearchParams(location.search);
  const mode = q.get('mode') ?? 'e2e';
  const runs = Number(q.get('runs') ?? '3');
  let result: Record<string, unknown>;
  try {
    const adapter = await adapterName();
    log(`bench ${mode} on ${adapter}`);
    result = { ...(mode === 'micro' ? await micro() : await e2e(runs)), adapter };
  } catch (e) {
    result = { error: e instanceof Error ? `${e.message}\n${e.stack}` : String(e) };
  }
  log('done');
  await post('/bench/result', JSON.stringify(result));
}

void main();

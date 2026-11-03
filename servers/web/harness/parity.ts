// Parity checks of the engine classes against the CPU ggml goldens in public/goldens (scripts/make_goldens.sh).
// Each harness/checks/<area>.ts exports `checks: Record<string, Check>`; ?only=a,b runs a subset. Results land in
// #result[data-done] as JSON {name: {ok, metric}} (read by harness/parity.mjs).
import { Gpu, initGpu } from '../src/gpu/device';
import { modelsBaseUrl } from '../src/gpu/weights';
import type { Check, CheckResult, Ctx } from './checks/util';
import { goldenNpy, goldenBytes } from './checks/util';

const modules = import.meta.glob('./checks/*.ts', { eager: true }) as Record<string, { checks?: Record<string, Check> }>;
const all: Record<string, Check> = {};
for (const m of Object.values(modules)) Object.assign(all, m.checks ?? {});

const logEl = document.getElementById('log')!;
const log = (s: string) => { logEl.textContent += s + '\n'; console.log(s); };

async function main(): Promise<void> {
  const only = new URLSearchParams(location.search).get('only')?.split(',').filter(Boolean) ?? null;
  const names = Object.keys(all).filter(n => !only || only.includes(n));
  const results: Record<string, CheckResult & { error?: string }> = {};
  const gpu = new Gpu(await initGpu(m => log('fatal: ' + m)));
  const ctx: Ctx = { gpu, base: modelsBaseUrl(), log, golden: goldenBytes, npy: goldenNpy };
  for (const n of names) {
    const t0 = performance.now();
    try {
      results[n] = await all[n](ctx);
    } catch (e) {
      results[n] = { ok: false, metric: {}, error: e instanceof Error ? `${e.message}\n${e.stack}` : String(e) };
    }
    log(`${results[n].ok ? 'PASS' : 'FAIL'} ${n} ${JSON.stringify(results[n].metric)} ${results[n].error ?? ''} (${Math.round(performance.now() - t0)} ms)`);
  }
  const el = document.getElementById('result')!;
  el.textContent = JSON.stringify(results);
  el.dataset.done = '1';
}

main().catch(e => {
  const el = document.getElementById('result')!;
  el.textContent = JSON.stringify({ harness: { ok: false, metric: {}, error: String(e) } });
  el.dataset.done = '1';
});

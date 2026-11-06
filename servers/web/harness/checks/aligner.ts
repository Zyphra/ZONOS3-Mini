// FuseAlign parity: the WebGPU Aligner vs the PyTorch reference (fusealign_hub.load_aligner on CPU, batch size 1) on
// short default-voice clips with their transcripts (harness/goldens/aligner, scripts/make_aligner_goldens.py). Every
// token's start and end must agree within 20 ms, its placed flag must match, and presence within 0.05.
import { Aligner, ALIGN_MAX_BYTES } from '../../src/aligner';
import { HttpError } from '../../src/api/http';
import goldens from '../goldens/aligner/goldens.json';
import type { Check, CheckResult, Ctx } from './util';

interface GoldenWord { word: string; start: number; end: number; presence: number; placed: boolean }
interface GoldenClip { name: string; file: string; samples: number; tokens: string[]; words: GoldenWord[] }

const TOL_SECONDS = 0.02;
const TOL_PRESENCE = 0.05;
const audio = import.meta.glob('../goldens/aligner/*.s16', { query: '?url', import: 'default', eager: true }) as Record<string, string>;

let aligner: Promise<Aligner> | null = null;

async function clipPcm(clip: GoldenClip): Promise<Float32Array> {
  const url = audio[`../goldens/aligner/${clip.file}`];
  if (!url) throw new Error(`missing golden audio ${clip.file}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${clip.file}: ${res.status}`);
  const s16 = new Int16Array(await res.arrayBuffer());
  if (s16.length !== clip.samples) throw new Error(`${clip.file}: ${s16.length} samples, expected ${clip.samples}`);
  return Float32Array.from(s16, v => v / 32768);
}

async function compare(ctx: Ctx, clip: GoldenClip): Promise<CheckResult> {
  const al = await (aligner ??= Aligner.load(ctx.gpu, ctx.base));
  const pcm = await clipPcm(clip);
  const t0 = performance.now();
  const got = await al.align(pcm, clip.tokens);
  const ms = Math.round(performance.now() - t0);
  let maxStart = 0, maxEnd = 0, maxPresence = 0, placedMismatch = 0, over = 0;
  clip.words.forEach((want, i) => {
    const g = got[i];
    const ds = Math.abs(g.start - want.start), de = Math.abs(g.end - want.end);
    maxStart = Math.max(maxStart, ds);
    maxEnd = Math.max(maxEnd, de);
    maxPresence = Math.max(maxPresence, Math.abs(g.presence - want.presence));
    if (g.placed !== want.placed) placedMismatch++;
    if (ds > TOL_SECONDS || de > TOL_SECONDS) over++;
  });
  const sameTokens = got.length === clip.words.length && got.every((g, i) => g.word === clip.words[i].word);
  const metric = {
    words: clip.words.length, maxStartMs: Math.round(maxStart * 1000), maxEndMs: Math.round(maxEnd * 1000),
    over20ms: over, maxPresence: Number(maxPresence.toFixed(4)), placedMismatch, alignMs: ms,
  };
  const ok = sameTokens && over === 0 && placedMismatch === 0 && maxPresence <= TOL_PRESENCE;
  return { ok, metric };
}

async function rejects400(run: () => Promise<unknown>): Promise<boolean> {
  try {
    await run();
    return false;
  } catch (e) {
    return e instanceof HttpError && e.status === 400;
  }
}

// resolveJsonModule types the committed goldens.json structurally, matching GoldenClip.
const clips: GoldenClip[] = goldens.clips;

export const checks: Record<string, Check> = {
  ...Object.fromEntries(clips.map(clip => [`aligner_${clip.name}`, (ctx: Ctx) => compare(ctx, clip)])),
  async aligner_limits(ctx) {
    const al = await (aligner ??= Aligner.load(ctx.gpu, ctx.base));
    const longAudio = await rejects400(() => al.align(new Float32Array(30 * 16000 + 1), ['hello']));
    const longText = await rejects400(() => al.align(new Float32Array(16000), ['x'.repeat(ALIGN_MAX_BYTES + 1)]));
    const badToken = await rejects400(() => al.align(new Float32Array(16000), ['two words']));
    // Under one 512-sample patch: every word takes the uniform fallback, no forward.
    const short = await al.align(new Float32Array(400), ['a', 'b']);
    const shortOk = short.length === 2 && !short[0].placed && short[1].end === 400 / 16000;
    return { ok: longAudio && longText && badToken && shortOk, metric: { longAudio, longText, badToken, shortOk } };
  },
};

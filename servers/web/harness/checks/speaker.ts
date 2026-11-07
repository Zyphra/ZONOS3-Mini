// Speaker-encoder parity: SpeakerEncoder vs spk-encoder-cli (goldens spk.npy, spk/*.npy) on assets/default_voices/AmericanFemale.
// speaker_embed decodes the mp3 in the browser (OfflineAudioContext, 44.1 kHz, channels averaged) as the client does;
// speaker_embed_pcm feeds the exact 44.1 kHz PCM ffmpeg produced for the CLI (goldens/spk/pcm44k.npy).
import { SpeakerEncoder } from '../../src/speaker';
import { cosine, maxAbs, type Check, type CheckResult, type Ctx } from './util';

let encoder: Promise<SpeakerEncoder> | null = null;

async function decodeMp3(url: string): Promise<Float32Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
  const buf = await new OfflineAudioContext(1, 1, 44100).decodeAudioData(await res.arrayBuffer());
  const out = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < out.length; i++) out[i] += ch[i] / buf.numberOfChannels;
  }
  return out;
}

async function compare(ctx: Ctx, pcm: Float32Array, exactInput: boolean): Promise<CheckResult> {
  const enc = await (encoder ??= SpeakerEncoder.load(ctx.gpu, ctx.base));
  const want = (await ctx.npy('spk.npy')).data;
  const wantFbank = await ctx.npy('spk/fbank.npy');
  const { fbank, T } = enc.features(pcm);
  const t0 = performance.now();
  const run = await enc.run(fbank, T, true);
  const ms = Math.round(performance.now() - t0);
  const t1 = performance.now();
  await enc.embed(pcm);
  const embedMs = Math.round(performance.now() - t1);
  const Tw = wantFbank.shape[0];
  const n = Math.min(T, Tw);
  const featCos: number[] = [];
  for (let s = 0; s < 4; s++) {
    const g = (await ctx.npy(`spk/feat_${s + 1}.npy`)).data;
    featCos.push(Number(cosine(run.feats[s].subarray(0, n * 2560), g.subarray(0, n * 2560)).toPrecision(9)));
  }
  let norm = 0;
  for (const v of run.embedding) norm += v * v;
  norm = Math.sqrt(norm);
  const cos = cosine(run.embedding, want);
  const metric = {
    T, goldenT: Tw, samples44k: pcm.length,
    fbankMaxAbs: Number(maxAbs(fbank.subarray(0, n * 80), wantFbank.data.subarray(0, n * 80)).toFixed(6)),
    featCos, pooledCos: Number(cosine(run.pooled, (await ctx.npy('spk/pooled.npy')).data).toPrecision(9)),
    cosine: Number(cos.toPrecision(9)), norm: Number(norm.toFixed(4)), goldenNorm: 4.2146, firstRunMs: ms, embedMs,
  };
  const ok = cos >= 0.999 && Math.abs(norm - 4.21) <= 0.01 && (!exactInput || T === Tw);
  return { ok, metric };
}

export const checks: Record<string, Check> = {
  async speaker_embed(ctx) {
    return compare(ctx, await decodeMp3('/models/voices/AmericanFemale.mp3'), false);
  },
  async speaker_embed_pcm(ctx) {
    return compare(ctx, (await ctx.npy('spk/pcm44k.npy')).data, true);
  },
};

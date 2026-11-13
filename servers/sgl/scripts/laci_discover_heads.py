#!/usr/bin/env python3
"""Find ZONOS3 Mini's text-alignment heads for LACI.

Generates --n zero-shot sentences with an all-heads probe (every q head of both global-attention layers, layers 3 and
7), records per decode step each head's attention argmax / centroid / text mass / entropy over the text span, and keeps
the heads whose reading is a monotone sweep over the whole text.

Per head, per utterance, then averaged over utterances:
  text_mass  mean softmax mass on the text keys
  corr       Pearson(step, centroid)
  mono       fraction of steps whose argmax >= the previous argmax - 1
  coverage   max argmax / (text_len - 1)
  entropy    mean entropy (nats) of the normalized text distribution
Qualifying head: text_mass >= 0.3, corr >= 0.9, mono >= 0.9, coverage >= 0.9; at most 4 are kept, ranked by
corr * text_mass. With none, the best head by corr * text_mass is taken as a fallback when its corr >= 0.8; otherwise
the table is printed and the exit code is 2. Per-head ACI radius = clamp(round(8 * entropy) + 1, 2, 8).

The set (--data-dir) holds ``text`` and ``prompt_wav.scp`` (``<uid> <value>`` lines; the prompt clip of a uid is
``waveform/<file name of its value>``). The file written (default: assets/laci_heads/<model-name>_heads.json for
English, <model-name>_heads_<speech code>.json otherwise) is what zonos3.tts.laci.discover_laci_heads picks up.

    source servers/sgl/scripts/env.sh
    CUDA_VISIBLE_DEVICES=<gpu> uv run --locked --package zonos3-sgl python servers/sgl/scripts/laci_discover_heads.py \\
        --language <request code> --data-dir <set dir> [--model-path PATH] [--out PATH]
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from zonos3 import languages
from zonos3.tokenizer.matched_codec import entry
from zonos3.tts.prompt import TTSPromptConfig, silence_rows_from_codes
from zonos3.utils.hashing import sha256_file
from zonos3.utils.release import RELEASE_LM, resolve_dir_or_local
from zonos3_sgl.tts.align_probe import AlignProbe
from zonos3_sgl.tts.llm import TTSLLM, TTSSamplingParams

REPO = Path(__file__).resolve().parents[3]

MIN_MASS, MIN_CORR, MIN_MONO, MIN_COV, MAX_HEADS, FALLBACK_CORR = 0.3, 0.9, 0.9, 0.9, 4, 0.8


def load_set(data_dir) -> list[tuple[str, str, Path]]:
    """(uid, text, prompt wav) of a set, in ``text`` order; uids without a prompt clip are left out."""

    def read(name):
        path = Path(data_dir) / name
        if not path.is_file():
            return {}
        rows = (line.split(maxsplit=1) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        return {row[0]: row[1] if len(row) > 1 else "" for row in rows}

    texts, prompts = read("text"), read("prompt_wav.scp")
    items = []
    for uid, text in texts.items():
        if uid not in prompts:
            continue
        wav = Path(data_dir) / "waveform" / Path(prompts[uid]).name
        if not wav.is_file():
            raise SystemExit(f"error: {uid}: no prompt clip {wav}")
        items.append((uid, text, wav))
    if not items:
        raise SystemExit(f"error: no utterances in {data_dir}")
    return items


class Recorder:
    """Per step: [argmax, centroid, text mass, entropy] for every head (kept on the GPU until the run ends)."""

    def __init__(self):
        self.heads = None
        self.stats = []

    def on_attention(self, probe, step, text_attn, text_mass):
        if self.heads is None:
            self.heads = sorted(text_attn)
        a = torch.stack([text_attn[h] for h in self.heads])  # [H, L], rows sum to 1
        idx = torch.arange(a.shape[1], device=a.device, dtype=torch.float32)
        mass = torch.stack([text_mass[h] for h in self.heads])
        ent = -(a * (a + 1e-12).log()).sum(-1)
        self.stats.append(torch.stack([a.argmax(-1).float(), (a * idx).sum(-1), mass, ent]))

    def table(self):
        """[steps, 4, heads] numpy array for this utterance."""
        return torch.stack(self.stats).cpu().numpy()


def head_metrics(s, text_len):
    steps = np.arange(1, s.shape[0] + 1, dtype=np.float64)
    out = []
    for h in range(s.shape[2]):
        am, cen, mass, ent = (s[:, k, h].astype(np.float64) for k in range(4))
        corr = 0.0 if cen.std() < 1e-9 else float(np.corrcoef(steps, cen)[0, 1])
        mono = float(np.mean(am[1:] >= am[:-1] - 1)) if len(am) > 1 else 0.0
        out.append(
            {
                "text_mass": float(mass.mean()),
                "corr": corr,
                "mono": mono,
                "coverage": float(am.max() / max(text_len - 1, 1)),
                "entropy": float(ent.mean()),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--language", required=True, help="server language code (zonos3.languages)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--silence-prefill", type=int, default=3)
    ap.add_argument("--model-path", default=RELEASE_LM)
    ap.add_argument("--model-name", default="zonos3_mini")
    ap.add_argument("--out", default=None, help="default: the language's assets/laci_heads file")
    args = ap.parse_args()
    try:
        language = languages.normalize(args.language)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    speech = languages.BY_CODE[language].speech
    if args.out is None:
        suffix = "" if speech == "en" else f"_{speech}"
        args.out = str(REPO / "assets" / "laci_heads" / f"{args.model_name}_heads{suffix}.json")

    items = load_set(args.data_dir)
    uids = [uid for uid, _, _ in items]
    pick = set(random.Random(args.seed).sample(uids, min(args.n, len(uids))))
    items = [it for it in items if it[0] in pick]

    tts = TTSLLM(
        args.model_path,
        decode_audio=False,
        max_running_req=len(items),
        attention_backend="fa",
        cache_type="naive",
        memory_ratio=0.6,
    )
    cfg = tts.engine.model_config
    frame = entry(cfg.audio_codec).get("silence_frame")
    if frame is None:
        raise SystemExit(f"error: codec {cfg.audio_codec!r} has no silence_frame")
    silence = silence_rows_from_codes(
        torch.tensor(frame, dtype=torch.int32),
        args.silence_prefill,
        TTSPromptConfig(n_codebooks=cfg.n_codebooks, audio_pad_id=cfg.audio_pad_id, text_vocab=int(cfg.text_vocab)),
    )

    recs = [Recorder() for _ in items]
    probes = [AlignProbe(heads=(), observer=r, all_heads=True) for r in recs]
    res = tts.generate(
        [t for _, t, _ in items],
        TTSSamplingParams(seed=args.seed, max_tokens=args.max_tokens),
        decode_audio=False,
        language=language,
        text_normalization=True,
        speaker_reference_audio=[str(w) for *_, w in items],
        audio_prefill_rows=[silence] * len(items),
        align_probes=probes,
    )

    per_utt, fpt = [], []
    for (u, _, _), r, rec, probe in zip(items, res, recs, probes):
        if len(rec.stats) < 20:
            print(f"skip {u}: only {len(rec.stats)} steps", file=sys.stderr)
            continue
        per_utt.append(head_metrics(rec.table(), probe.text_len))
        fpt.append(len(r["audio_tokens"]) / probe.text_len)
        heads = rec.heads
    if not per_utt:
        sys.exit("error: no usable utterance")
    keys = ("text_mass", "corr", "mono", "coverage", "entropy")
    table = []
    for h, (layer, head) in enumerate(heads):
        row = {"layer": layer, "head": head}
        row.update({k: statistics.fmean(m[h][k] for m in per_utt) for k in keys})
        table.append(row)

    print(f"{len(per_utt)} utterances; frames/token median {statistics.median(fpt):.2f}")
    print(f"{'layer':>5} {'head':>4} {'mass':>6} {'corr':>6} {'mono':>6} {'cover':>6} {'entropy':>7}")
    for t in sorted(table, key=lambda t: -t["corr"] * t["text_mass"]):
        print(
            f"{t['layer']:5d} {t['head']:4d} {t['text_mass']:6.3f} {t['corr']:6.3f} {t['mono']:6.3f} "
            f"{t['coverage']:6.3f} {t['entropy']:7.3f}"
        )

    ranked = sorted(table, key=lambda t: -t["corr"] * t["text_mass"])
    qual = [
        t
        for t in ranked
        if t["text_mass"] >= MIN_MASS and t["corr"] >= MIN_CORR and t["mono"] >= MIN_MONO and t["coverage"] >= MIN_COV
    ][:MAX_HEADS]
    fallback = False
    if not qual:
        if ranked[0]["corr"] < FALLBACK_CORR:
            print(
                f"no head qualifies and the best corr is {ranked[0]['corr']:.3f} < {FALLBACK_CORR}: "
                "no usable alignment head",
                file=sys.stderr,
            )
            sys.exit(2)
        qual, fallback = ranked[:1], True
    for t in qual:
        t["radius"] = min(8, max(2, round(8 * t["entropy"]) + 1))

    out = {
        "model": args.model_name,
        "model_path": args.model_path,
        "model_sha256": sha256_file(Path(resolve_dir_or_local(args.model_path)) / "model.pth"),
        "data_dir": args.data_dir,
        "n_utts": len(per_utt),
        "seed": args.seed,
        "frames_per_token": statistics.median(fpt),
        "fallback": fallback,
        "heads": qual,
        "table": table,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}: heads {[(t['layer'], t['head']) for t in qual]} fallback={fallback}")


if __name__ == "__main__":
    main()

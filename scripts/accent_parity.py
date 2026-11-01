#!/usr/bin/env python3
"""GPU parity checks for per-request accent adapters (README "Accent adapters").

    python scripts/accent_parity.py --base-model checkpoints/gdn375k \
        --accent-dir ADAPTERS --merged-model checkpoints/MERGED --merged-accent NAME \
        --out-dir OUT [--mixed-accents A,B,...] \
        [--reference-audio WAV] [--speaker-embedding VECTOR.npy]

Runs three offline engines, one process each (an engine owns the process-wide
context), on the same seeded requests, records every step's logits and samples,
and writes OUT/report.json:

(a) base rows: requests without an accent, batched with accent requests on the
    engine with adapters loaded, against the engine without adapters;
(b) mixed batch: every request of a batch mixing accents and strengths against
    the same request run alone (same engine). --mixed-accents picks the accents
    beside NAME (e.g. adapters with expert factors of different ranks and
    adapters without); default: the first three others;
(c) merged: the NAME adapter at strength 1 against the base + adapter checkpoint
    (lingua accent_adapter --reconstruct, then convert_lingua_checkpoint.py),
    teacher-forced on the adapter's token stream so every step sees the same
    history; the base model on the same streams gives the scale of the
    adapter's whole effect. Skipped without --merged-model.

Requests cycle over the speaker prompts given: none, the raw reference prefix of
--reference-audio and the speaker-embedding row of --speaker-embedding (the
ZONOS3 Mini release LM takes both; its condition rows are left unknown).

Logit differences are max |delta| over all recorded steps (prefill last row and
every decode step, float32 logits after the soft cap). Token agreement is the
fraction of (step, codebook) samples that match; "argmax" compares greedy picks.
Exact mode (the five ZONOS3_USE_TRAINING_* variables) must be set by the caller.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

TEXTS = [
    "The weather turned cold overnight, so we lit the fire early.",
    "Please send the quarterly report to everyone on the finance team before Friday.",
    "She said she would come back tomorrow.",
    "Learning a new language takes patience, practice, and a good sense of humour about "
    "your own mistakes.",
]
MAX_TOKENS = 300


def _requests(accents: list[str], merged_accent: str, mixed: list[str] | None = None,
              prompts: tuple[str, ...] = ("none",)) -> list[dict]:
    """Mixed-batch requests: base rows, the merged adapter at 1.0, others at assorted strengths."""
    others = mixed or [name for name in accents if name != merged_accent][:3] or [merged_accent]
    plan = [
        (0, None, 1.0),
        (1, None, 1.0),
        (2, merged_accent, 1.0),
        (3, others[0], 0.6),
        (0, merged_accent, 1.0),
        (1, others[1 % len(others)], 1.5),
        (2, others[2 % len(others)], 1.0),
    ]
    strengths = (0.7, 1.3, 1.0, 0.6, 1.5)
    plan += [
        (i % len(TEXTS), name, strengths[i % len(strengths)]) for i, name in enumerate(others[3:])
    ]
    # One base row and one merged-adapter row per speaker prompt kind beyond the first two.
    extra = max(0, len(prompts) - 2)
    plan += [(3, None, 1.0)] * extra + [(1, merged_accent, 1.0)] * extra
    # Speaker prompt per request: base rows, merged-adapter rows and the other rows
    # each cycle over the available kinds (none, the raw reference prefix of
    # --reference-audio, the speaker-embedding row of --speaker-embedding), so (a)
    # and (c) cover every kind.
    counters: dict = {}
    reqs = []
    for index, (text, accent, strength) in enumerate(plan):
        group = "base" if accent is None else "merged" if accent == merged_accent else "other"
        kind = prompts[counters.get(group, 0) % len(prompts)]
        counters[group] = counters.get(group, 0) + 1
        reqs.append({"text": TEXTS[text], "accent": accent, "strength": strength,
                     "seed": 1000 + index, "prompt": kind})
    return reqs


# ----------------------------------------------------------------------------- worker
def _worker(spec_path: str) -> None:
    import torch
    from zonos3.message.tts import TTSSamplingParams
    from zonos3.tts.llm import TTSLLM

    spec = json.loads(Path(spec_path).read_text())
    kwargs = dict(
        attention_backend="fa",
        cache_type="naive",
        max_running_req=16,
        cuda_graph_max_bs=16,
        max_extend_tokens=256,
        memory_ratio=0.5,
        decode_audio=False,
    )
    if spec.get("accent_adapters"):
        kwargs["accent_adapters"] = spec["accent_adapters"]
    tts = TTSLLM(model_path=spec["model_path"], **kwargs)

    records: dict[int, dict] = {}
    forced: dict[int, torch.Tensor] | None = None

    def record(uid: int) -> dict:
        return records.setdefault(uid, {"logits": {}, "sampled": {}})

    engine_forward = tts.engine.forward_batch_tts

    def forward(batch):
        logits = engine_forward(batch)
        for row, req in enumerate(batch.reqs):
            record(req.uid)["logits"][req.total_generated] = logits[row].float().cpu()
        return logits

    sample = tts._sample_cfg

    def sample_and_force(logits, sample_args, batch):
        out = sample(logits, sample_args, batch)
        for row, req in enumerate(batch.reqs):
            record(req.uid)["sampled"][req.total_generated] = out[row].cpu()
        if forced is not None:
            rows = [forced[req.uid][req.total_generated] for req in batch.reqs]
            out = torch.stack(rows).to(device=out.device, dtype=out.dtype)
        return out

    tts.engine.forward_batch_tts = forward
    tts._sample_cfg = sample_and_force

    results = {}
    for job in spec["jobs"]:
        records.clear()
        forced = None
        if job.get("force"):
            source = torch.load(job["force"], weights_only=True)
            forced = {
                uid: source[name][index]["sampled"]
                for uid, (name, index) in enumerate(job["force_from"])
            }
        reqs = job["requests"]
        prompt_kinds = [r.get("prompt", "none") for r in reqs]
        extra = {}
        if "reference" in prompt_kinds:
            extra["speaker_reference_audio"] = [
                spec["reference_audio"] if kind == "reference" else None for kind in prompt_kinds]
        if "embedding" in prompt_kinds:
            import numpy as np

            vector = torch.from_numpy(np.load(spec["speaker_embedding"]).astype("float32")).reshape(-1)
            extra["speaker_embedding"] = [vector if kind == "embedding" else None for kind in prompt_kinds]
        tts.generate(
            [r["text"] for r in reqs],
            [TTSSamplingParams(seed=r["seed"]) for r in reqs],
            max_tokens=MAX_TOKENS,
            accent=[r["accent"] for r in reqs] if spec.get("accent_adapters") else None,
            accent_strength=[float(r["strength"]) for r in reqs],
            **extra,
        )
        results[job["name"]] = [
            {
                "logits": torch.stack([rec["logits"][s] for s in sorted(rec["logits"])]),
                "sampled": torch.stack([rec["sampled"][s] for s in sorted(rec["sampled"])]),
            }
            for _, rec in sorted(records.items())
        ]
        print(f"{spec['name']}/{job['name']}: steps {[len(r['sampled']) for r in results[job['name']]]}",
              flush=True)
    torch.save(results, spec["out"])


# ----------------------------------------------------------------------------- driver
def _run(name: str, model_path: str, adapters: str | None, jobs: list[dict], out_dir: Path,
         prompts: dict) -> Path:
    out = out_dir / f"{name}.pt"
    spec = {"name": name, "model_path": model_path, "accent_adapters": adapters, "jobs": jobs,
            "out": str(out), **prompts}
    spec_path = out_dir / f"{name}.json"
    spec_path.write_text(json.dumps(spec, indent=1))
    # A private rendezvous port: shared nodes may run other engines on the default one.
    port = 30000 + (os.getpid() * 3 + len(list(out_dir.glob("*.json")))) % 20000
    env = dict(os.environ, ZONOS3_DISTRIBUTED_ADDR=f"tcp://127.0.0.1:{port}")
    subprocess.run([sys.executable, __file__, "--worker", str(spec_path)], check=True, env=env)
    return out


def _compare(a: dict, b: dict) -> dict:
    """Logit and token agreement of two records over their common steps."""
    import torch

    steps = min(len(a["logits"]), len(b["logits"]))
    la, lb = a["logits"][:steps], b["logits"][:steps]
    sa, sb = a["sampled"][:steps], b["sampled"][:steps]
    n_cb = la.shape[1]
    return {
        "steps": [len(a["sampled"]), len(b["sampled"])],
        "bit_equal": bool(len(a["logits"]) == len(b["logits"]) and torch.equal(la, lb)),
        "max_abs_dlogit": float((la - lb).abs().max()),
        "mean_abs_dlogit": float((la - lb).abs().mean()),
        "token_agreement": float((sa[:, :n_cb] == sb[:, :n_cb]).float().mean()),
        "argmax_agreement": float((la.argmax(-1) == lb.argmax(-1)).float().mean()),
        "tokens_equal": bool(len(a["sampled"]) == len(b["sampled"]) and torch.equal(sa, sb)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--base-model")
    parser.add_argument("--accent-dir")
    parser.add_argument("--merged-model", help="base + adapter checkpoint; omit to skip (c)")
    parser.add_argument("--merged-accent",
                        help="the adapter the merged checkpoint merges (the mixed batch's main accent)")
    parser.add_argument("--out-dir")
    parser.add_argument("--mixed-accents", help="comma-separated accents of the mixed batch")
    parser.add_argument("--reference-audio", help="wav: some requests clone it as the raw reference prefix")
    parser.add_argument("--speaker-embedding",
                        help=".npy speaker vector: some requests take it as the speaker-embedding row")
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
        return
    for flag in ("base_model", "accent_dir", "merged_accent", "out_dir"):
        if not getattr(args, flag):
            parser.error(f"--{flag.replace('_', '-')} is required")

    import torch
    from zonos3.models.accent_adapters import list_accent_adapters

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accents = list_accent_adapters(args.accent_dir)
    mixed = args.mixed_accents.split(",") if args.mixed_accents else None
    for name in [args.merged_accent] + (mixed or []):
        if name not in accents:
            parser.error(f"accent {name} is not in {args.accent_dir}")
    kinds = ("none",) + (("reference",) if args.reference_audio else ()) + (
        ("embedding",) if args.speaker_embedding else ())
    prompts = {"reference_audio": args.reference_audio, "speaker_embedding": args.speaker_embedding}
    reqs = _requests(accents, args.merged_accent, mixed, kinds)
    base_rows = [i for i, r in enumerate(reqs) if r["accent"] is None]
    merged_rows = [i for i, r in enumerate(reqs) if r["accent"] == args.merged_accent]

    jobs = [{"name": "mixed", "requests": reqs}]
    jobs += [{"name": f"solo{i}", "requests": [r]} for i, r in enumerate(reqs)]
    adapted = _run("adapters", args.base_model, args.accent_dir, jobs, out_dir, prompts)
    # The merged-checkpoint requests, teacher-forced on the adapter's solo streams.
    forced = {
        "name": "forced",
        "requests": [dict(reqs[i], accent=None) for i in merged_rows],
        "force": str(adapted),
        "force_from": [[f"solo{i}", 0] for i in merged_rows],
    }
    base_jobs = [{"name": "base", "requests": [reqs[i] for i in base_rows]}]
    if args.merged_model:
        base_jobs.append(forced)
    base = _run("base", args.base_model, None, base_jobs, out_dir, prompts)
    merged = _run("merged", args.merged_model, None, [forced], out_dir, prompts) if args.merged_model else None

    base, adapted = (torch.load(p, weights_only=True) for p in (base, adapted))
    report = {
        "requests": reqs,
        "a_base_rows_vs_no_adapter_server": {
            str(i): _compare(adapted["mixed"][i], base["base"][k]) for k, i in enumerate(base_rows)
        },
        "b_mixed_batch_vs_solo": {
            str(i): _compare(adapted["mixed"][i], adapted[f"solo{i}"][0]) for i in range(len(reqs))
        },
    }
    if merged is not None:
        merged = torch.load(merged, weights_only=True)
        report["c_adapter_strength1_vs_merged_checkpoint"] = {
            str(i): _compare(adapted[f"solo{i}"][0], merged["forced"][k])
            for k, i in enumerate(merged_rows)
        }
        # Scale for (c): the base model against the merged checkpoint on the same streams.
        report["c_reference_base_vs_merged_checkpoint"] = {
            str(i): _compare(base["forced"][k], merged["forced"][k])
            for k, i in enumerate(merged_rows)
        }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "requests"}, indent=1))


if __name__ == "__main__":
    main()

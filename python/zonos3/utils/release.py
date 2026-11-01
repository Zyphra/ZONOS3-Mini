"""hf:// locations for the ZONOS3 Mini release checkpoints (RELEASE_HF_20261021).

Registries, configs and the server's default --model-path may name files and folders in a Hugging Face repo as
``hf://<owner>/<repo>/<path inside the repo>``. ``resolve_file`` / ``resolve_dir_or_local`` fetch them into the local HF cache
(``HF_HOME``; ``HF_TOKEN`` or a stored login is needed while the repo is private) and return local paths; plain local paths pass
through unchanged, so the cluster servers keep running from their local registries. Default repo: ``ZONOS3_RELEASE_REPO``
(Zyphra/ZONOS3_Mini_release_ckpts) and its language model ``lm/mect_60k`` (MECT60K_SWITCH_20261022: was ``lm/fullenc_120k``).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

PREFIX = "hf://"
RELEASE_REPO = os.environ.get("ZONOS3_RELEASE_REPO", "Zyphra/ZONOS3_Mini_release_ckpts")
RELEASE_LM = os.environ.get("ZONOS3_RELEASE_LM", f"{PREFIX}{RELEASE_REPO}/lm/mect_60k")
RELEASE_CODEC_REGISTRY = f"{PREFIX}{RELEASE_REPO}/registries/codec_registry.json"
RELEASE_SPEAKER_REGISTRY = f"{PREFIX}{RELEASE_REPO}/registries/speaker_encoder_registry.json"


def is_hf(path) -> bool:
    return isinstance(path, str) and path.startswith(PREFIX)


def split(uri: str) -> tuple[str, str]:
    """hf://owner/repo/a/b -> ('owner/repo', 'a/b'); hf://owner/repo -> ('owner/repo', '')."""
    if not is_hf(uri):
        raise ValueError(f"not an hf:// location: {uri!r}")
    parts = uri[len(PREFIX):].strip("/").split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"hf:// location needs owner/repo: {uri!r}")
    return f"{parts[0]}/{parts[1]}", (parts[2] if len(parts) > 2 else "")


def hf_join(base, *parts) -> str:
    """Join path parts onto a base that may be an hf:// location (pathlib would collapse the '//')."""
    if is_hf(base):
        return base.rstrip("/") + "".join("/" + str(p).strip("/") for p in parts if str(p))
    return str(Path(str(base), *[str(p) for p in parts]))


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


@lru_cache(maxsize=None)
def resolve_file(path) -> str:
    """Local path for a file: hf:// locations are downloaded into the HF cache (once per process), others pass through."""
    if path is None:
        return path
    path = str(path)
    if not is_hf(path):
        return os.path.expanduser(path)
    repo_id, inside = split(path)
    if not inside:
        raise ValueError(f"hf:// file location needs a path inside the repo: {path!r}")
    from huggingface_hub import hf_hub_download

    print(f"[release] fetching {path}", file=sys.stderr, flush=True)
    return hf_hub_download(repo_id=repo_id, filename=inside, repo_type="model", token=_token())


@lru_cache(maxsize=None)
def resolve_dir_or_local(path) -> str:
    """Local directory for a folder: hf:// locations are snapshot-downloaded (that sub-folder only), others pass through."""
    if path is None:
        return path
    path = str(path)
    if not is_hf(path):
        return os.path.expanduser(path)
    repo_id, inside = split(path)
    from huggingface_hub import snapshot_download

    print(f"[release] fetching folder {path}", file=sys.stderr, flush=True)
    root = snapshot_download(repo_id=repo_id, repo_type="model", token=_token(),
                             allow_patterns=[f"{inside}/*", f"{inside}/**"] if inside else None)
    return str(Path(root) / inside) if inside else root


def resolve_tokenizer_path(path, model_dir) -> str:
    """The text tokenizer file of a checkpoint: hf://, absolute, or relative to the checkpoint folder. A cluster-absolute
    path that is missing here falls back to tokenizer/<basename> next to params.json (the release layout)."""
    if path is None:
        return path
    path = str(path)
    if is_hf(path):
        return resolve_file(path)
    model_dir = Path(str(model_dir)) if model_dir is not None else None
    p = Path(os.path.expanduser(path))
    if p.is_absolute():
        if p.exists() or model_dir is None:
            return str(p)
        alt = model_dir / "tokenizer" / p.name
        return str(alt) if alt.exists() else str(p)
    return str(model_dir / p) if model_dir is not None else str(p)

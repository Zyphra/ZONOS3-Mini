"""hf:// location handling without network access."""

import json
from pathlib import Path

import pytest
from zonos3.utils import release


def test_split_and_join():
    assert release.is_hf("hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/fullenc_120k")
    assert not release.is_hf("/data/x") and not release.is_hf(None)
    assert release.split("hf://Zyphra/R/a/b.json") == ("Zyphra/R", "a/b.json")
    assert release.split("hf://Zyphra/R") == ("Zyphra/R", "")
    assert (
        release.hf_join("hf://Zyphra/R/heads/full_mr", "step_0050000.pt")
        == "hf://Zyphra/R/heads/full_mr/step_0050000.pt"
    )
    assert release.hf_join("/local/campaign", "cfg.json") == str(Path("/local/campaign/cfg.json"))
    with pytest.raises(ValueError):
        release.split("hf://onlyowner")


def test_local_paths_pass_through(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{}")
    assert release.resolve_file(str(f)) == str(f)
    assert release.resolve_dir_or_local(str(tmp_path)) == str(tmp_path)


def test_hf_file_download_is_routed_to_hub(monkeypatch, tmp_path):
    calls = []
    import huggingface_hub

    def fake(repo_id, filename, repo_type, token):
        calls.append((repo_id, filename, repo_type))
        p = tmp_path / Path(filename).name
        p.write_text("ok")
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    release.resolve_file.cache_clear()
    out = release.resolve_file("hf://Zyphra/R/registries/codec_registry_keep2_plain.json")
    assert Path(out).read_text() == "ok" and calls == [
        ("Zyphra/R", "registries/codec_registry_keep2_plain.json", "model")
    ]
    assert (
        release.resolve_file("hf://Zyphra/R/registries/codec_registry_keep2_plain.json") == out and len(calls) == 1
    )  # cached


def test_tokenizer_path_resolution(tmp_path):
    model = tmp_path / "lm"
    (model / "tokenizer").mkdir(parents=True)
    tok = model / "tokenizer" / "superbpe8192.json"
    tok.write_text("{}")
    assert release.resolve_tokenizer_path("tokenizer/superbpe8192.json", model) == str(tok)
    assert release.resolve_tokenizer_path("/cluster/only/superbpe8192.json", model) == str(
        tok
    )  # missing absolute -> release layout
    assert release.resolve_tokenizer_path(str(tok), model) == str(tok)
    assert release.resolve_tokenizer_path("/cluster/only/other.json", model) == "/cluster/only/other.json"


def test_release_registries_are_consistent():
    """The repo registries point at the release repo and keep the sha of the generator checkpoint."""
    root = Path(release.__file__).resolve().parents[1] / "registries"
    codec = json.loads((root / "codec_registry_keep2_plain.json").read_text())
    ((key, e),) = codec.items()
    assert e["checkpoint"].startswith(f"hf://{release.RELEASE_REPO}/autoencoder/generator.pt")
    assert e["checkpoint_sha256"] == "1cedeb12d7d964026c0a16afc32abd1de23dc484533111f24c950775f1941172"
    assert e["decoder"]["campaign"].startswith(f"hf://{release.RELEASE_REPO}/autoencoder/heads/")
    spk = json.loads((root / "speaker_encoder_registry.json").read_text())
    assert spk["ownmect_s2"]["checkpoint"].startswith(f"hf://{release.RELEASE_REPO}/speaker_encoders/ownmect_s2/")
    assert release.RELEASE_LM.endswith("/lm/rate30_20k_r2grpo")

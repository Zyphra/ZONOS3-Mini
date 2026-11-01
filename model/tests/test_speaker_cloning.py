import pytest
from zonos3.models.speaker_cloning import resolve_speaker_embedding_backend


@pytest.mark.parametrize(
    ("codec", "dimension", "expected"),
    [
        ("w2vbert_sv", 256, "w2vbert_sv"),
        ("w2v-bert-sv", 256, "w2vbert_sv"),
        ("qwen3", 2048, "qwen3"),
        (None, 256, "w2vbert_sv"),
        (None, 2048, "qwen3"),
    ],
)
def test_resolve_speaker_embedding_backend(codec, dimension, expected):
    assert resolve_speaker_embedding_backend(codec, dimension) == expected


def test_resolve_speaker_embedding_backend_rejects_unknown_codec():
    with pytest.raises(ValueError, match="Unsupported speaker embedding codec"):
        resolve_speaker_embedding_backend("unknown", 256)


def test_resolve_speaker_embedding_backend_rejects_unknown_legacy_dimension():
    with pytest.raises(ValueError, match="no legacy backend"):
        resolve_speaker_embedding_backend(None, 123)

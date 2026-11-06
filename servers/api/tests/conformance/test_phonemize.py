"""POST /v2/phonemize."""

from __future__ import annotations

from .support import check, error, needs


def test_phonemize(client, caps):
    needs(caps["phonemize_enabled"], "the server has no pronunciation lexicon")
    text = "Hello world, the quick brown fox."
    lang = "en_us" if "en_us" in caps["phonemize_languages"] else caps["phonemize_languages"][0]
    r = client.post("/v2/phonemize", json={"text": text, "language": lang})
    assert r.status_code == 200, r.text[:300]
    check(r.json(), "#/components/schemas/PhonemizeResponse")
    assert "".join(s["text"] for s in r.json()["segments"]) == text


def test_disabled_is_404(client, caps):
    needs(not caps["phonemize_enabled"], "the server has a pronunciation lexicon")
    error(client.post("/v2/phonemize", json={"text": "hello"}), 404)


def test_too_long_is_422(client):
    error(client.post("/v2/phonemize", json={"text": "a" * 5001}), 422)

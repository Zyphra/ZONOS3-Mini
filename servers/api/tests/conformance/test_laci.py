"""LACI requests and reading-position span follow."""

from __future__ import annotations

from .support import error, generate, needs, pcm


def _language(caps) -> str:
    return "en_us" if "en_us" in caps["laci_languages"] else caps["laci_languages"][0]


def test_laci(client, caps):
    needs(caps["laci_languages"], "the server has no LACI heads")
    pcm(generate(client, {"laci": True, "language": _language(caps)}))


def test_laci_with_speaker_cfg_is_400(client, caps, default_voice):
    needs(caps["laci_languages"] and caps["speaker_cfg_supported"], "needs LACI heads and speaker CFG")
    error(
        generate(
            client,
            {
                "laci": True,
                "language": _language(caps),
                "speaker_embedding_id": default_voice["id"],
                "speaker_cfg_scale": 1.5,
            },
        ),
        400,
    )


def test_language_without_heads_is_400(client, caps):
    needs(caps["laci_languages"], "the server has no LACI heads")
    others = [lang for lang in caps["text_norm_languages"] if lang not in caps["laci_languages"]]
    needs(others, "every language has LACI heads")
    error(generate(client, {"laci": True, "language": others[0]}), 400)


def test_reading_follow(client, caps):
    needs(caps["laci_languages"], "the server has no LACI heads")
    r = client.get("/v2/controls")
    needs(r.status_code == 200, "the server has no steering bundle")
    hidden = [d for d, v in r.json()["directions"].items() if "vector" not in v]
    needs(hidden, "the bundle has no hidden-space direction")
    r = generate(
        client,
        {
            "language": _language(caps),
            "control_span_follow": "reading",
            "control_spans": [{"start": 0, "end": 5, "controls": {hidden[0]: 1.0}}],
        },
    )
    pcm(r)
    assert r.headers["x-control-span-follow"] == "reading"

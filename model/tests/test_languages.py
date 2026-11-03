"""The language registry (zonos3.languages, registries/languages.json): request-code normalization and the
cross-references the servers' tables are derived from."""

import pytest
from zonos3 import languages


def test_normalize_accepts_case_and_dashes_and_rejects_others():
    assert languages.normalize("EN-US") == "en_us"
    assert languages.normalize(" cmn ") == "cmn"
    with pytest.raises(ValueError, match="Unsupported language code: 'klingon'. Supported: en_us, en_gb"):
        languages.normalize("klingon")


def test_every_reference_resolves():
    for lang in languages.SERVER:
        assert lang.speech in languages.SPOKEN_BY_CODE, lang
    for lang in languages.SPOKEN:
        assert lang.tts is None or lang.tts in languages.BY_CODE, lang
    assert len(set(languages.CODES)) == len(languages.CODES)
    assert len(languages.SPOKEN_BY_CODE) == len(languages.SPOKEN)

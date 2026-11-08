"""Wrapper-level text normalization: line breaks, per-language pre-passes, sentence-level fallback (2026-10-19)."""

import pytest
from zonos3.tokenizer.textnorm import TTSTextNormalizer, collapse_line_breaks, post_normalize, pre_normalize


def test_collapse_line_breaks_only_touches_runs_with_a_break():
    assert collapse_line_breaks("spend less.\n\nHunt’s Tolan") == "spend less. Hunt’s Tolan"
    assert collapse_line_breaks("Line one.\r\nLine two.") == "Line one. Line two."
    assert collapse_line_breaks("Tabbed.\tNext") == "Tabbed. Next"
    assert collapse_line_breaks("a b") == "a b"
    assert collapse_line_breaks("Three\n\n\nbreaks") == "Three breaks"
    # plain spaces (even doubled) and single-line text are byte-identical
    for s in ["one line", "two  spaces", "  lead", "trail  ", ""]:
        assert collapse_line_breaks(s) == s


def test_pre_pass_en_spoken_conventions():
    assert pre_normalize("We are open 24/7.", "en") == "We are open 24 7 ."  # digit-period spacing as before, then 24 7
    assert pre_normalize("a 50/50 call, an 80/20 split", "en") == "a 50 50 call, an 80 20 split"
    assert pre_normalize("on 9/11", "en") == "on 9 11"
    assert (
        pre_normalize("3/4 of the cake on 10/19", "en") == "3/4 of the cake on 10/19"
    )  # fractions and dates untouched
    assert pre_normalize("Save $20k or $1.5k or £2k.", "en") == "Save $20,000 or $1,500 or £2,000."
    assert pre_normalize("In 1990, he left.", "en") == "In 1990 , he left."  # the digit-punct rule as before


def test_pre_pass_de_period_heuristic():
    assert pre_normalize("Am 3. Oktober 2026 um 10 Uhr.", "de") == "Am 3. Oktober 2026 um 10 Uhr."
    assert pre_normalize("Der 50. Geburtstag.", "de") == "Der 50. Geburtstag."
    assert pre_normalize("Seit 1990. Danach nichts.", "de") == "Seit 1990 . Danach nichts."
    assert pre_normalize("Ich zähle 1, 2, 3.", "de") == "Ich zähle 1 , 2 , 3 ."
    assert pre_normalize("Es geschah am 19.10.2026.", "de") == "Es geschah am 19.10.2026 ."
    assert pre_normalize("Er sagte „1990.“", "de") == "Er sagte „1990 .“"


def test_pre_pass_pt_spaces_punctuation_off_digit_tokens():
    assert pre_normalize("ou 10 %; 1º, 2º, 3º andar.", "pt") == "ou 10 % ; 1º , 2º , 3º andar."
    assert pre_normalize("Ele chegou às 9h.", "pt") == "Ele chegou às 9h ."
    assert pre_normalize("Página 3 (de 10).", "pt") == "Página 3 (de 10 )."
    assert pre_normalize("(10 pessoas) vieram", "pt") == "( 10 pessoas) vieram"
    assert post_normalize("dez por cento ; primeiro , segundo .", "pt") == "dez por cento; primeiro, segundo."
    assert post_normalize("( dez pessoas) vieram", "pt") == "(dez pessoas) vieram"


def test_pre_pass_cjk_removes_spaces_between_cjk_characters():
    assert pre_normalize("値段は5.5ドルです。 次。", "ja") == "値段は5.5ドルです。次。"
    assert pre_normalize("3.5 km、100万 人。", "ja") == "3.5 km、100万人。"
    assert pre_normalize("一月一日。 二月二日。", "zh") == "一月一日。二月二日。"
    assert pre_normalize("Hello 世界", "ja") == "Hello 世界"


class _Boom:
    def normalize(self, text, verbose=False, punct_pre_process=False, punct_post_process=False):
        if "BOOM" in text:
            raise RuntimeError("Operation failed")
        return text.upper()


def test_sentence_level_fallback_keeps_only_the_failing_sentence(monkeypatch):
    tn = TTSTextNormalizer()
    monkeypatch.setattr(tn, "get", lambda lang: _Boom())
    out = tn.normalize("I have 2 cats. BOOM 3 dogs. And 4 birds.", "en_us")
    assert out == "I HAVE 2 CATS. BOOM 3 dogs. AND 4 BIRDS."
    assert tn.normalize("全部BOOM。", "ja") == "全部BOOM。"


def test_unsupported_language_returns_collapsed_text():
    tn = TTSTextNormalizer()
    assert tn.normalize("a.\n\nb", "xx") == "a. b"


pynini = pytest.importorskip("pynini", reason="NeMo grammars need pynini")


@pytest.fixture(scope="module")
def tn():
    return TTSTextNormalizer()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("spend less.\n\nHunt’s Tolan is called Harper.", "spend less. Hunt’s Tolan is called Harper."),
        ("Really?\nYes.", "Really? Yes."),
        ("It costs $5.\nNext day.", "It costs five dollars. Next day."),
        ("We are open 24/7.", "We are open twenty four seven."),
        ("Save $20k.", "Save twenty thousand dollars."),
        ("In 1990, he moved.", "In nineteen ninety, he moved."),
    ],
)
def test_english_grammar_end_to_end(tn, text, expected):
    assert tn.normalize(text, "en_us") == expected


def test_japanese_sentences_separated_by_a_break_normalize(tn):
    assert tn.normalize("値段は5.5ドルです。\n次。", "ja") == "値段は五点五ドルです。次。"


def test_portuguese_percent_before_punctuation_normalizes(tn):
    out = tn.normalize("Custa 10%. Depois 20%, talvez.", "pt_br")
    assert out == "Custa dez por cento. Depois vinte por cento, talvez."


def test_german_ordinal_dates_keep_the_ordinal(tn):
    assert tn.normalize("Am 3. Oktober war es kalt.", "de").startswith("Am dritter oktober")
    assert tn.normalize("Seit 1990. Danach nichts.", "de") == "Seit ein tausend neun hundert neunzig. Danach nichts."


def test_phoneme_groups_are_kept_as_typed_and_the_text_between_normalized(monkeypatch):
    normalizer = TTSTextNormalizer()
    seen = []

    def fake(text, language):
        seen.append(text)
        return text.upper()

    monkeypatch.setattr(normalizer, "_normalize_text", fake)
    text = "I paid 5 \ue000kæt\ue001 for\n\ue000dɔɡ\ue001 food."
    assert normalizer.normalize(text, "en_us") == "I PAID 5 \ue000kæt\ue001 FOR \ue000dɔɡ\ue001 FOOD."
    assert seen == ["I paid 5", "for", "food."]

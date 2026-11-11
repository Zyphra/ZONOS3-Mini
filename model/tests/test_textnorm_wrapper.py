"""Text normalization through the Rust extension (zyphra_textnorm): line breaks, markup, pronunciation groups and
the per-language fixes. Expected strings are the outputs of the pre-port reference pipeline (the oracle of
github.com/Zyphra/zyphra-textnorm)."""

import pytest
import zyphra_textnorm
from zonos3 import languages
from zonos3.tokenizer.textnorm import TTSTextNormalizer, collapse_line_breaks, load_normalizer, normalize_request_text


def test_every_server_language_has_a_grammar():
    assert {lang.nemo for lang in languages.SERVER} <= set(zyphra_textnorm.LANGUAGES)
    assert all(zyphra_textnorm.supported(lang.nemo) for lang in languages.SERVER)


def test_collapse_line_breaks_only_touches_runs_with_a_break():
    assert collapse_line_breaks("spend less.\n\nHunt’s Tolan") == "spend less. Hunt’s Tolan"
    assert collapse_line_breaks("Line one.\r\nLine two.") == "Line one. Line two."
    assert collapse_line_breaks("Tabbed.\tNext") == "Tabbed. Next"
    assert collapse_line_breaks("a b") == "a b"
    assert collapse_line_breaks("Three\n\n\nbreaks") == "Three breaks"
    # plain spaces (even doubled) and single-line text are byte-identical
    for s in ["one line", "two  spaces", "  lead", "trail  ", ""]:
        assert collapse_line_breaks(s) == s


def test_unsupported_language_returns_collapsed_text():
    assert TTSTextNormalizer().normalize("a.\n\nb", "xx") == "a. b"
    # a grammar code is not a server code: "en" is not a request language
    assert TTSTextNormalizer().normalize("It costs $5.\nNext.", "en") == "It costs $5. Next."


def test_normalization_switch(monkeypatch):
    monkeypatch.setenv("ZONOS3_TTS_NORM", "0")
    assert load_normalizer() is None
    assert normalize_request_text("It costs $5.\nNext.", "en_us", load_normalizer()) == "It costs $5. Next."
    monkeypatch.setenv("ZONOS3_TTS_NORM", "1")
    assert normalize_request_text("It costs $5.\nNext.", "en_us", load_normalizer()) == "It costs five dollars. Next."


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
        ("Call me at 123 Main Street, please.", "Call me at one twenty three Main Street, please."),
    ],
)
def test_english_end_to_end(tn, text, expected):
    assert tn.normalize(text, "en_us") == expected


@pytest.mark.parametrize(
    "language,text,expected",
    [
        ("ja", "値段は5.5ドルです。\n次。", "値段は五点五ドルです。次。"),
        ("ja", "値段は5.5ドルです。 次。", "値段は五点五ドルです。次。"),
        ("cmn", "一月一日。 二月二日。", "一月一日。二月二日。"),
        ("pt_br", "Custa 10%. Depois 20%, talvez.", "Custa dez por cento. Depois vinte por cento, talvez."),
        ("pt_br", "Página 3 (de 10).", "Página três (de dez)."),
        ("pt_br", "Ele chegou às 9h.", "Ele chegou às nove horas."),
        ("de", "Seit 1990. Danach nichts.", "Seit ein tausend neun hundert neunzig. Danach nichts."),
    ],
)
def test_other_languages_end_to_end(tn, language, text, expected):
    assert tn.normalize(text, language) == expected


def test_german_ordinal_dates_keep_the_ordinal(tn):
    assert tn.normalize("Am 3. Oktober war es kalt.", "de").startswith("Am dritter oktober")


def test_phoneme_groups_are_kept_as_typed_and_the_text_between_normalized(tn):
    text = "I paid 5 \ue000kæt\ue001 for\n\ue000dɔɡ\ue001 food."
    assert tn.normalize(text, "en_us") == "I paid five \ue000kæt\ue001 for \ue000dɔɡ\ue001 food."
    g = "\ue000həˈloʊ\ue001"
    assert tn.normalize(f"say  {g}  5", "en_us") == f"say  {g}  five"  # whitespace around groups kept exactly


@pytest.mark.parametrize(
    "language,text,expected",
    [
        ("en_us", "So I said <laugh/> no way, $5.", "So I said <laugh/> no way, five dollars."),
        ("en_us", "<shouting>Get out!</shouting>  ok", "<shouting>Get out!</shouting> ok"),
        ("en_us", "no markup 5", "no markup five"),
        ("cmn", "我说<laugh/>不行5", "我说<laugh/>不行五"),
        ("ko", "a <laugh/> <sigh/> b\n<cough/>", "a <laugh/> <sigh/> b <cough/>"),
        ("xx", "keep  <laugh/>  as typed", "keep  <laugh/>  as typed"),
        (
            "en_us",
            "<shouting>Get out!</shouting> It costs $5 <sigh/> today.",
            "<shouting>Get out!</shouting> It costs five dollars <sigh/> today.",
        ),
        ("cmn", "<shouting>出去！</shouting>他花了5元<sigh/>。", "<shouting>出去！</shouting>他花了五元<sigh/>。"),
        ("cmn", "我有3个 <laugh/> 苹果", "我有三个 <laugh/> 苹果"),
        ("ja", "<shouting>出て行け！</shouting>5円です<sigh/>。", "<shouting>出て行け！</shouting>五円です<sigh/>。"),
        # whole-text normalization raised on the ASCII space next to the marker and left "3日" unread
        ("ja", "今日は3日 <breath/> です", "今日は三日 <breath/> です"),
        ("ko", "<shouting>나가!</shouting> 5원이에요 <sigh/>.", "<shouting>나가!</shouting> 오 원이에요 <sigh/>."),
    ],
)
def test_markup_is_never_normalized_and_keeps_its_spacing(tn, language, text, expected):
    assert tn.normalize(text, language) == expected


def test_markup_and_pronunciation_groups_together(tn):
    g = "\ue000həˈloʊ\ue001"
    assert tn.normalize(f"say {g} <laugh/> 5 times", "en_us") == f"say {g} <laugh/> five times"
    assert tn.normalize(f"<shouting>{g} 5</shouting>", "en_us") == f"<shouting>{g} five</shouting>"

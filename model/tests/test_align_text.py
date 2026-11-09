from zonos3.tts import textmatch
from zonos3.tts.align_text import UNSPACED, align_tokens, text_units


def test_text_units_french_merges_dash_and_roundtrips():
    text = "Les requins — existent, bien."
    units = text_units(text, "fr")
    assert [u["text"] for u in units] == ["Les", "requins —", "existent,", "bien."]
    assert "".join(text[a["start"] : b["start"]] for a, b in zip(units, units[1:])) + units[-1]["text"] == text


def test_text_units_leading_punctuation_joins_first_unit():
    units = text_units("— Bonjour monde", "fr")
    assert units[0]["text"] == "— Bonjour"


def test_text_units_chinese_one_per_word():
    text = "鲨鱼确实存在，但很少攻击人类。"
    units = text_units(text, "zh")
    assert [u["text"] for u in units] == ["鲨鱼", "确实", "存在，", "但", "很少", "攻击", "人类。"]
    assert units[-1]["end"] == len(text)


def test_text_units_unspaced_latin_run_is_one_unit():
    units = text_units("我用iPhone15手机", "zh")
    assert [u["text"] for u in units] == ["我", "用", "iPhone15", "手机"]


def test_text_units_non_word_inside_a_run_attaches_to_the_word_before():
    assert [u["text"] for u in text_units("東京・大阪", "ja")] == ["東京・", "大阪"]


def test_align_tokens():
    assert {"ja", "zh"} <= UNSPACED and "en" not in UNSPACED
    assert align_tokens("我用 iPhone15，手机", "zh") == ["我", "用", "iPhone15，", "手机"]
    assert align_tokens("今日は学校に行きます。", "ja") == ["今日", "は", "学校", "に", "行き", "ます。"]
    assert align_tokens("  hello,  brave new\tworld ", "en") == ["hello,", "brave", "new", "world"]


def test_edit_tokens_spell_the_aligned_words():
    """The edit's word diff reads the target's tokens against the aligned words' tokens: for unspaced text they
    must cut the same words, or every word of a CJK line differs."""
    for text, lang in [
        ("我今天去学校，你呢？", "zh"),
        ("「今日は」学校に行きます。", "ja"),
        ("我用 iPhone15，手机", "zh"),
    ]:
        words = align_tokens(text, lang)
        assert textmatch.tokens(text) == [t for w in words for t in textmatch.tokens(w)]
    assert textmatch.tokens("我今天去学校，你呢？") == ["我", "今天", "去", "学校", "你", "呢"]

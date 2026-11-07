from zonos3.tts.align_text import UNSPACED, align_tokens, text_units


def test_text_units_french_merges_dash_and_roundtrips():
    text = "Les requins — existent, bien."
    units = text_units(text, "fr")
    assert [u["text"] for u in units] == ["Les", "requins —", "existent,", "bien."]
    assert "".join(text[a["start"] : b["start"]] for a, b in zip(units, units[1:])) + units[-1]["text"] == text


def test_text_units_leading_punctuation_joins_first_unit():
    units = text_units("— Bonjour monde", "fr")
    assert units[0]["text"] == "— Bonjour"


def test_text_units_chinese_one_per_han_char():
    text = "鲨鱼确实存在，但很少攻击人类。"
    units = text_units(text, "zh")
    assert len(units) == 13
    assert units[5]["text"] == "在，"
    assert units[-1]["text"] == "类。" and units[-1]["end"] == len(text)


def test_text_units_unspaced_latin_run_is_one_unit():
    units = text_units("我用iPhone15手机", "zh")
    assert [u["text"] for u in units] == ["我", "用", "iPhone15", "手", "机"]


def test_align_tokens():
    assert {"ja", "zh"} <= UNSPACED and "en" not in UNSPACED
    assert align_tokens("我用 iPhone15，手机", "zh") == ["我", "用", "iPhone15，", "手", "机"]
    assert align_tokens("こんにちは。", "ja") == ["こ", "ん", "に", "ち", "は。"]
    assert align_tokens("  hello,  brave new\tworld ", "en") == ["hello,", "brave", "new", "world"]

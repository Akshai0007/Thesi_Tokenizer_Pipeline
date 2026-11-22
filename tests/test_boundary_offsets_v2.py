from types import SimpleNamespace

import pytest

from mltok import registry


def evaluate(text, tokens, offsets, *, unit="codepoint", special_ids=(), byte_fallback=True):
    result = SimpleNamespace(ids=list(range(len(tokens))), offsets=offsets, tokens=tokens)
    tokenizer = SimpleNamespace(
        encode=lambda *a, **kw: result,
        get_added_tokens_decoder=lambda: {i: SimpleNamespace(special=True) for i in special_ids},
        model=SimpleNamespace(byte_fallback=byte_fallback),
    )
    return registry.make("metric", {"type": "boundary_offsets_v2", "offset_unit": unit}).compute(
        SimpleNamespace(tok=tokenizer), [(text, "fixture")])


@pytest.mark.parametrize(("unit", "spans"), [
    ("codepoint", [(0, 2), (2, 4)]), ("utf8_byte", [(0, 6), (6, 12)]),
])
def test_conjunct_violation_has_identical_character_and_byte_boundaries(unit, spans):
    r = evaluate("ക്ഷി", ["ക്", "ഷി"], spans, unit=unit)
    assert r["offset_v2_eligible"] == r["offset_v2_akshara_breaks"] == r["offset_v2_egc_breaks"] == 1
    assert r["offset_v2_examples"][0]["character_offset"] == 2
    assert r["offset_v2_examples"][0]["utf8_offset"] == 6


def test_emoji_bytes_do_not_join_malayalam_or_invent_joiner_boundaries():
    text = "ക\u200d👩\u200dക"
    tokens = ["ക", "\u200d", "<0xF0>", "<0x9F>", "<0x91>", "<0xA9>", "\u200d", "ക"]
    r = evaluate(text, tokens, [(0, 3), (3, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 13), (13, 16)],
                 unit="utf8_byte")
    assert r["offset_v2_eligible"] == 0
    assert r["offset_v2_byte_fallback_excluded"] == 5
    assert r["offset_v2_outside_malayalam"] == 2
    assert r["offset_v2_akshara_rate"] is None


def test_mid_scalar_and_noncontiguous_offsets_are_not_boundaries():
    mid = evaluate("കി", ["ക", "ി"], [(0, 2), (2, 6)], unit="utf8_byte")
    gap = evaluate("ക ക", ["ക", "ക"], [(0, 1), (2, 3)])
    assert mid["offset_v2_ambiguous_offsets_excluded"] == gap["offset_v2_ambiguous_offsets_excluded"] == 1
    assert mid["offset_v2_eligible"] == gap["offset_v2_eligible"] == 0


def test_special_detection_uses_registered_ids_not_piece_spelling():
    spans = [(0, 1), (1, 2)]
    ordinary = evaluate("കി", ["<ക>", "ി"], spans)
    special = evaluate("കി", ["<ക>", "ി"], spans, special_ids=(0,))
    assert ordinary["offset_v2_eligible"] == 1
    assert special["offset_v2_special_excluded"] == 1
    assert special["offset_v2_eligible"] == 0


def test_akshara_and_extended_graphemes_can_disagree():
    r = evaluate("ക\u200cക", ["ക", "\u200c", "ക"], [(0, 1), (1, 2), (2, 3)])
    assert r["offset_v2_eligible"] == 2
    assert r["offset_v2_akshara_breaks"] == 0
    assert r["offset_v2_egc_breaks"] == r["offset_v2_references_disagree"] == 1


def test_byte_token_spelling_is_only_excluded_for_fallback_models():
    r = evaluate("കി", ["<0x41>", "ി"], [(0, 1), (1, 2)], byte_fallback=False)
    assert r["offset_v2_byte_fallback_excluded"] == 0
    assert r["offset_v2_eligible"] == 1


def test_backend_without_offsets_reports_unavailable():
    metric = registry.make("metric", "boundary_offsets_v2")
    assert metric.compute(SimpleNamespace(), [("കി", "fixture")]) == {"offset_v2_available": False}

from types import SimpleNamespace

from mltok import registry


def test_audit_uses_real_offsets_and_excludes_overlaps_and_byte_tokens():
    cases = {
        "ക്ഷി": SimpleNamespace(tokens=["ക്", "ഷി"], offsets=[(0, 2), (2, 4)]),
        "ക്ഷ": SimpleNamespace(tokens=["ക്", "ഷ"], offsets=[(0, 2), (1, 3)]),
        "കി": SimpleNamespace(tokens=["ക", "<0xE0>"], offsets=[(0, 1), (1, 2)]),
    }

    def encode(text, *, add_special_tokens):
        assert not add_special_tokens
        return cases[text]

    audit = registry.make("metric", "boundary_audit")
    result = audit.compute(SimpleNamespace(tok=SimpleNamespace(encode=encode)), [(x, "test") for x in cases])
    assert result["boundary_malayalam_boundaries"] == 1
    assert result["boundary_akshara_violations"] == result["boundary_egc_violations"] == 1
    assert result["boundary_akshara_by_kind"] == {"after_virama": 1}
    assert result["boundary_ambiguous_offset_transitions_excluded"] == 1
    assert result["boundary_byte_or_special_transitions_excluded"] == 1
    assert result["boundary_examples"][0]["offset"] == 2

import json

import pytest

from experiments.mom_bpe import scale_data, scale_fit
from experiments.mom_bpe.corpus_materialize import normalizer
from experiments.mom_bpe.corpus_overlap import DuplicateIndex, signatures
from experiments.mom_bpe.diagnose import format_json, sha
from experiments.mom_bpe.scale_proxy_plan import parameter_counts
from mltok import registry


def document(identifier, text, normalize, group=None):
    raw, canonical = text.encode(), normalize(text).encode()
    return {"id": identifier, "group": group or identifier, "src": "authored", "domain": "engineering",
            "text": text, "raw_bytes": len(raw), "sha256": sha(raw), "canonical_bytes": len(canonical),
            "canonical_sha256": sha(canonical)}


def data_plan(out, *, duplicate=False):
    out.mkdir()
    normalize, fingerprint = normalizer()
    train = document("train", "മലയാളം മഴ മഴയിൽ കടലോളം മഴയിൽ മഴ വീടിൽ കുട്ടികൾ " * 20, normalize)
    validation = document("eval", "വീട്ടിൽ മഴയാണ് 🌧️", normalize, group="train" if duplicate else None)
    recipe = {"qualification": scale_data.CAPACITY, "scale": "A", "canonical_target": 3000,
              "scope": scale_data.scope(fingerprint)}
    scale_data.publish(out, recipe, {"train": [train], "validation": [validation]}.__getitem__, normalize)
    return normalize, fingerprint, out / "manifest.json"


def test_prepared_plan_hashes_and_checks_document_groups(tmp_path):
    normalize, fingerprint, path = data_plan(tmp_path / "valid")
    paths, digest = scale_data.verified_plan(path, normalize, fingerprint)
    assert digest == sha(path.read_bytes())
    assert paths["validation"][0].parent == path.parent
    manifest = json.loads(path.read_bytes())
    manifest["splits"]["train"]["canonical_stream_sha256"] = "0" * 64
    path.write_bytes(format_json(manifest))
    with pytest.raises(ValueError, match="accounting"):
        scale_data.verified_plan(path, normalize, fingerprint)

    normalize, fingerprint, path = data_plan(tmp_path / "overlap", duplicate=True)
    with pytest.raises(ValueError, match="connected group"):
        scale_data.verified_plan(path, normalize, fingerprint)


def test_ordered_source_stream_frames_document_boundaries():
    left = [document(str(i), text, str) for i, text in enumerate(["ab", "c"])]
    right = [document(str(i), text, str) for i, text in enumerate(["a", "bc"])]
    stats = [scale_data.split_identity(rows, str) for rows in (left, right)]
    assert stats[0]["canonical_bytes"] == stats[1]["canonical_bytes"] == 3
    assert stats[0]["canonical_stream_sha256"] != stats[1]["canonical_stream_sha256"]


@pytest.mark.parametrize("with_sketch", [False, True])
def test_production_candidate_filter_preserves_exact_near_duplicate_decisions(tmp_path, with_sketch):
    baseline = "abcdefghijklmnopqrstuvwxyzaaaaaaaaaaaaa"
    texts = [baseline + suffix for suffix in ("aa", "ab", "ac")] + [baseline[:22], baseline * 3]
    db = tmp_path / "duplicates.sqlite3"
    ordering = (lambda term: (term.count("a"), term[::-1])) if with_sketch else None
    index = DuplicateIndex(db, feature_order=ordering)
    try:
        sets = []
        for i, text in enumerate(texts):
            with index.db:
                index.insert(document(str(i), text, str), text)
            if i == 1:
                index.db.close()
                index = DuplicateIndex(db, feature_order=ordering)
            sets.append(signatures(text))
        expected = {(i + 1, j + 1) for i, left in enumerate(sets) for j, right in enumerate(sets[:i])
                    if 100 * len(left & right) >= 85 * len(left | right)}
        assert set(index.db.execute("SELECT n,other FROM links WHERE kind='near_duplicate'")) == expected
    finally:
        index.db.close()


def test_underfilled_fit_reuses_verified_result_and_blocks_evaluation(tmp_path, monkeypatch):
    _, _, manifest = data_plan(tmp_path / "plan")
    out = tmp_path / "fit"
    first = scale_fit.run(manifest, out, "T5", fit_only=True)
    assert first["occupied_vocab"] < scale_fit.VOCAB
    assert first["training"]["stop_reason"] == "no_pairs_left"
    assert first["checkpoints"]
    monkeypatch.setattr(registry.plugin_class("algorithm", "picky_bpe"), "train",
                        lambda *_: pytest.fail("Completed fit unexpectedly restarted"))
    assert scale_fit.run(manifest, out, "T5", fit_only=True) == first
    with pytest.raises(ValueError, match="underfilled"):
        scale_fit.run(manifest, out, "T5")
    assert not (out / "validation.json").exists()


@pytest.mark.parametrize(("layers", "heads", "width"), [(7, 7, 448), (12, 12, 768)])
def test_proxy_counts_match_the_project_architecture(layers, heads, width):
    import torch

    registry.load_plugins()
    plugin = registry.make("proxy_model", {"type": "gpt", "n_layer": layers, "n_head": heads, "d_model": width})
    with torch.device("meta"):
        model = plugin.build(48000, 512)
    assert sum(p.numel() for p in model.parameters()) == parameter_counts(layers, width, 512)["total"]

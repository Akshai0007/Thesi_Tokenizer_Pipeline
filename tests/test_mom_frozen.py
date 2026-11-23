import hashlib
import json

import pytest
import yaml

from experiments.mom_bpe import prepare, verify_frozen
from mltok import registry


def digest(data):
    return hashlib.sha256(data).hexdigest()


def jsonl(*rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                   for row in rows).encode()


def member(root, name, data):
    (root / name).write_bytes(data)
    return {"path": name, "sha256": digest(data), "size": len(data)}


def document(identifier, group, text):
    content = text.encode()
    return {"id": identifier, "group": group, "text": text, "domain": "news", "src": "authored",
            "sha256": digest(content), "canonical_sha256": digest(content), "canonical_bytes": len(content)}


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes(json.dumps({"panels": {
        "natural_text": {"target_per_domain": {"news": 1}},
        "orthography": {"target_reviewed_cases": 1},
        "morphology": {"target_reviewed_forms": 1, "target_minimum_groups": 1},
    }}).encode())
    monkeypatch.setattr(verify_frozen, "PROTOCOL", protocol)
    reviewed = ["one", "two"]
    registry.load_plugins()
    normalizer_spec = yaml.safe_load((verify_frozen.HERE / "arms.yaml").read_bytes())["normalizer"]
    manifest = {
        "schema": "mom-frozen-inputs-v1", "protocol_sha256": digest(protocol.read_bytes()), "reviewers": reviewed,
        "normalizer_fingerprint": registry.fingerprint("normalizer", normalizer_spec),
        "sources": {"authored": {"revision": "fixtures-v1", "license": "fixture", "permission_evidence": "authored"}},
        "near_duplicate_review": member(tmp_path, "near.json", b"{}"),
        "train_eval_separation_review": member(tmp_path, "split.json", b"{}"),
        "source_counts": {"train": {"authored": 1}, "eval": {"authored": 1}},
        "splits": {
            "train": [member(tmp_path, "train.jsonl", jsonl(document("train", "first", "മലയാളം")))],
            "eval": [member(tmp_path, "eval.jsonl", jsonl(document("eval", "second", "കുട്ടി")))],
        },
        "orthography_panel": member(tmp_path, "orth.jsonl", jsonl({
            "id": "orth-1", "text": "ക്ഷേത്രം", "category": "conjunct",
            "status": "adjudicated", "reviewers": reviewed,
        })),
        "morphology_panel": member(tmp_path, "morph.jsonl", jsonl({
            "id": "one", "status": "adjudicated", "reviewers": reviewed, "text": "പൂക്കൾ",
            "offsets": [2], "group": "pu",
        })),
        "training_alignments": member(tmp_path, "align.json", json.dumps({
            "schema": "mom-surface-alignments-v1", "qualification": "adjudicated_training",
            "forms": {"കടൽക്കര": [{"group": "kadal", "reviewers": reviewed, "operation": "gemination",
                                  "parts": ["കടൽ", "കര"], "surface_parts": ["കടൽ", "ക്കര"],
                                  "confidence_milli": 750}]},
        }).encode()),
    }
    return manifest


def verify(root, manifest):
    path = root / "manifest.json"
    path.write_bytes(json.dumps(manifest).encode())
    return verify_frozen.verify(path)


def test_freeze_checks_hashed_shards_and_independent_review(tmp_path, frozen):
    assert verify(tmp_path, frozen)["documents"] == {"eval": 1, "train": 1}
    (tmp_path / "train.jsonl").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="SHA-256"):
        verify(tmp_path, frozen)


def test_freeze_rejects_train_evaluation_overlap(tmp_path, frozen):
    frozen["splits"]["train"] = [member(tmp_path, "train.jsonl", jsonl(document("train", "independent", "കുട്ടി")))]
    with pytest.raises(ValueError, match="overlap"):
        verify(tmp_path, frozen)


def test_freeze_rejects_duplicates_with_distinct_ids_in_held_out_documents(tmp_path, frozen):
    frozen["splits"]["eval"] = [member(tmp_path, "eval.jsonl", jsonl(
        document("one", "first", "കുട്ടി"), document("two", "second", "കുട്ടി"),
    ))]
    with pytest.raises(ValueError, match="Duplicate evaluation"):
        verify(tmp_path, frozen)


def test_freeze_rejects_path_escape_and_unaudited_rules(tmp_path, frozen):
    original = frozen["orthography_panel"]
    frozen["orthography_panel"] = {**original, "path": "../orth.jsonl"}
    with pytest.raises(ValueError, match="unsafe"):
        verify(tmp_path, frozen)
    frozen["orthography_panel"] = original
    frozen["training_alignments"] = member(tmp_path, "align.json", json.dumps({
        "schema": "mom-surface-alignments-v1", "qualification": "diagnostic", "forms": {},
    }).encode())
    with pytest.raises(ValueError, match="independently adjudicated"):
        verify(tmp_path, frozen)


def test_plan_pins_the_shared_data_and_morphology_disabled_control(tmp_path, frozen):
    receipt = verify(tmp_path, frozen)
    plan = prepare.prepare(tmp_path / "manifest.json")
    assert plan["freeze"] == receipt and plan["ordered_shards"] == frozen["splits"]
    assert plan["training_started"] is False and plan["vocab"] == 48_000
    off, active = [plan["arms"][name]["algorithm"] for name in ("T3-off", "T3")]
    assert active["morphology_penalty_milli"] == 400
    assert off == {**active, "morphology_penalty_milli": 0}
    assert off["alignment_sha256"] == frozen["training_alignments"]["sha256"]


def test_plan_rejects_a_manifest_replaced_after_verification(tmp_path, frozen, monkeypatch):
    receipt = verify(tmp_path, frozen)

    def replace(path):
        path.write_bytes(b"{}")
        return receipt

    monkeypatch.setattr(prepare, "verify", replace)
    with pytest.raises(ValueError, match="changed after verification"):
        prepare.prepare(tmp_path / "manifest.json")

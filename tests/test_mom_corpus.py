import io
import random

import pytest

from experiments.mom_bpe import corpus_acquire, corpus_build, corpus_materialize, corpus_quality
from experiments.mom_bpe.corpus_overlap import DuplicateIndex, THRESHOLD, prefix, signatures
from experiments.mom_bpe.diagnose import format_json, sha
from mltok.lib.bpe_progress import canonical


def jsonl(rows):
    return b"".join(canonical(row) + b"\n" for row in rows)


def fixture_lid(_path, _digest):
    return lambda _text: {"ml_score": 0.995, "windows": 1, "top_languages": {"ml": 1}, "non_ml_windows": 0}


def test_independent_language_and_historic_unicode_survive_screening():
    text = "മലയാളത്തിലെ പഴയ ലിപിയിൽ ൎ എന്ന ചിഹ്നവും പുൽ എന്ന വാക്കും ഉണ്ടായിരുന്നു."
    metrics = corpus_quality.profile(text, text, fixture_lid(None, None))
    assert metrics["historical_chars"] == 1 and metrics["atomic_chillus"] == 2
    assert corpus_build.provisional_filter(metrics) == []
    metrics = corpus_quality.profile(text + " <div>\ufffd\n\u0d4d\u0d4d</div>", text, fixture_lid(None, None))
    assert set(corpus_build.provisional_filter(metrics)) == {
        "replacement_chars", "unattached_ml_marks", "repeated_virama", "html_hits",
    }
    english = corpus_quality.profile("English only for the script check.", "", fixture_lid(None, None))
    assert english["language"]["ml_score"] > 0.9
    assert corpus_build.provisional_filter(english) == ["low_malayalam_script_share"]


def test_range_rejects_full_and_mismatched_responses_before_reading(monkeypatch):
    class Response:
        def __init__(self, status, header, data):
            self.status_code = status
            self.headers = {"Content-Range": header}
            self.raw = io.BytesIO(data)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    class Session:
        def get(self, *_args, **_kwargs):
            return response

    from io import SEEK_END

    ledger = {"bytes": 0, "ranges": []}
    stream = corpus_acquire.RangeFile(Session(), "https://example.org/file", 9, ledger)
    stream.seek(-3, SEEK_END)
    for code, header in ((200, ""), (206, "bytes 1-3/9")):
        response = Response(code, header, b"abcdefghi")
        with pytest.raises(OSError, match="Unexpected response"):
            stream.read(3)
        assert response.raw.tell() == 0 and ledger["bytes"] == 0
    response = Response(206, "bytes 6-8/9", b"ab")
    with pytest.raises(OSError, match="Truncated"):
        stream.read(3)
    assert ledger["bytes"] == 2 and stream.tell() == 6
    monkeypatch.setattr(corpus_acquire, "CAP", 4)
    with pytest.raises(OSError, match="budget"):
        stream.read(3)
    with corpus_acquire.public_session("test") as client:
        assert not client.trust_env


def test_duplicate_prefix_candidates_match_brute_force(tmp_path):
    rng = random.Random(581)
    index = DuplicateIndex(tmp_path / "dedup.db")
    base = "".join(rng.choices("abcdefghijklmnopqrst", k=260))
    texts = [base[:i] + rng.choice("uvwxyz") + base[i + 1:] for i in range(25, 200, 11)]
    texts.extend([base, base[::-1], base[23:], base + base[::-1]])
    pairs = set()
    with index.db:
        for i, text in enumerate(texts, 1):
            grams = signatures(text)
            for j, old in enumerate(texts[:i - 1], 1):
                old_grams = signatures(old)
                if 100 * len(grams & old_grams) >= THRESHOLD * len(grams | old_grams):
                    assert set(prefix(grams)) & set(prefix(old_grams))
                    pairs.add((i, j))
            digest = sha(text.encode())
            index.insert({"id": str(i), "src": "test", "group": str(i),
                          "sha256": digest, "canonical_sha256": digest}, text)
    assert set(index.db.execute("SELECT n,other FROM links WHERE kind='near_duplicate'")) == pairs
    index.db.close()


def test_authored_corpus_is_idempotent_and_verified(tmp_path):
    a = corpus_materialize.diagnostic(tmp_path, 4096)
    b = corpus_materialize.diagnostic(tmp_path, 4096)
    assert a == b and not a["verified_freeze"]
    groups = []
    for member in a["splits"].values():
        path = tmp_path / member["file"]
        assert sha(path.read_bytes()) == member["sha256"]
        groups.extend(row["group"] for row in corpus_build.capture_rows(path))
    assert len(groups) == len(set(groups))
    (tmp_path / "validation.jsonl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="existing corpus artifact differs"):
        corpus_materialize.diagnostic(tmp_path, 4096)
    with pytest.raises(ValueError, match="Different evidence"):
        corpus_materialize.diagnostic(tmp_path, 2048)


def capture(tmp_path, name, texts):
    rows = [{"src": "fixture", "row_index": key, "text": text,
             "attributes": {"url": f"https://example.org/{key.removesuffix('-variant')}#test"}}
            for key, text in texts.items()]
    blob = jsonl(rows)
    (tmp_path / f"{name}.jsonl").write_bytes(blob)
    manifest = tmp_path / f"{name}.json"
    manifest.write_bytes(format_json({
        "source": "fixture", "shard": name, "repo": "local-fixture", "revision": "test-v1",
        "file": f"{name}.jsonl", "sha256": sha(blob), "bytes": len(blob), "rows": len(rows),
    }))
    return manifest


def member(root, name, data):
    (root / name).write_bytes(data)
    return {"path": name, "sha256": sha(data), "size": len(data)}


def test_index_replays_failed_shard_and_proposal_never_crosses_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_quality, "FastTextLanguage", fixture_lid)
    mock_protocol = {"identity": "fixture-test-only", "panels": {"natural_text": {"target_per_domain": {"news": 1}}}}
    (tmp_path / "evaluation-protocol-v1.json").write_bytes(format_json(mock_protocol))
    monkeypatch.setattr(corpus_build, "AUDIT", tmp_path)
    monkeypatch.setattr(corpus_materialize, "AUDIT", tmp_path)
    texts = {
        "one": "രാവിലെ കാറ്റും മഴയും വന്നപ്പോൾ കുട്ടികൾ കടൽത്തീരത്തേക്കുള്ള യാത്ര മാറ്റിവെച്ചു.",
        "two": "ഗ്രാമത്തിലെ വായനശാല എല്ലാ വെള്ളിയാഴ്ചയും പുതിയ പുസ്തകങ്ങൾ പരിചയപ്പെടുത്തുന്നു.",
        "three": "ഭൂമിയുടെ അന്തരീക്ഷത്തിലുള്ള ജലബാഷ്പം തണുക്കുമ്പോൾ മേഘങ്ങൾ രൂപപ്പെടുന്നു.",
        "four": "പൊതുഗതാഗതത്തിന്റെ സമയക്രമത്തിൽ അടുത്ത ആഴ്ച മുതൽ മാറ്റങ്ങളുണ്ടാകും.",
    }
    same = {"one-variant": "  " + texts["one"] + "\n"}
    first = capture(tmp_path, "first", texts)
    second = capture(tmp_path, "second", same)
    study = tmp_path / "index"
    pin = tmp_path / "lid.json"
    pin.write_bytes(b'{"model_sha256":"synthetic-fixture-only"}')
    failure = {"enabled": True}

    def interrupted_source(path):
        for row in corpus_build.capture_rows(path):
            yield row
            if failure["enabled"] and path.stem == "second":
                raise RuntimeError("interrupted after inserting the second shard row")

    components = corpus_build.Components(source=interrupted_source)
    with pytest.raises(RuntimeError, match="second shard"):
        corpus_build.build([first, second], study, pin, index_only=True, components=components)
    assert not (study / "index-complete.json").exists()
    index = DuplicateIndex(study / "study.sqlite3")
    assert index.db.execute("SELECT count(*) FROM documents").fetchone() == (4,)
    assert index.db.execute("SELECT count(*) FROM settings").fetchone() == (1,)
    index.db.close()
    failure["enabled"] = False
    done = corpus_build.build([first, second], study, pin, index_only=True, components=components)
    assert not done["verified_freeze"]
    decisions = [{"id": f"fixture:{k}", "domain": "news", "sha256": sha(v.encode()),
                  "eligible_splits": list(corpus_materialize.SPLITS), "permission_scope": "authored-test-fixture"}
                 for k, v in {**texts, **same}.items()]
    rights = {
        "schema": "mom-corpus-permissions-v1", "version": "synthetic-test-only",
        "reviewers": ["test-reviewer-a", "test-reviewer-b"],
        "study_recipe_sha256": done["recipe_sha256"],
        "sources": {"fixture": {"revision": "test-v1", "license": "authored test fixture",
                                "local_research_use": True, "attribution": "fixture", "priority": 0,
                                "permission_evidence": member(tmp_path, "terms", b"fixture")}},
        "document_ledger": member(tmp_path, "decisions.jsonl", jsonl(decisions)),
    }
    review = tmp_path / "review.json"
    review.write_bytes(format_json(rights))
    out = tmp_path / "proposal"
    budget = sum(len(t.encode()) for t in texts.values())
    result = corpus_materialize.reviewed_proposal(
        study, review, out, raw_budget=budget, validation_per_domain=1, mode="development",
    )
    assert result["status"] == "REVIEWED_PROPOSAL_ONLY" and not result["verified_freeze"]
    members = [row for f in result["splits"].values() for row in corpus_build.capture_rows(out / f["file"])]
    assert len(members) == len({r["group"] for r in members}) == 4
    assert result["heldout_shortfalls"] == result["validation_shortfalls"] == {"news": 0}
    assert result == corpus_materialize.reviewed_proposal(study, review, out, raw_budget=budget,
                                                           validation_per_domain=1, mode="development")
    rights["reviewers"] = ["test-reviewer-a", "TEST-REVIEWER-A"]
    review.write_bytes(format_json(rights))
    with pytest.raises(ValueError, match="independent reviewers"):
        corpus_materialize.reviewed_proposal(study, review, tmp_path / "bad", raw_budget=budget,
                                             validation_per_domain=1, mode="development")

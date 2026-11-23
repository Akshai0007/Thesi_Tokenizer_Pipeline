"""Run the 48K audit on the pinned original using bounded, CPU-only diagnostics."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "measurement-v2"
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import regex
from tokenizers import Tokenizer

from mltok import registry
from mltok.lib.atomic import atomic_writer
from mltok.lib.backends import HFBackend
from mltok.lib.unicode_ml import CHILLU, SP, VIRAMA, VOWEL_SIGNS, ZWJ, ZWNJ, is_combining, is_ml
from mltok.snapshots import file_identity

BYTE = re.compile(r"<0x([0-9A-F]{2})>\Z")
UNUSED = re.compile(r"<\|unused_\d+\|>\Z")
ML_LETTER = regex.compile(r"[\p{Script=Malayalam}&&\p{L}]", regex.V1)
LEGACY = regex.compile(r"([ണനരലളക])\u0d4d\u200d")
VOWEL_RUN = re.compile(f"[{''.join(VOWEL_SIGNS)}]{{2,}}")


def write(name, data):
    with atomic_writer(OUT / name) as stream:
        stream.write(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False).encode())
        stream.write(b"\n")


def digest(path):
    return file_identity(path)[1]


def versions():
    return {"python": sys.version.split()[0], "unicode": unicodedata.unidata_version,
            **{p: importlib.metadata.version(p) for p in ("tokenizers", "transformers", "regex", "numpy")}}


def lexical_normalizer(serialized):
    norm = serialized["normalizer"]
    steps = norm["normalizers"]
    if steps[-2:] != [
        {"type": "Prepend", "prepend": SP},
        {"type": "Replace", "pattern": {"String": " "}, "content": SP},
    ]:
        raise ValueError("Declared marker transport differs from the audited form")
    copied = {**serialized, "normalizer": {**norm, "normalizers": steps[:-2]}}
    lexical = Tokenizer.from_str(json.dumps(copied)).normalizer

    class DeclaredNormalization:
        def normalize(self, text):
            return lexical.normalize_str(text)

    return DeclaredNormalization()


def audit_vocab(raw, model, specials, norm, frequencies):
    pairs = []

    def object_pairs(rows):
        seen = set()
        for k, _ in rows:
            if k in seen:
                pairs.append(k)
            seen.add(k)
        return dict(rows)

    json.loads(raw, object_pairs_hook=object_pairs)
    vocab = model["vocab"]
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in model["merges"]]
    targets = {a + b for a, b in merges}
    operands = {x for pair in merges for x in pair}
    available = set(vocab) - targets
    errors = defaultdict(list)
    duplicates = Counter(merges)
    for rank, (a, b) in enumerate(merges):
        for category, bad in {
            "missing_operand": a not in vocab or b not in vocab,
            "missing_target": a + b not in vocab,
            "operand_not_available_at_rank": a not in available or b not in available,
        }.items():
            if bad:
                errors[category].append({"rank": rank, "pair": [a, b]})
        available.add(a + b)
    ids = defaultdict(list)
    nfc, equivalent, legacy = defaultdict(list), defaultdict(list), []
    flags = Counter()
    flag_examples = defaultdict(list)
    contextual_cases = []
    lexical_count = 0
    with atomic_writer(OUT / "vocabulary.jsonl") as stream:
        for token, idx in sorted(vocab.items(), key=lambda row: row[1]):
            ids[idx].append(token)
            body = token.removeprefix(SP)
            labels = []
            if token in specials:
                labels.append("special")
            elif BYTE.fullmatch(token):
                labels.append("byte_fallback")
            elif UNUSED.fullmatch(token):
                labels.append("reserved_padding")
            else:
                lexical_count += 1
                nfc[unicodedata.normalize("NFC", token)].append(token)
                equivalent[norm.normalize(token)].append(token)
                if LEGACY.search(token):
                    labels.append("legacy_chillu_sequence")
                    if norm.normalize(token) in vocab:
                        legacy.append({"legacy": token, "atomic_equivalent": norm.normalize(token)})
                if any(c in CHILLU for c in token):
                    labels.append("atomic_chillu_present")
                if unicodedata.normalize("NFC", token) != token:
                    labels.append("non_nfc")
                if body and is_ml(body[0]) and is_combining(body[0]):
                    labels.append("malayalam_combining_start")
                if any(c.isalpha() and not ML_LETTER.fullmatch(c) for c in token):
                    labels.append("other_script_letter")
                    if ML_LETTER.search(token):
                        labels.append("mixed_malayalam_and_other_letter")
                if any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} and c not in "\t\n" + ZWJ + ZWNJ
                       for c in token):
                    labels.append("unexpected_control_or_format")
                if any(unicodedata.category(c) == "Cn" for c in token):
                    labels.append("unassigned_by_python_unicode_version")
                if VIRAMA * 2 in body:
                    labels.append("repeated_virama_probe")
                if any(run.group() not in {"\u0d46\u0d3e", "\u0d47\u0d3e", "\u0d46\u0d57"}
                       for run in VOWEL_RUN.finditer(body)):
                    labels.append("consecutive_vowel_sign_probe")
                if body and body[-1] == VIRAMA:
                    labels.append("virama_final")
            if token in operands and token in targets:
                labels.append("merge_intermediate")
                if not frequencies[idx]:
                    labels.append("merge_intermediate_unobserved")
            if not frequencies[idx]:
                labels.append("unobserved_on_diagnostic_corpus")
            elif frequencies[idx] == 1:
                labels.append("observed_once_on_diagnostic_corpus")
            flags.update(labels)
            for label in labels:
                if len(flag_examples[label]) < 12:
                    flag_examples[label].append({"id": idx, "token": token})
            if token.startswith(SP) and body and body in vocab and len(contextual_cases) < 12:
                contextual_cases.append([token, body])
            stream.write(json.dumps({"id": idx, "token": token, "diagnostic_count": frequencies[idx],
                                     "flags": labels}, ensure_ascii=True).encode() + b"\n")
    groups = {"nfc": [v for v in nfc.values() if len(v) > 1],
              "declared_lexical_normalization": [v for v in equivalent.values() if len(v) > 1]}
    return {
        "denominator": len(vocab), "lexical_or_surface_entries": lexical_count,
        "counts": dict(flags),
        "percentages_of_full_vocab": {k: 100 * v / len(vocab) for k, v in flags.items()},
        "examples": dict(flag_examples), "duplicate_serialized_object_keys": pairs,
        "id_aliases": {idx: v for idx, v in ids.items() if len(v) > 1},
        "missing_ids": sorted(set(range(len(vocab))) - ids.keys()),
        "equivalent_group_counts": {k: len(v) for k, v in groups.items()},
        "equivalent_entry_counts": {k: sum(map(len, v)) for k, v in groups.items()},
        "equivalence_groups": groups, "legacy_atomic_pairs": legacy,
        "intentional_context_examples": contextual_cases,
        "merges": {"count": len(merges), "unique_outputs": len(targets),
                   "duplicate_pair_excess": sum(v - 1 for v in duplicates.values()),
                   "error_counts": {k: len(errors[k]) for k in
                                    ("missing_operand", "missing_target", "operand_not_available_at_rank")},
                   "error_examples": {k: v[:20] for k, v in errors.items()}},
    }


def audit_unicode(tok, norm, project, cases):
    rows = []
    for name, text in cases:
        normalized = norm.normalize(text)
        enc = tok.encode(text, add_special_tokens=False)
        dec = tok.decode(enc.ids, skip_special_tokens=False)
        transported = tok.normalizer.normalize_str(text)
        rows.append({
            "case": name, "input": text, "input_codepoints": [f"U+{ord(c):04X}" for c in text],
            "nfc": unicodedata.normalize("NFC", text), "lexical_normalized": normalized,
            "normalizer_transport_output": transported, "lexical_idempotent": norm.normalize(normalized) == normalized,
            "lexical_output_is_nfc": unicodedata.is_normalized("NFC", normalized),
            "project_normalizer_matches": project.normalize(text) == normalized,
            "tokens": enc.tokens, "ids": enc.ids, "input_offsets": enc.offsets, "decoded": dec,
            "raw_exact": dec == text, "normalized_exact": dec == normalized,
        })
    malformed = []
    for name, text in [("isolated_high_surrogate", "\ud800"), ("isolated_low_surrogate", "\udfff"),
                       ("embedded_surrogate", "ക\ud800ക")]:
        try:
            tok.encode(text, add_special_tokens=False)
        except (TypeError, ValueError, UnicodeError) as exc:
            malformed.append({"case": name, "outcome": "rejected", "error_type": type(exc).__name__})
        else:
            malformed.append({"case": name, "outcome": "accepted"})
    try:
        b"\xe0\xb4".decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        malformed.append({"case": "truncated_utf8_before_tokenizer", "outcome": "rejected_by_strict_decode"})
    return {"cases": rows, "malformed": malformed, "counts": {
        k: sum(r[k] for r in rows) for k in ("lexical_idempotent", "lexical_output_is_nfc",
                                             "project_normalizer_matches", "raw_exact", "normalized_exact")}}


def audit_morphology(tok, norm, entries):
    rows, counts = [], Counter()
    groups = defaultdict(list)
    analyzer = None
    analyzer_state = {"available": importlib.util.find_spec("mlmorph") is not None}
    if analyzer_state["available"]:
        try:
            from mlmorph import Analyser

            analyzer = Analyser()
            analyzer_state["version"] = importlib.metadata.version("mlmorph")
            analyzer_state["sfst_version"] = importlib.metadata.version("sfst")
        except Exception as exc:
            analyzer_state["error_type"] = type(exc).__name__
    for item in entries:
        parts = item["parts"]
        word = item.get("word", "".join(parts))
        enc = tok.encode(word, add_special_tokens=False)
        raw_boundaries = {r[0] for l, r in zip(enc.offsets, enc.offsets[1:])
                          if 0 <= l[0] < l[1] == r[0] < r[1] <= len(word)}
        status = item.get("alignment", "manual_surface_fixture")
        scored = (status != "unscored_sandhi" and len(parts) > 1 and "".join(parts) == word
                  and norm.normalize(word) == word and not any(BYTE.fullmatch(t) for t in enc.tokens))
        offsets = [sum(map(len, parts[:i])) for i in range(1, len(parts))] if scored else []
        hits = sum(off in raw_boundaries for off in offsets)
        key = "exploratory" if status == "surface_hypothesis" else "manual"
        if scored:
            counts[f"{key}_scored_words"] += 1
            counts[f"{key}_reference_boundaries"] += len(offsets)
            counts[f"{key}_matched_boundaries"] += hits
            if key == "manual":
                counts["manual_predicted_boundaries"] += len(raw_boundaries)
                first_end = len(parts[0])
                first = tuple(t for t, (a, b) in zip(enc.tokens, enc.offsets) if a < b <= first_end)
                groups[(item["group"], parts[0])].append(
                    {"word": word, "stem_span_isolatable": first_end in raw_boundaries, "stem_tokens": list(first)})
        if status == "unscored_sandhi":
            counts["unscored_sandhi_or_gemination"] += 1
        row = {**item, "word": word, "status": status, "scored": scored, "reference_offsets": offsets,
               "matched_reference_offsets": [off for off in offsets if off in raw_boundaries],
               "token_offsets": sorted(raw_boundaries), "tokens": enc.tokens, "offsets": enc.offsets}
        if analyzer is not None:
            try:
                suggestions = analyzer.analyse(word)
                row["mlmorph_analysis_no_surface_offsets"] = list(suggestions or [])[:5]
                counts["mlmorph_analyzed_cases"] += bool(suggestions)
                counts["mlmorph_ambiguous_cases"] += len(suggestions or []) > 1
            except Exception as exc:
                row["mlmorph_error_type"] = type(exc).__name__
        rows.append(row)
    repeated = []
    for (group, stem), forms in groups.items():
        if len(forms) > 1:
            isolatable = [f for f in forms if f["stem_span_isolatable"]]
            repeated.append({"group": group, "exact_surface_stem": stem, "forms": forms,
                             "all_isolatable": len(isolatable) == len(forms),
                             "consistent_if_all_isolatable": (len({tuple(f["stem_tokens"]) for f in forms}) == 1
                                                              if len(isolatable) == len(forms) else None)})
    prec = counts["manual_matched_boundaries"] / max(counts["manual_predicted_boundaries"], 1)
    recall = counts["manual_matched_boundaries"] / max(counts["manual_reference_boundaries"], 1)
    return {"annotation_kind": "author-specified surface fixtures, not adjudicated gold",
            "mlmorph": analyzer_state, "counts": dict(counts), "manual_precision": prec, "manual_recall": recall,
            "manual_f1": 2 * prec * recall / (prec + recall) if prec + recall else 0,
            "repeated_surface_stem_groups": repeated, "cases": rows}


def audit_hf(path, tok, docs, specials):
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling

    hf = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    enc = tok.encode_batch(docs, add_special_tokens=False)
    expected = [e.ids for e in enc]
    observed = hf(docs, add_special_tokens=False, truncation=False)["input_ids"]
    ids = {role: {"token": getattr(hf, f"{role}_token"), "id": getattr(hf, f"{role}_token_id")}
           for role in ("pad", "bos", "eos", "unk")}
    restored = Tokenizer.from_str(tok.to_str())
    reloaded = [e.ids for e in restored.encode_batch(docs, add_special_tokens=False)]
    batch = DataCollatorForLanguageModeling(hf, mlm=False)(
        [{"input_ids": x} for x in observed[:3]])
    padded = batch["attention_mask"] == 0
    padded_ok = bool((batch["labels"][padded] == -100).all())
    data_ok = bool((batch["labels"][~padded] == batch["input_ids"][~padded]).all())
    fast_vocab, raw_vocab = hf.get_vocab(), tok.get_vocab()
    return {"autotokenizer_class": type(hf).__name__, "fast": hf.is_fast,
            "vocab_id_mismatch_count": sum(fast_vocab.get(t) != i for t, i in raw_vocab.items())
                                       + len(fast_vocab.keys() - raw_vocab.keys()),
            "special_ids": ids, "model_special_inventory_count": len(specials),
            "batch_ids_exact": observed == expected,
            "batch_decode_exact": hf.batch_decode(observed, skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=False)
                                  == [tok.decode(x, skip_special_tokens=False) for x in expected],
            "serializer_reload_ids_exact": reloaded == expected,
            "clm_collator_batch_shape": list(batch["input_ids"].shape),
            "clm_padding_labels_masked": padded_ok, "clm_tokens_equal_labels": data_ok,
            "chat_template_renders": "<|im_start|>assistant" in hf.apply_chat_template(
                [{"role": "user", "content": "സുഖമാണോ?"}], tokenize=False, add_generation_prompt=True)}


def audit_boundary_causes(tok, serialized, docs):
    """Trace the current tokenizer's decisions without editing merges or training."""
    from mltok.lib.unicode_ml import akshara_starts

    model = serialized["model"]
    ranks = {tuple(pair.split(" ") if isinstance(pair, str) else pair): i
             for i, pair in enumerate(model["merges"])}
    found, historical_indirect = [], []
    pre = tok.pre_tokenizer
    for text, source in docs:
        enc = tok.encode(text, add_special_tokens=False)
        starts = akshara_starts(text)
        transported = tok.normalizer.normalize_str(text)
        segments = pre.pre_tokenize_str(transported)
        pre_boundaries = {end for _, (_, end) in segments}
        def ordinary(token):
            return not BYTE.fullmatch(token) and not (token.startswith("<") and token.endswith(">"))

        for i, (left, right) in enumerate(zip(enc.tokens, enc.tokens[1:])):
            a, pos = enc.offsets[i]
            start, b = enc.offsets[i + 1]
            if (ordinary(left) and ordinary(right) and 0 <= a < pos == start < b <= len(text)
                    and is_ml(text[pos - 1]) and is_ml(text[pos]) and pos not in starts):
                expected_transport = SP + text.replace(" ", SP)
                found.append({"source": source, "text": text, "raw_offset": pos, "tokens": enc.tokens,
                              "token_pair": [left, right], "left_ends_in_virama": left.endswith(VIRAMA),
                              "right_begins_with_combining_mark": is_combining(right[0]),
                              "pair_merge_rank": ranks.get((left, right)),
                              "pair_concatenation_in_vocab": left + right in model["vocab"],
                              "inside_pretokenized_piece": (pos + 1 not in pre_boundaries
                                                           if transported == expected_transport else None)})

        keep = [i for i, t in enumerate(enc.tokens) if ordinary(t)]
        joined = "".join(enc.tokens[i] for i in keep)
        legacy_starts = akshara_starts(joined)
        pos = 0
        for left_idx, right_idx in zip(keep, keep[1:]):
            pos += len(enc.tokens[left_idx])
            if not (0 < pos < len(joined) and is_ml(joined[pos - 1]) and is_ml(joined[pos])
                    and pos not in legacy_starts):
                continue
            left_end, right_start = enc.offsets[left_idx][1], enc.offsets[right_idx][0]
            if (right_idx != left_idx + 1 or left_end != right_start or left_end in starts):
                historical_indirect.append({"source": source, "text": text,
                                            "token_slice": enc.tokens[left_idx:right_idx + 1],
                                            "offset_slice": enc.offsets[left_idx:right_idx + 1],
                                            "synthetic_surface_offset": pos,
                                            "discontinuous_or_different_reference": True})
    return {"exact_offset_violations": found, "legacy_only_or_indirect_examples": historical_indirect}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if not OUT.is_relative_to(HERE) or OUT == HERE or (OUT / "COMPLETE.json").exists():
        raise SystemExit("Select a fresh audit output subdirectory")
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = json.loads((HERE / "baseline-artifacts.json").read_bytes())
    path = HERE / artifact["directory"]
    for entry in artifact["artifacts"]:
        if file_identity(path / entry["path"]) != (entry["size"], entry["sha256"]):
            raise RuntimeError("Pinned artifact integrity check failed")
    raw = (path / "tokenizer.json").read_bytes()
    serial = json.loads(raw)
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    norm = lexical_normalizer(serial)
    registry.load_plugins()
    project = registry.make("normalizer", "ml")
    lexer = HFBackend(tok, norm)
    specials = {s["content"] for s in serial["added_tokens"] if s["special"]}
    fixtures = json.loads((HERE / "fixtures.json").read_bytes())
    docs = [(text, label) for label, items in fixtures["domains"].items() for text in items]
    normalized = [(norm.normalize(text), label) for text, label in docs]
    profile = {
        "artifact_revision": artifact["revision"], "tokenizer_sha256": hashlib.sha256(raw).hexdigest(),
        "fixture_sha256": digest(HERE / "fixtures.json"), "script_sha256": digest(Path(__file__)),
        "metric_sources": {name: digest(Path(registry.plugin_class("metric", name)._plugin_file)) for name in
                           ("basic", "structure", "roundtrip", "stress", "boundary_audit")},
        "shared_boundary_source": digest(ROOT / "mltok/lib/unicode_ml.py"), "versions": versions(),
        "diagnostic_corpus_kind": fixtures["corpus_kind"], "document_count": len(docs),
        "boundary_reference": {"akshara": "pipeline regex", "grapheme": "installed regex \\X",
                               "eligibility": "consecutive non-byte, non-special tokens with a shared exact offset whose two adjacent characters satisfy is_ml"},
        "normalization_reference": "actual serialized lexical steps, excluding the verified marker transport suffix",
    }
    write("configuration.json", profile)
    write("tokenizer-configuration.json", {
        "repository": artifact["repository"], "revision": artifact["revision"], "vocab_size": tok.get_vocab_size(),
        "model": {k: v for k, v in serial["model"].items() if k not in ("vocab", "merges")},
        "merge_count": len(serial["model"]["merges"]), "first_merges": serial["model"]["merges"][:8],
        **{k: serial[k] for k in ("normalizer", "pre_tokenizer", "decoder", "post_processor",
                                  "truncation", "padding", "added_tokens")},
    })
    counts = Counter(i for row in lexer.encode_batch([t for t, _ in docs]) for i in row)
    vocab = audit_vocab(raw, serial["model"], specials, norm, counts)
    write("vocabulary-analysis.json", vocab)
    unicode_result = audit_unicode(tok, norm, project, fixtures["unicode_cases"])
    write("unicode-analysis.json", unicode_result)
    morph = audit_morphology(tok, norm, fixtures["morphology"])
    write("morphology-analysis.json", morph)
    boundary = registry.make("metric", "boundary_audit")
    norm_boundary = boundary.compute(lexer, normalized)
    raw_boundary = boundary.compute(lexer, docs)
    write("boundary-analysis.json", {"raw": raw_boundary, "normalized": norm_boundary,
                                    "unicode_fixture_raw": boundary.compute(
                                        lexer, [(text, label) for label, text in fixtures["unicode_cases"]]),
                                    "unicode_fixture_normalized": boundary.compute(
                                        lexer, [(norm.normalize(text), label) for label, text in fixtures["unicode_cases"]]),
                                    "historical_definition_on_diagnostic_only": registry.make(
                                        "metric", "structure").compute(lexer, docs)})
    write("boundary-root-causes.json", audit_boundary_causes(tok, serial, normalized))
    basic, roundtrip, stress = (registry.make("metric", name) for name in ("basic", "roundtrip", "stress"))
    by_domain = {}
    for label, texts in fixtures["domains"].items():
        subset = [(t, label) for t in texts]
        d = boundary.compute(lexer, [(norm.normalize(t), label) for t in texts])
        by_domain[label] = {**basic.compute(lexer, subset), **roundtrip.compute(lexer, subset),
                            **{k: d[k] for k in ("boundary_malayalam_boundaries", "boundary_akshara_violations",
                                                  "boundary_egc_violations")}}
    write("domain-analysis.json", by_domain)

    bench = [t for t, _ in docs] * 16
    ids = lexer.encode_batch(bench)
    tok.decode_batch(ids, skip_special_tokens=False)
    encode_samples, decode_samples = [], []
    tokens = sum(map(len, ids))
    for _ in range(5):
        started = time.perf_counter()
        lexer.encode_batch(bench)
        encode_samples.append(tokens / (time.perf_counter() - started))
        started = time.perf_counter()
        tok.decode_batch(ids, skip_special_tokens=False)
        decode_samples.append(tokens / (time.perf_counter() - started))
    long_text = ("ഇത് ഒരു ദീർഘ മലയാളം പരിശോധനയാണ്. " * 4096).rstrip()
    long_ids = lexer.encode_batch([long_text])[0]
    hf = audit_hf(path, tok, [t for t, _ in docs] + [t for _, t in fixtures["unicode_cases"]], specials)
    write("hf-compatibility.json", hf)
    result = {
        "corpus": profile["diagnostic_corpus_kind"], "measurement": basic.compute(lexer, docs),
        "normalized_fidelity": roundtrip.compute(lexer, docs), "stress": stress.compute(lexer, docs),
        "raw_exact_docs": sum(lexer.decode(lexer.encode_batch([t])[0]) == t for t, _ in docs),
        "throughput": {"unit": "tokens/second", "warmup_batches": 1, "docs_per_batch": len(bench),
                       "encode_samples": encode_samples, "decode_samples": decode_samples,
                       "encode_median": statistics.median(encode_samples), "decode_median": statistics.median(decode_samples)},
        "long_input": {"chars": len(long_text), "bytes": len(long_text.encode()), "tokens": len(long_ids),
                       "normalized_exact": lexer.decode(long_ids) == norm.normalize(long_text)},
        "original_integrity_rechecked": all(file_identity(path / e["path"]) == (e["size"], e["sha256"])
                                           for e in artifact["artifacts"]),
    }
    write("measurements.json", result)
    with atomic_writer(OUT / "diagnostic-tokenizations.jsonl") as out:
        for text, source in docs:
            row = tok.encode(text, add_special_tokens=False)
            out.write(json.dumps({"text": text, "source": source, "ids": row.ids, "tokens": row.tokens,
                                  "offsets": row.offsets}, ensure_ascii=False).encode() + b"\n")
    files = []
    for output in sorted(OUT.iterdir()):
        if output.is_file():
            size, sha256 = file_identity(output)
            files.append({"path": output.name, "size": size, "sha256": sha256})
    write("COMPLETE.json", {"artifact_revision": artifact["revision"], "files": files})
    print(json.dumps({"vocab": vocab["denominator"], "docs": len(docs),
                      "fertility": result["measurement"]["fertility"],
                      "akshara_exact_offset": norm_boundary["boundary_akshara_rate"],
                      "egc_exact_offset": norm_boundary["boundary_egc_rate"],
                      "raw_exact_docs": result["raw_exact_docs"], "unicode_counts": unicode_result["counts"],
                      "huggingface": hf}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

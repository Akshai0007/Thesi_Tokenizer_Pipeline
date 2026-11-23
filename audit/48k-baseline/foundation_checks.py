"""Repeat pinned, CPU-only metric and fidelity probes without editing the original."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from diagnose import HERE, ROOT, lexical_normalizer
from tokenizers import Tokenizer

from mltok import registry
from mltok.lib.atomic import atomic_writer
from mltok.lib.backends import HFBackend
from mltok.snapshots import file_identity


def main():
    artifact = json.loads((HERE / "baseline-artifacts.json").read_bytes())
    root = HERE / artifact["directory"]
    for row in artifact["artifacts"]:
        if file_identity(root / row["path"]) != (row["size"], row["sha256"]):
            raise SystemExit("Original tokenizer does not match the audit manifest")
    original = root / "tokenizer.json"
    serial = json.loads(original.read_bytes())
    tok = Tokenizer.from_file(str(original))
    norm = lexical_normalizer(serial)
    spec = HERE / "fidelity-cases.json"
    fixtures = json.loads((HERE / "fixtures.json").read_bytes())
    registry.load_plugins()
    backend = HFBackend(tok, norm)
    v1 = registry.make("metric", "boundary_audit")
    v2 = registry.make("metric", "boundary_offsets_v2")
    v2_bytes = registry.make("metric", {"type": "boundary_offsets_v2", "offset_unit": "utf8_byte"})
    domains = [(s, label) for label, rows in fixtures["domains"].items() for s in rows]
    unicode = [(s, case) for case, s in fixtures["unicode_cases"]]

    def byte_offsets(text, **kwargs):
        enc = tok.encode(text, **kwargs)
        native = [0]
        for char in text:
            native.append(native[-1] + len(char.encode("utf-8")))
        return SimpleNamespace(ids=enc.ids, tokens=enc.tokens,
                               offsets=[(native[a], native[b]) for a, b in enc.offsets])

    converted = SimpleNamespace(tok=SimpleNamespace(
        encode=byte_offsets, get_added_tokens_decoder=tok.get_added_tokens_decoder, model=tok.model))
    summaries = {}
    for name, docs in (("ordinary", domains), ("unicode_probes", unicode)):
        for input_view, rows in (
            ("raw", docs), ("canonical", [(norm.normalize(t), label) for t, label in docs]),
        ):
            chars = v2.compute(backend, rows)
            bytes_ = v2_bytes.compute(converted, rows)
            exclude = {"offset_v2_native_unit", "offset_v2_examples"}
            if {k: v for k, v in chars.items() if k not in exclude} != {
                    k: v for k, v in bytes_.items() if k not in exclude}:
                raise AssertionError("Codepoint and UTF-8-byte counts differ")
            summaries[f"{name}_{input_view}"] = {"v2": chars, "equivalent_explicit_utf8_offsets": True}
    ordinary = summaries["ordinary_raw"]["v2"]
    if (ordinary["offset_v2_eligible"], ordinary["offset_v2_akshara_breaks"],
            ordinary["offset_v2_egc_breaks"], ordinary["offset_v2_akshara_after_virama"]) != (133, 6, 6, 6):
        raise AssertionError("Pinned boundary outcome changed")
    legacy = registry.make("metric", "structure").compute(backend, domains)
    fidelity = []
    for case in json.loads(spec.read_bytes())["cases"]:
        text = case["text"]
        canonical = norm.normalize(text)
        enc = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(enc.ids, skip_special_tokens=False)
        row = {**case, "canonical": canonical, "decoded": decoded, "tokens": enc.tokens, "ids": enc.ids,
               "raw_equal": decoded == text, "normalized_equal": decoded == canonical}
        for expected, observed in (("raw", "raw_equal"), ("normalized", "normalized_equal"), ("observed", "decoded")):
            if expected in case and case[expected] != row[observed]:
                raise AssertionError(f"Fidelity probe changed: {case['id']} / {expected}")
        if "registered_special" in case:
            identifier = tok.token_to_id(case["registered_special"])
            row["recognized_registered_id_with_add_special_tokens_false"] = identifier in enc.ids
            if not row["recognized_registered_id_with_add_special_tokens_false"]:
                raise AssertionError(f"Expected registered-token match: {case['id']}")
        fidelity.append(row)
    result = {
        "baseline_revision": artifact["revision"], "baseline_sha256": file_identity(original)[1],
        "fixture_sha256": file_identity(spec)[1], "boundary_reference_sha256": file_identity(
            ROOT / "mltok/lib/unicode_ml.py")[1],
        "legacy_structure_on_authored_ordinary_fixtures": legacy,
        "boundary_audit_v1_on_authored_ordinary_fixtures": {
            k: v for k, v in v1.compute(backend, domains).items() if k != "boundary_examples"},
        "boundary_offsets_v2": summaries,
        "fidelity_baseline_observations": fidelity,
        "protocol_sha256": hashlib.sha256((HERE / "evaluation-protocol-v1.json").read_bytes()).hexdigest(),
    }
    output = HERE / "foundation-evidence.json"
    content = (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode()
    if output.exists():
        if content != output.read_bytes():
            raise SystemExit("Existing foundation evidence differs; select a new artifact version")
    else:
        with atomic_writer(output) as stream:
            stream.write(content)
    print(json.dumps({"ordinary_boundaries": ordinary["offset_v2_eligible"],
                      "akshara_breaks": ordinary["offset_v2_akshara_breaks"],
                      "egc_breaks": ordinary["offset_v2_egc_breaks"],
                      "fidelity_probes": len(fidelity)}))


if __name__ == "__main__":
    main()

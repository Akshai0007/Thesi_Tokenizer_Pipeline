"""Compare verified 48K fits on common canonical byte positions."""

import argparse
import json
import re
from collections import Counter
from itertools import accumulate, combinations
from pathlib import Path
from statistics import median
from time import perf_counter

from experiments.mom_bpe.diagnose import format_json, immutable, sha
from experiments.mom_bpe.performance import checked_canonical
from experiments.mom_bpe.scale_data import code_digest
from experiments.mom_bpe.scale_fit import checked_run
from experiments.mom_bpe.scale_mlbpe import MatchedMLBPE
from experiments.mom_bpe.verify_frozen import load_rows
from mltok.lib.bpe_progress import canonical
from mltok.lib.orthographic_units import UNITIZERS
from mltok.lib.picky_backend import ARTIFACT, EventBackend
from mltok.lib.unicode_ml import ML_SET
from mltok.snapshots import _lock, file_identity

BYTE = re.compile(r"<0x([0-9A-F]{2})>")


def token_payloads(encoding, model, native):
    """Reconstruct token payloads, including native metaspace decoder semantics."""
    start = 0
    trim = native and bool(encoding.ids)
    for identifier, piece in zip(encoding.ids, encoding.tokens, strict=True):
        fallback = identifier in model.fallback_ids
        if fallback:
            match = BYTE.fullmatch(piece)
            if not match:
                raise ValueError("An emitted fallback ID has an invalid byte spelling")
            payload = bytes([int(match[1], 16)])
        else:
            payload = (piece.replace("▁", " ") if native else piece).encode()
        if trim and payload:
            payload = payload.removeprefix(b" ")
            trim = False
        end = start + len(payload)
        yield start, end, identifier, piece, fallback, payload
        start = end


def category(payload, fallback):
    if fallback:
        return "byte_fallback"
    text = payload.decode()
    if not text:
        return "empty_decoder_prefix"
    if text.isspace():
        return "whitespace_only"
    if any(c.isspace() for c in text):
        return "whitespace_with_other"
    if any(c.isalpha() for c in text):
        mal = any(c in ML_SET for c in text)
        other = any(c.isalpha() and c not in ML_SET for c in text)
        return "mixed_letter_scripts" if mal and other else "malayalam_lexical" if mal else "other_lexical"
    return "punctuation_digits_other"


def site_reference(text):
    scalar = list(accumulate((len(c.encode()) for c in text), initial=0))
    refs = {}
    for name in ("akshara", "grapheme"):
        boundaries = set(accumulate(map(len, UNITIZERS[name](text))))
        refs[name] = {scalar[i] for i in range(1, len(text)) if i not in boundaries
                      and (name != "akshara" or text[i - 1] in ML_SET and text[i] in ML_SET)}
    return refs, set(scalar)


def load_fit(name, folder, digest, eval_sha):
    report = json.loads((folder / "validation.json").read_bytes())
    claimed = report.pop("result_sha256")
    fit = report["fit"]
    if (sha(canonical(report)) != claimed or fit["occupied_vocab"] != 48000
            or fit["identity"]["arm"] != name
            or fit["identity"]["manifest_sha256"] != digest
            or fit["identity"]["validation_stream_sha256"] != eval_sha):
        raise ValueError(f"{name} does not describe this verified 48K fit")
    native = "files_sha256" in fit
    if native:
        if fit["files_sha256"]["tokenizer.json"] != file_identity(folder / "tokenizer.json")[1]:
            raise ValueError("Native tokenizer changed")
        model = MatchedMLBPE(folder)
        specification = json.loads((folder / "tokenizer.json").read_bytes())["model"]
        merges = specification["merges"]
        types = {"initial_symbols": len(specification["vocab"]) - len(merges)
                 - len(model.special_ids) - len(model.fallback_ids),
                 "merges": len(merges), "removals": 0, "protected_atoms": None}
    else:
        if fit["model_sha256"] != file_identity(folder / ARTIFACT)[1]:
            raise ValueError("Event tokenizer changed")
        model = EventBackend.load(folder)
        events = Counter(row[0] for row in model.artifact["events"])
        types = {"initial_symbols": len(model.artifact["initial"]), "protected_atoms": len(model.artifact["atoms"]),
                 "merges": events["merge"], "removals": events["remove"]}
    return model, native, claimed, report, types


def paired_stats(values):
    return {"difference_sum": sum(values), "median": median(values), "min": min(values), "max": max(values),
            "fewer": sum(n < 0 for n in values), "more": sum(n > 0 for n in values), "ties": values.count(0)}


def analyze(manifest, candidates, out, *, trace_count=0):
    if trace_count < 0:
        raise ValueError("Trace count must be nonnegative")
    plan, digest, files, normalizer, _, _ = checked_run(manifest, allow_large=False)
    validation = plan["splits"]["validation"]
    fitted = {name: load_fit(name, folder, digest, validation["canonical_stream_sha256"])
              for name, folder in candidates.items()}
    identity = {"manifest_sha256": digest, "validation_sha256": validation["sha256"],
                "analysis_code_sha256": code_digest(__file__), "reference": next(iter(fitted)),
                "arms_in_order": list(fitted), "fits": {name: item[2] for name, item in fitted.items()},
                "trace_count": trace_count}
    if (out / "summary.json").exists():
        previous = json.loads((out / "summary.json").read_bytes())
        expected = previous.pop("result_sha256")
        names = {"per-document.json", "private-traces.json"} if trace_count else {"per-document.json"}
        if (sha(canonical(previous)) != expected or any(previous.get(k) != v for k, v in identity.items())
                or set(previous["members_sha256"]) != names
                or any(file_identity(out / name)[1] != value for name, value in previous["members_sha256"].items())):
            raise ValueError("Completed frontend analysis differs from its identity")
        return {**previous, "result_sha256": expected}
    totals = {name: Counter() for name in fitted}
    lengths = {name: Counter() for name in fitted}
    frequencies = {name: Counter() for name in fitted}
    boundaries = {name: Counter() for name in fitted}
    encoding_seconds = Counter()
    ordered, private = [], []
    reference_name = next(iter(fitted))
    common = Counter()
    for row in load_rows(files["validation"][0]):
        text = checked_canonical(row, normalizer)
        data = text.encode()
        refs, scalar = site_reference(text)
        tokens, traced = {}, {}
        for name, (model, native, _, _, _) in fitted.items():
            start = perf_counter()
            encoded = model.encode_offsets(text)
            encoding_seconds[name] += perf_counter() - start
            if encoded.text != text or model.decode(encoded.ids) != text or set(encoded.ids) & set(model.special_ids.values()):
                raise ValueError(f"{name} failed canonical fidelity or emitted reserved controls")
            spans = list(token_payloads(encoded, model, native))
            if b"".join(part[-1] for part in spans) != data:
                raise ValueError(f"{name} token payloads cannot be placed on canonical bytes")
            tokens[name] = len(spans)
            traced[name] = spans
            for _, _, identifier, _, fallback, payload in spans:
                totals[name][category(payload, fallback)] += 1
                lengths[name][len(payload)] += 1
                frequencies[name][identifier] += 1
            nonempty = [span for span in spans if span[0] < span[1]]
            for left, right in zip(nonempty, nonempty[1:]):
                edge = left[1]
                if edge not in scalar:
                    boundaries[name]["inside_utf8_scalar"] += 1
                    continue
                boundaries[name]["scalar_transitions"] += 1
                kind = "byte_backed" if left[4] or right[4] else "lexical"
                for unit, sites in refs.items():
                    if edge in sites:
                        boundaries[name][f"{kind}_{unit}_cuts"] += 1
        common.update({f"internal_{unit}_sites": len(sites) for unit, sites in refs.items()})
        ordered.append({"canonical_sha256": row["canonical_sha256"], "tokens": tokens})
        if trace_count:
            largest = max(tokens.values()) - tokens[reference_name]
            challenger = max(tokens, key=tokens.get)
            extra = Counter(s[0] // 400 for s in traced[challenger])
            extra.subtract(s[0] // 400 for s in traced[reference_name])
            before = max(extra, key=extra.get, default=0) * 400
            left = max(x for x in scalar if x <= before)
            right = max(x for x in scalar if x <= min(len(data), left + 400))
            private.append((largest, {"document_sha256": row["sha256"], "canonical_byte_start": left,
                "canonical_span": data[left:right].decode(),
                "tokenizer_outputs": {name: [{"piece": s[3], "start": s[0], "end": s[1]} for s in spans
                                             if s[0] < right and s[1] > left] for name, spans in traced.items()}}))
            private.sort(key=lambda pair: -pair[0])
            del private[trace_count:]
    summary = {}
    for name, (model, native, tied, report, types) in fitted.items():
        observed = sum(frequencies[name].values())
        if observed != report["validation"]["ids"]:
            raise ValueError(f"{name} traced counts differ from its evaluation receipt")
        summary[name] = {"result_sha256": tied, "types": types, "token_classes": dict(totals[name]),
                         "payload_bytes_distribution": {str(length): count for length, count in lengths[name].items()},
                         "observed_id_frequency_histogram": {str(count): number for count, number in
                                                             Counter(frequencies[name].values()).items()},
                         "verified_common_site_cuts": dict(boundaries[name]), "evidence": report["validation"],
                         "same_session_encode_bytes_per_second": validation["canonical_bytes"] / encoding_seconds[name],
                         "fit": {k: report["fit"].get(k) for k in ("occupied_vocab", "seconds", "peak_rss_bytes")}}
    members = {"per-document.json": format_json(ordered)}
    if trace_count:
        members["private-traces.json"] = format_json([p for _, p in private])
    result = {**identity, "documents": len(ordered), "common_canonical_sites": dict(common), "arms": summary,
              "members_sha256": {name: sha(data) for name, data in members.items()},
              "paired_token_differences": {f"{b}-minus-{a}": paired_stats(
                  [r["tokens"][b] - r["tokens"][a] for r in ordered]) for a, b in combinations(fitted, 2)}}
    result["result_sha256"] = sha(canonical(result))
    with _lock(out / ".analysis.lock"):
        for name, data in members.items():
            immutable(out / name, data)
        immutable(out / "summary.json", format_json(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True, help="NAME=fit-directory")
    parser.add_argument("--trace-count", type=int, default=0)
    args = parser.parse_args()
    candidates = {n: Path(p) for n, p in (arg.split("=", 1) for arg in args.arm)}
    if len(candidates) != len(args.arm):
        parser.error("each fitted arm must be specified once")
    result = analyze(args.manifest, candidates, args.out, trace_count=args.trace_count)
    print(json.dumps({k: v["token_classes"] for k, v in result["arms"].items()}, sort_keys=True))

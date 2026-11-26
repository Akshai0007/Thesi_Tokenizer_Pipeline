"""Fit the unmodified legacy MLBPE plugin on a verified development manifest."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from tokenizers import Tokenizer

from experiments.mom_bpe.diagnose import format_json, immutable, sha
from experiments.mom_bpe.performance import BYTE_SPELLING, PublishedBaseline
from experiments.mom_bpe.pilot import peak_rss_bytes
from experiments.mom_bpe.scale_data import code_digest
from experiments.mom_bpe.scale_fit import VOCAB, checked_run, measurements
from mltok import registry
from mltok.contracts import TrainContext
from mltok.lib.bpe_core import DEFAULT_SPECIALS
from mltok.lib.bpe_progress import canonical
from mltok.lib.io import count_words
from mltok.snapshots import _lock, file_identity

FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
ALGORITHM = {"type": "mlbpe"}
PRETOKENIZER = {"type": "word"}


class MatchedMLBPE(PublishedBaseline):
    """Score native offsets under the same coverage policy as the historical model."""

    def __init__(self, out):
        self.tokenizer = Tokenizer.from_file(str(out / FILES[0]))
        vocabulary = self.tokenizer.get_vocab()
        self.special_ids = {token: vocabulary[token] for token in DEFAULT_SPECIALS}
        self.fallback_ids = frozenset(i for token, i in vocabulary.items() if BYTE_SPELLING.fullmatch(token))
        self.occupied = self.tokenizer.get_vocab_size()


def run(manifest, out):
    plan, digest, files, normalizer, cfg, config_digest = checked_run(manifest, allow_large=False)
    identity = {"arm": "M48", "manifest_sha256": digest, "config_sha256": config_digest, "requested_vocab": VOCAB,
                "train_stream_sha256": plan["splits"]["train"]["canonical_stream_sha256"],
                "validation_stream_sha256": plan["splits"]["validation"]["canonical_stream_sha256"],
                "normalizer": cfg["normalizer"], "pretokenizer": PRETOKENIZER, "marker": "legacy metaspace",
                "plugin_fingerprints": {kind: registry.fingerprint(kind, spec) for kind, spec in
                    (("algorithm", ALGORITHM), ("pretokenizer", PRETOKENIZER), ("normalizer", cfg["normalizer"]))},
                "algorithm": ALGORITHM, "driver_sha256": code_digest(__file__)}
    with _lock(out / ".arm.lock"):
        immutable(out / "identity.json", format_json(identity))
        fit_path, result_path = out / "fit.json", out / "validation.json"
        if fit_path.exists():
            fit = json.loads(fit_path.read_bytes())
            files_verified = {name: file_identity(out / name)[1] for name in FILES}
            if fit["identity"] != identity or fit["files_sha256"] != files_verified:
                raise ValueError("Existing matched MLBPE fit escaped its recorded identity")
            model = MatchedMLBPE(out)
        else:
            stage = out / "attempts" / uuid4().hex[:12]
            start = perf_counter()
            paths = list(map(str, files["train"]))
            counts = count_words(paths, cfg["normalizer"], PRETOKENIZER, max_local_types=0, log=lambda *_: None)
            pretokenizer = registry.make("pretokenizer", PRETOKENIZER)
            _, info = registry.make("algorithm", ALGORITHM).train(
                TrainContext(VOCAB, DEFAULT_SPECIALS, normalizer, pretokenizer, stage, counts, paths, {}, lambda *_: None))
            for name in FILES:
                immutable(out / name, (stage / name).read_bytes())
            model = MatchedMLBPE(out)
            fit = {"identity": identity, "occupied_vocab": model.occupied,
                   "files_sha256": {name: file_identity(out / name)[1] for name in FILES},
                   "training": info, "seconds": perf_counter() - start, "peak_rss_bytes": peak_rss_bytes(),
                   "merge_domain_types_used": len(counts), "mid_fit_checkpointing": False}
            immutable(fit_path, format_json(fit))
        if result_path.exists():
            result = json.loads(result_path.read_bytes())
            expected = result.pop("result_sha256")
            if result["fit"] != fit or expected != sha(canonical(result)):
                raise ValueError("Existing matched evaluation does not match its fit")
            return {**result, "result_sha256": expected}
        result = {**measurements(model, files["validation"][0], normalizer, model.occupied, baseline=True), "fit": fit,
                  "qualification": plan["qualification"], "scientific_holdout": False,
                  "special_ids": model.special_ids, "matches_requested_occupancy": model.occupied == VOCAB}
        result["result_sha256"] = sha(canonical(result))
        immutable(result_path, format_json(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.out)
    print(json.dumps({"occupied": result["fit"]["occupied_vocab"], "seconds": result["fit"]["seconds"],
                      "validation_tokens": result["validation"]["ids"]}, sort_keys=True))

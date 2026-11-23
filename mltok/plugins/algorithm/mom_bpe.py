"""MOM-BPE with replaceable normalization, pretokenization, and initial units."""

import json
from collections import Counter
from itertools import accumulate, islice

from mltok.contracts import Algorithm
from mltok.lib.bpe_core import CONFIDENCE_SCALE, train_bpe
from mltok.lib.bpe_progress import BPEProgress
from mltok.lib.io import data_fingerprint, iter_docs
from mltok.lib.mom_backend import (
    MOMBackend, SCHEMA, encoder_identity, implementation_identity, preprocessing, pretokenizer_split,
    runtime_identity,
)
from mltok.lib.morph_alignment import load_reviewed_forms, project
from mltok.lib.orthographic_units import UNITIZERS, merge_domains
from mltok.registry import fingerprint, keyhash, register


@register("algorithm", "mom_bpe")
class MOMBPE(Algorithm):
    needs_word_counts = False  # Plain-text words preserve literal whitespace and markers.

    def __init__(self, segmenter="akshara", morphology_penalty_milli=0, alignment_file=None,
                 alignment_sha256=None, alphabet_coverage=0.99999, require_full_vocabulary=False):
        if segmenter not in UNITIZERS:
            raise ValueError("Expected codepoint, grapheme, or akshara segmentation")
        if type(morphology_penalty_milli) is not int or not 0 <= morphology_penalty_milli <= CONFIDENCE_SCALE:
            raise ValueError("morphology_penalty_milli must be an integer between 0 and 1000")
        if morphology_penalty_milli and (segmenter != "akshara" or not alignment_file or not alignment_sha256):
            raise ValueError("Weighted morphology needs an akshara arm and a hash-pinned alignment artifact")
        if not 0 < alphabet_coverage <= 1:
            raise ValueError("alphabet_coverage must be within (0, 1]")
        self.segmenter = segmenter
        self.penalty = morphology_penalty_milli
        self.alignment_file = alignment_file
        self.alignment_sha256 = alignment_sha256
        self.alphabet_coverage = alphabet_coverage
        self.require_full_vocabulary = require_full_vocabulary

    @staticmethod
    def variant_label(spec):
        # The cache key still includes every parameter. Keep artifact paths
        # portable even when the external alignment file has an absolute path.
        return "mom_bpe-" + keyhash(spec)[:12]

    @staticmethod
    def fingerprint_dependencies():
        return {**runtime_identity(), **implementation_identity(
            "bpe_core.py", "bpe_progress.py", "io.py", "mom_backend.py", "morph_alignment.py",
            "orthographic_units.py", "unicode_ml.py")}

    def train(self, ctx):
        unitize = UNITIZERS[self.segmenter]
        pre = preprocessing(ctx.normalizer, ctx.pretokenizer)
        def split(text):
            return pretokenizer_split(pre.pre_tokenizer, text)

        reviewed = load_reviewed_forms(self.alignment_file, self.alignment_sha256) if self.alignment_file else {}
        counts = Counter()
        exposure = Counter()
        limit = ctx.resources.get("max_docs_per_shard", 0)
        for path in ctx.shard_paths:
            for raw, _ in islice(iter_docs(path), limit or None):
                text = pre.normalizer.normalize_str(raw)
                exposure["documents"] += 1
                exposure["raw_bytes"] += len(raw.encode("utf-8"))
                exposure["canonical_bytes"] += len(text.encode("utf-8"))
                counts.update(merge_domains(text, split))
        frequencies = Counter()
        for word, count in counts.items():
            for unit in unitize(word):
                frequencies[unit] += count
        available = ctx.vocab - len(ctx.specials) - 256
        if available < 2 or not frequencies:
            raise ValueError("MOM-BPE requires a nonempty corpus and a vocabulary above the fallback budget")
        alphabet, mass = [], 0
        total = sum(frequencies.values())
        for unit, count in sorted(frequencies.items(), key=lambda pair: (-pair[1], pair[0])):
            if len(alphabet) >= available // 2 or mass >= self.alphabet_coverage * total:
                break
            alphabet.append(unit)
            mass += count
        max_types = ctx.resources.get("max_word_types", 0)
        eligible_words = counts.keys()
        if max_types and reviewed:
            eligible_words = set(sorted(counts, key=lambda word: (-counts[word], word))[:max_types])
        kept_units = set(alphabet)
        weights = {}
        for word in reviewed.keys() & counts.keys():
            projected = project(word, reviewed[word], unitize)
            units = unitize(word)
            reachable = {
                offset for offset, left, right in zip(accumulate(map(len, units)), units, units[1:])
                if left in kept_units and right in kept_units
            }
            weights[word] = {
                offset: weight for offset, weight in projected.items()
                if word in eligible_words and offset in reachable
            }
        if self.penalty and not any(weights.values()):
            raise ValueError("Weighted morphology found no supported training surfaces with reachable boundaries")
        progress = None
        interval = ctx.resources.get("merge_checkpoint_every", 0)
        if interval:
            identity = keyhash(
                "mom-merge-progress-v1", data_fingerprint(ctx.shard_paths),
                fingerprint("algorithm", {"type": self.name, **self.params}), ctx.vocab, ctx.specials,
                json.loads(pre.to_str()), alphabet, weights, limit, max_types,
            )
            progress = BPEProgress(ctx.out_dir / "merge-checkpoints", identity, interval)
        trained = train_bpe(
            counts, vocab_target=ctx.vocab, n_specials=len(ctx.specials), alphabet=alphabet,
            initial_units=unitize, boundary_weights=weights, penalty_milli=self.penalty,
            max_word_types=max_types, progress=progress, log=ctx.log,
        )
        occupied = len(trained.tokens) + len(ctx.specials) + 256
        if self.require_full_vocabulary and occupied != ctx.vocab:
            raise ValueError(f"Training filled {occupied} of {ctx.vocab} slots")
        artifact = {
            "schema": SCHEMA, "runtime": runtime_identity(), "encoder": encoder_identity(), "vocab_size": ctx.vocab,
            "segmenter": self.segmenter, "specials": ctx.specials, "initial": alphabet,
            "merges": trained.merges, "preprocessing": json.loads(pre.to_str()),
            "provenance": {"implementation": self.fingerprint_dependencies(),
                           "morphology_penalty_milli": self.penalty, "alignment_sha256": self.alignment_sha256},
        }
        backend = MOMBackend(artifact)
        backend.save(ctx.out_dir)
        return backend, {
            **trained.info, **exposure, "occupied_vocab": occupied,
            "initial_unit_coverage": mass / total, "supported_morphology_forms": sum(map(bool, weights.values())),
            "abstained_morphology_forms": sum(not x for x in weights.values()),
        }

    def load(self, out_dir, normalizer):
        return MOMBackend.load(out_dir)

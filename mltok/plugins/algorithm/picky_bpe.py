"""Matched orthographic BPE with optional chronological Picky BPE refinement."""

import hashlib
import json
from collections import Counter
from itertools import groupby, islice
from pathlib import Path

from mltok.contracts import Algorithm
from mltok.lib.bpe_core import CONFIDENCE_SCALE
from mltok.lib.bpe_progress import BPEProgress
from mltok.lib.hybrid_units import initial_units, select_alphabet
from mltok.lib.io import data_fingerprint, iter_docs
from mltok.lib.mom_backend import implementation_identity, preprocessing, pretokenizer_split, runtime_identity
from mltok.lib.morph_alignment import load_reviewed_forms, project
from mltok.lib.orthographic_units import UNITIZERS, merge_domains
from mltok.lib.picky_backend import EventBackend, SCHEMA, event_encoder_identity
from mltok.lib.picky_training import fit
from mltok.registry import fingerprint, keyhash, register


def training_examples(counts, alphabet, unitize, reviewed):
    initial = set(alphabet)
    examples = []
    seen_forms, surfaced = set(), set()
    fallback_bytes = 0
    for word, count in counts:
        weights = {}
        if word in reviewed:
            seen_forms.add(word)
            # Scalar backoff must never authorize a linguistic boundary inside
            # the independently defined Malayalam reference unit.
            weights = project(word, reviewed[word], UNITIZERS["akshara"])
        start = 0
        for known, chunk in groupby(unitize(word), initial.__contains__):
            units = list(chunk)
            coordinates = {}
            offset = start
            for unit in units[:-1]:
                offset += len(unit)
                if offset in weights:
                    coordinates[offset - start] = weights[offset]
            if known:
                examples.append((units, count, coordinates))
                if coordinates:
                    surfaced.add(word)
            else:
                fallback_bytes += count * sum(len(unit.encode("utf-8")) for unit in units)
            start += sum(map(len, units))
    return examples, fallback_bytes, len(seen_forms), len(surfaced)


@register("algorithm", "picky_bpe")
class PickyBPE(Algorithm):
    needs_word_counts = False

    def __init__(self, segmenter="hybrid", picky_threshold_milli=None, hybrid_min_count=8, hybrid_max_atoms=512,
                 alphabet_coverage=.99999, morphology_penalty_milli=0, alignment_file=None,
                 alignment_sha256=None, require_full_vocabulary=False):
        if segmenter not in (*UNITIZERS, "hybrid"):
            raise ValueError("Unknown event BPE segmenter")
        if picky_threshold_milli is not None and (
            type(picky_threshold_milli) is not int or not 1 <= picky_threshold_milli <= CONFIDENCE_SCALE
        ):
            raise ValueError("Expected a Picky threshold in 1..1000 thousandths")
        if type(hybrid_min_count) is not int or hybrid_min_count < 1 or type(hybrid_max_atoms) is not int or hybrid_max_atoms < 0:
            raise ValueError("Invalid hybrid atom selection policy")
        if not 0 < alphabet_coverage <= 1:
            raise ValueError("Alphabet coverage must be within (0, 1]")
        if type(morphology_penalty_milli) is not int or not 0 <= morphology_penalty_milli <= CONFIDENCE_SCALE:
            raise ValueError("Expected a morphology penalty in 0..1000 thousandths")
        if morphology_penalty_milli and (segmenter not in ("akshara", "hybrid") or not alignment_file or not alignment_sha256):
            raise ValueError("Weighted morphology requires reviewed, pinned akshara or hybrid surface alignments")
        self.segmenter, self.threshold = segmenter, picky_threshold_milli
        self.minimum, self.atom_cap = hybrid_min_count, hybrid_max_atoms
        self.coverage, self.penalty = alphabet_coverage, morphology_penalty_milli
        self.alignment, self.alignment_sha = alignment_file, alignment_sha256
        self.require_full = require_full_vocabulary

    @staticmethod
    def variant_label(spec):
        return "picky_bpe-" + keyhash(spec)[:12]

    @staticmethod
    def fingerprint_dependencies():
        return {**runtime_identity(), **event_encoder_identity(), **implementation_identity(
            "picky_training.py", "morph_alignment.py", "bpe_progress.py", "io.py")}

    def train(self, ctx):
        native = preprocessing(ctx.normalizer, ctx.pretokenizer)

        def presplit(surface):
            return pretokenizer_split(native.pre_tokenizer, surface)

        reviewed = {}
        if self.alignment:
            payload = Path(self.alignment).read_bytes()
            if (hashlib.sha256(payload).hexdigest() != self.alignment_sha
                    or json.loads(payload).get("qualification") != "adjudicated_training"):
                raise ValueError("Morphology evidence must be marked as adjudicated training alignments")
            reviewed = load_reviewed_forms(self.alignment, self.alignment_sha)
        counts, exposure = Counter(), Counter()
        limit = ctx.resources.get("max_docs_per_shard", 0)
        max_types = ctx.resources.get("max_word_types", 0)
        for path in ctx.shard_paths:
            for raw, _ in islice(iter_docs(path), limit or None):
                text = native.normalizer.normalize_str(raw)
                exposure["documents"] += 1
                exposure["raw_bytes"] += len(raw.encode("utf-8"))
                exposure["canonical_bytes"] += len(text.encode("utf-8"))
                counts.update(merge_domains(text, presplit))
        budget = ctx.vocab - len(ctx.specials) - 256
        if budget < 2 or not counts:
            raise ValueError("Event BPE needs a nonempty corpus and lexical capacity above fallback")
        alphabet, atoms, coverage = select_alphabet(
            counts, self.segmenter, budget // 2, self.coverage, self.minimum, self.atom_cap,
        )
        selected = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        if max_types:
            selected = selected[:max_types]
        examples, fallback_bytes, seen, supported = training_examples(
            selected, alphabet, initial_units(self.segmenter, atoms), reviewed,
        )
        if self.penalty and not supported:
            raise ValueError("No adjudicated training surface has a reachable weighted boundary")
        pre_json = json.loads(native.to_str())
        progress = None
        if interval := ctx.resources.get("merge_checkpoint_every", 0):
            identity = keyhash("picky-event-progress-v1", data_fingerprint(ctx.shard_paths),
                               fingerprint("algorithm", {"type": self.name, **self.params}),
                               ctx.vocab, ctx.specials, pre_json, alphabet, atoms, limit, max_types)
            progress = BPEProgress(ctx.out_dir / "event-checkpoints", identity, interval)
        program, info = fit(alphabet, examples, budget, threshold=self.threshold, penalty=self.penalty, progress=progress)
        occupied = len(program.active) + len(ctx.specials) + 256
        if self.require_full and occupied != ctx.vocab:
            raise ValueError(f"Event BPE filled {occupied} of {ctx.vocab} requested slots")
        backend = EventBackend({
            "schema": SCHEMA, "runtime": runtime_identity(), "encoder": event_encoder_identity(),
            "vocab_size": ctx.vocab, "specials": list(ctx.specials), "segmenter": self.segmenter,
            "atoms": atoms, "initial": alphabet, "events": program.events, "preprocessing": pre_json,
            "morphology_penalty_milli": self.penalty, "alignment_sha256": self.alignment_sha,
            "picky_threshold_milli": self.threshold,
        })
        backend.save(ctx.out_dir)
        return backend, {
            **exposure, **info, "occupied_vocab": occupied, "protected_atoms": len(atoms),
            "initial_unit_coverage": coverage, "training_fallback_ids_on_selected_types": fallback_bytes,
            "word_types_used": len(selected), "word_mass_covered": sum(c for _, c in selected) / sum(counts.values()),
            "reviewed_training_forms_encountered": seen, "supported_morphology_forms": supported,
            "abstained_morphology_forms": seen - supported,
        }

    def load(self, out_dir, normalizer):
        return EventBackend.load(out_dir)

"""Hybrid Picky BPE with independent training-time orthographic repair."""

import json
from collections import Counter
from itertools import islice

from mltok.contracts import Algorithm
from mltok.lib.bpe_core import CONFIDENCE_SCALE
from mltok.lib.bpe_progress import BPEProgress
from mltok.lib.hybrid_units import select_alphabet
from mltok.lib.io import data_fingerprint, iter_docs
from mltok.lib.mom_backend import implementation_identity, preprocessing, pretokenizer_split, runtime_identity
from mltok.lib.orthographic_guard import guarded_examples
from mltok.lib.orthographic_units import merge_domains
from mltok.lib.picky_backend import EventBackend, SCHEMA, event_encoder_identity
from mltok.lib.picky_training import fit
from mltok.registry import fingerprint, keyhash, register


@register("algorithm", "boundary_guarded_hybrid")
class BoundaryGuardedHybrid(Algorithm):
    needs_word_counts = False

    def __init__(self, guard_milli=1000, picky_threshold_milli=900, hybrid_min_count=8,
                 hybrid_max_atoms=512, alphabet_coverage=.99999, require_full_vocabulary=False):
        if type(guard_milli) is not int or not 0 <= guard_milli <= CONFIDENCE_SCALE:
            raise ValueError("Guard strength must be in 0..1000 thousandths")
        if picky_threshold_milli is not None and (
            type(picky_threshold_milli) is not int or not 1 <= picky_threshold_milli <= CONFIDENCE_SCALE
        ):
            raise ValueError("Picky threshold must be in 1..1000 thousandths")
        if type(hybrid_min_count) is not int or hybrid_min_count < 1 or type(hybrid_max_atoms) is not int or hybrid_max_atoms < 0:
            raise ValueError("Invalid hybrid alphabet policy")
        if not 0 < alphabet_coverage <= 1:
            raise ValueError("Alphabet coverage must be within (0, 1]")
        self.guard, self.threshold = guard_milli, picky_threshold_milli
        self.minimum, self.atom_cap, self.coverage = hybrid_min_count, hybrid_max_atoms, alphabet_coverage
        self.require_full = require_full_vocabulary

    @staticmethod
    def variant_label(spec):
        return "boundary_guarded_hybrid-" + keyhash(spec)[:12]

    @staticmethod
    def fingerprint_dependencies():
        return {**runtime_identity(), **event_encoder_identity(), **implementation_identity(
            "orthographic_guard.py", "picky_training.py", "bpe_progress.py", "io.py")}

    def train(self, ctx):
        native = preprocessing(ctx.normalizer, ctx.pretokenizer)

        def presplit(text):
            return pretokenizer_split(native.pre_tokenizer, text)

        counts, exposure = Counter(), Counter()
        limit = ctx.resources.get("max_docs_per_shard", 0)
        max_types = ctx.resources.get("max_word_types", 0)
        for shard in ctx.shard_paths:
            for raw, _ in islice(iter_docs(shard), limit or None):
                text = native.normalizer.normalize_str(raw)
                exposure.update(documents=1, raw_bytes=len(raw.encode()), canonical_bytes=len(text.encode()))
                counts.update(merge_domains(text, presplit))
        budget = ctx.vocab - len(ctx.specials) - 256
        if budget < 2 or not counts:
            raise ValueError("Guarded hybrid needs a nonempty corpus and lexical capacity above fallback")
        alphabet, atoms, coverage = select_alphabet(
            counts, "hybrid", budget // 2, self.coverage, self.minimum, self.atom_cap,
        )
        selected = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        if max_types:
            selected = selected[:max_types]
        examples, fallback_bytes, weighted_sites = guarded_examples(selected, alphabet, atoms)
        pre_json = json.loads(native.to_str())
        progress = None
        if interval := ctx.resources.get("merge_checkpoint_every", 0):
            identity = keyhash(
                "guarded-hybrid-progress-v1", data_fingerprint(ctx.shard_paths),
                fingerprint("algorithm", {"type": self.name, **self.params}),
                ctx.vocab, ctx.specials, pre_json, alphabet, atoms, limit, max_types,
            )
            progress = BPEProgress(ctx.out_dir / "event-checkpoints", identity, interval)
        program, info = fit(alphabet, examples, budget, threshold=self.threshold, penalty=self.guard, progress=progress)
        occupied = len(program.active) + len(ctx.specials) + 256
        if self.require_full and occupied != ctx.vocab:
            raise ValueError(f"Guarded hybrid filled {occupied} of {ctx.vocab} requested slots")
        backend = EventBackend({
            "schema": SCHEMA, "runtime": runtime_identity(), "encoder": event_encoder_identity(),
            "vocab_size": ctx.vocab, "specials": list(ctx.specials), "segmenter": "hybrid",
            "atoms": atoms, "initial": alphabet, "events": program.events, "preprocessing": pre_json,
            "morphology_penalty_milli": 0, "alignment_sha256": None, "picky_threshold_milli": self.threshold,
            "orthographic_guard_milli": self.guard,
        })
        backend.save(ctx.out_dir)
        return backend, {
            **exposure, **info, "occupied_vocab": occupied, "protected_atoms": len(atoms),
            "initial_unit_coverage": coverage, "training_fallback_ids_on_selected_types": fallback_bytes,
            "word_types_used": len(selected), "word_mass_covered": sum(c for _, c in selected) / sum(counts.values()),
            "guarded_internal_sites": weighted_sites if self.guard else 0,
        }

    def load(self, out_dir, normalizer):
        return EventBackend.load(out_dir)

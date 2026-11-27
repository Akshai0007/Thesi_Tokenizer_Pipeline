"""Boundary-guarded hybrid Picky BPE v2 with coverage-enforced seed units and optional uniform prefix."""

import json
from collections import Counter
from itertools import islice
from pathlib import Path

from mltok.contracts import Algorithm
from mltok.lib.bpe_core import CONFIDENCE_SCALE
from mltok.lib.bpe_progress import BPEProgress
from mltok.lib.hybrid_units_v2 import select_alphabet_v2
from mltok.lib.io import data_fingerprint, iter_docs
from mltok.lib.mom_backend import (
    implementation_identity,
    preprocessing,
    pretokenizer_split,
    runtime_identity,
)
from mltok.lib.orthographic_guard import guarded_examples
from mltok.lib.orthographic_units import merge_domains
from mltok.lib.picky_backend import EventBackend, SCHEMA as V1_SCHEMA, event_encoder_identity
from mltok.lib.picky_training import fit
from mltok.lib.uniform_prefix import (
    ARTIFACT as V2_ARTIFACT,
    ESC,
    SCHEMA as V2_SCHEMA,
    SP,
    UniformPrefixEventBackend,
    encode_transport,
    uniform_prefix_encoder_identity,
)
from mltok.registry import fingerprint, keyhash, register


@register("algorithm", "boundary_guarded_hybrid_v2")
class BoundaryGuardedHybridV2(Algorithm):
    """Guarded Hybrid Picky BPE v2 supporting Candidate A (T8-LS) and Candidate B (M48-Guard)."""

    needs_word_counts = False

    def __init__(
        self,
        guard_milli=1000,
        picky_threshold_milli=900,
        hybrid_min_count=8,
        hybrid_max_atoms=512,
        alphabet_coverage=0.99999,
        require_full_vocabulary=False,
        uniform_prefix="auto",
    ):
        if type(guard_milli) is not int or not 0 <= guard_milli <= CONFIDENCE_SCALE:
            raise ValueError("Guard strength must be in 0..1000 thousandths")
        if picky_threshold_milli is not None and (
            type(picky_threshold_milli) is not int or not 1 <= picky_threshold_milli <= CONFIDENCE_SCALE
        ):
            raise ValueError("Picky threshold must be in 1..1000 thousandths")
        if (
            type(hybrid_min_count) is not int
            or hybrid_min_count < 1
            or type(hybrid_max_atoms) is not int
            or hybrid_max_atoms < 0
        ):
            raise ValueError("Invalid hybrid alphabet policy")
        if not 0 < alphabet_coverage <= 1:
            raise ValueError("Alphabet coverage must be within (0, 1]")
        if uniform_prefix != "auto" and type(uniform_prefix) is not bool:
            raise ValueError("uniform_prefix must be 'auto', True, or False")
        self.guard, self.threshold = guard_milli, picky_threshold_milli
        self.minimum, self.atom_cap, self.coverage = hybrid_min_count, hybrid_max_atoms, alphabet_coverage
        self.require_full = require_full_vocabulary
        self.uniform_prefix = uniform_prefix

    @staticmethod
    def variant_label(spec):
        return "boundary_guarded_hybrid_v2-" + keyhash(spec)[:12]

    @staticmethod
    def fingerprint_dependencies():
        return {
            **runtime_identity(),
            **uniform_prefix_encoder_identity(),
            **implementation_identity("hybrid_units_v2.py", "orthographic_guard.py", "picky_training.py", "bpe_progress.py", "io.py"),
        }

    def _use_uniform_prefix(self, pretokenizer):
        if self.uniform_prefix == "auto":
            return pretokenizer is not None and getattr(pretokenizer, "name", "") == "lossless_word_prefix"
        return bool(self.uniform_prefix)

    def train(self, ctx):
        native = preprocessing(ctx.normalizer, ctx.pretokenizer)
        use_uniform = self._use_uniform_prefix(ctx.pretokenizer)

        def presplit(text):
            return pretokenizer_split(native.pre_tokenizer, text)

        counts, exposure = Counter(), Counter()
        limit = ctx.resources.get("max_docs_per_shard", 0)
        max_types = ctx.resources.get("max_word_types", 0)
        for shard in ctx.shard_paths:
            for raw, _ in islice(iter_docs(shard), limit or None):
                canonical_text = native.normalizer.normalize_str(raw)
                exposure.update(
                    documents=1,
                    raw_bytes=len(raw.encode("utf-8")),
                    canonical_bytes=len(canonical_text.encode("utf-8")),
                )
                surface = encode_transport(canonical_text) if use_uniform else canonical_text
                counts.update(merge_domains(surface, presplit))

        budget = ctx.vocab - len(ctx.specials) - 256
        if budget < 2 or not counts:
            raise ValueError("Guarded hybrid v2 needs a nonempty corpus and lexical capacity above fallback")

        extra_forced = (SP, ESC) if use_uniform else ()
        alphabet, atoms, coverage = select_alphabet_v2(
            counts,
            "hybrid",
            budget // 2,
            self.coverage,
            self.minimum,
            self.atom_cap,
            extra_forced=extra_forced,
        )
        selected = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        if max_types:
            selected = selected[:max_types]
        examples, fallback_bytes, weighted_sites = guarded_examples(selected, alphabet, atoms)
        pre_json = json.loads(native.to_str())
        progress = None
        if interval := ctx.resources.get("merge_checkpoint_every", 0):
            identity = keyhash(
                "guarded-hybrid-v2-progress-v1",
                data_fingerprint(ctx.shard_paths),
                fingerprint("algorithm", {"type": self.name, **self.params}),
                ctx.vocab,
                ctx.specials,
                pre_json,
                alphabet,
                atoms,
                limit,
                max_types,
                use_uniform,
            )
            progress = BPEProgress(ctx.out_dir / "event-checkpoints", identity, interval)

        program, info = fit(alphabet, examples, budget, threshold=self.threshold, penalty=self.guard, progress=progress)
        occupied = len(program.active) + len(ctx.specials) + 256
        if self.require_full and occupied != ctx.vocab:
            raise ValueError(f"Guarded hybrid v2 filled {occupied} of {ctx.vocab} requested slots")

        if use_uniform:
            backend = UniformPrefixEventBackend({
                "schema": V2_SCHEMA,
                "runtime": runtime_identity(),
                "encoder": uniform_prefix_encoder_identity(),
                "vocab_size": ctx.vocab,
                "specials": list(ctx.specials),
                "segmenter": "hybrid",
                "atoms": atoms,
                "initial": alphabet,
                "events": program.events,
                "preprocessing": pre_json,
                "morphology_penalty_milli": 0,
                "alignment_sha256": None,
                "picky_threshold_milli": self.threshold,
                "orthographic_guard_milli": self.guard,
                "uniform_prefix": True,
                "alphabet_coverage": self.coverage,
            })
        else:
            backend = EventBackend({
                "schema": V1_SCHEMA,
                "runtime": runtime_identity(),
                "encoder": event_encoder_identity(),
                "vocab_size": ctx.vocab,
                "specials": list(ctx.specials),
                "segmenter": "hybrid",
                "atoms": atoms,
                "initial": alphabet,
                "events": program.events,
                "preprocessing": pre_json,
                "morphology_penalty_milli": 0,
                "alignment_sha256": None,
                "picky_threshold_milli": self.threshold,
                "orthographic_guard_milli": self.guard,
            })
        backend.save(ctx.out_dir)
        return backend, {
            **exposure,
            **info,
            "occupied_vocab": occupied,
            "protected_atoms": len(atoms),
            "initial_unit_coverage": coverage,
            "training_fallback_ids_on_selected_types": fallback_bytes,
            "word_types_used": len(selected),
            "word_mass_covered": sum(c for _, c in selected) / sum(counts.values()),
            "guarded_internal_sites": weighted_sites if self.guard else 0,
            "uniform_prefix": use_uniform,
        }

    def load(self, out_dir, normalizer):
        if (Path(out_dir) / V2_ARTIFACT).is_file():
            return UniformPrefixEventBackend.load(out_dir)
        return EventBackend.load(out_dir)

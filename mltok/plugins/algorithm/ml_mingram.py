"""Candidate C: ML-MinGram (Akshara-Safe Guarded-BPE-Seeded Hard-EM Unigram)."""

import json
from collections import Counter
from itertools import islice
from pathlib import Path

from mltok.contracts import Algorithm
from mltok.lib.bpe_core import CONFIDENCE_SCALE
from mltok.lib.hybrid_units_v2 import select_alphabet_v2
from mltok.lib.io import data_fingerprint, iter_docs
from mltok.lib.mingram_core import (
    MinGramBackend,
    SCHEMA,
    UniformPrefixMinGramBackend,
    export_hf_unigram_json,
    load_mingram_backend,
    mingram_encoder_identity,
    train_mingram_unigram,
)
from mltok.lib.mom_backend import (
    implementation_identity,
    preprocessing,
    pretokenizer_split,
    runtime_identity,
)
from mltok.lib.orthographic_guard import guarded_examples
from mltok.lib.orthographic_units import merge_domains
from mltok.lib.picky_training import fit
from mltok.lib.uniform_prefix import ESC, SP, encode_transport
from mltok.registry import keyhash, register


@register("algorithm", "ml_mingram")
class MLMinGram(Algorithm):
    """BPE-seeded Hard-EM Unigram tokenizer with orthographic guard and lossless HF export."""

    needs_word_counts = False
    uses_pretokenizer = True
    contract_vocab = 1200

    def __init__(
        self,
        seed_multiplier=1.15,
        guard_milli=1000,
        akshara_guard_milli=1000,
        picky_threshold_milli=900,
        alphabet_coverage=0.9999,
        shrink_factor=0.75,
        em_iters=2,
        final_em_iters=2,
        filter_malayalam_dangling=True,
        uniform_prefix="auto",
        export_hf_json=True,
        hybrid_min_count=8,
        atom_cap=512,
        require_full_vocabulary=False,
    ):
        if not isinstance(seed_multiplier, (int, float)) or isinstance(seed_multiplier, bool) or seed_multiplier < 1.0:
            raise ValueError("seed_multiplier must be >= 1.0")
        if type(guard_milli) is not int or not 0 <= guard_milli <= CONFIDENCE_SCALE:
            raise ValueError("guard_milli must be in 0..1000 thousandths")
        if type(akshara_guard_milli) is not int or akshara_guard_milli < 0:
            raise ValueError("akshara_guard_milli must be a non-negative integer")
        if picky_threshold_milli is not None and (
            type(picky_threshold_milli) is not int or not 1 <= picky_threshold_milli <= CONFIDENCE_SCALE
        ):
            raise ValueError("picky_threshold_milli must be in 1..1000 thousandths")
        if not 0 < alphabet_coverage <= 1:
            raise ValueError("alphabet_coverage must be within (0, 1]")
        if not 0 < shrink_factor <= 1:
            raise ValueError("shrink_factor must be within (0, 1]")
        if type(em_iters) is not int or em_iters < 1 or type(final_em_iters) is not int or final_em_iters < 0:
            raise ValueError("em_iters must be >= 1 and final_em_iters >= 0")
        if type(filter_malayalam_dangling) is not bool or type(export_hf_json) is not bool:
            raise ValueError("filter_malayalam_dangling and export_hf_json must be booleans")
        if uniform_prefix != "auto" and type(uniform_prefix) is not bool:
            raise ValueError("uniform_prefix must be 'auto', True, or False")
        if type(hybrid_min_count) is not int or hybrid_min_count < 1 or type(atom_cap) is not int or atom_cap < 0:
            raise ValueError("Invalid hybrid atom selection parameters")

        self.seed_multiplier = float(seed_multiplier)
        self.guard_milli = guard_milli
        self.akshara_guard_milli = akshara_guard_milli
        self.picky_threshold_milli = picky_threshold_milli
        self.alphabet_coverage = float(alphabet_coverage)
        self.shrink_factor = float(shrink_factor)
        self.em_iters = em_iters
        self.final_em_iters = final_em_iters
        self.filter_malayalam_dangling = filter_malayalam_dangling
        self.uniform_prefix = uniform_prefix
        self.export_hf_json = export_hf_json
        self.hybrid_min_count = hybrid_min_count
        self.atom_cap = atom_cap
        self.require_full_vocabulary = require_full_vocabulary

    @staticmethod
    def variant_label(spec):
        return "ml_mingram-" + keyhash(spec)[:12]

    @staticmethod
    def fingerprint_dependencies():
        return {
            **runtime_identity(),
            **mingram_encoder_identity(),
            **implementation_identity(
                "hybrid_units_v2.py",
                "orthographic_guard.py",
                "picky_training.py",
                "io.py",
            ),
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

        target_lexical_vocab = ctx.vocab - len(ctx.specials) - 256
        if target_lexical_vocab < 2 or not counts:
            raise ValueError("ML-MinGram requires a nonempty corpus and lexical capacity above fallback")

        seed_vocab = max(ctx.vocab, int(round(ctx.vocab * self.seed_multiplier)))
        seed_budget = seed_vocab - len(ctx.specials) - 256
        extra_forced = (SP, ESC) if use_uniform else ()
        alphabet, atoms, unit_coverage = select_alphabet_v2(
            counts,
            "hybrid",
            min(target_lexical_vocab // 2, seed_budget // 2),
            self.alphabet_coverage,
            self.hybrid_min_count,
            self.atom_cap,
            extra_forced=extra_forced,
        )

        selected = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        if max_types:
            selected = selected[:max_types]

        examples, fallback_bytes, weighted_sites = guarded_examples(selected, alphabet, atoms)
        seed_program, seed_info = fit(
            alphabet,
            examples,
            seed_budget,
            threshold=self.picky_threshold_milli,
            penalty=self.guard_milli,
        )

        selected_counts = dict(selected)
        protected = set(alphabet) | set(extra_forced)
        scores, em_stats = train_mingram_unigram(
            selected_counts,
            seed_program.vocabulary(),
            target_vocab=target_lexical_vocab,
            protected=protected,
            shrink_factor=self.shrink_factor,
            em_iters=self.em_iters,
            final_em_iters=self.final_em_iters,
            filter_malayalam_dangling=self.filter_malayalam_dangling,
            akshara_guard_milli=self.akshara_guard_milli,
        )

        occupied = len(scores) + len(ctx.specials) + 256
        if self.require_full_vocabulary and occupied != ctx.vocab:
            raise ValueError(f"ML-MinGram filled {occupied} of {ctx.vocab} requested slots")

        fallback_cost = max(20.0, -min(scores.values(), default=-10.0) + 5.0)
        pre_json = json.loads(native.to_str())
        artifact = {
            "schema": SCHEMA,
            "runtime": runtime_identity(),
            "encoder": mingram_encoder_identity(),
            "vocab_size": ctx.vocab,
            "specials": list(ctx.specials),
            "uniform_prefix": use_uniform,
            "atoms": atoms,
            "initial": alphabet,
            "scores": scores,
            "fallback_cost": fallback_cost,
            "preprocessing": pre_json,
            "data_fingerprint": data_fingerprint(ctx.shard_paths),
            "seed_multiplier": self.seed_multiplier,
            "orthographic_guard_milli": self.guard_milli,
            "akshara_guard_milli": self.akshara_guard_milli,
            "picky_threshold_milli": self.picky_threshold_milli,
            "alphabet_coverage": self.alphabet_coverage,
        }
        backend_cls = UniformPrefixMinGramBackend if use_uniform else MinGramBackend
        backend = backend_cls(artifact)
        backend.save(ctx.out_dir)

        hf_report = None
        if self.export_hf_json:
            hf_report = export_hf_unigram_json(
                backend,
                ctx.out_dir,
                hf_normalizer=ctx.normalizer.hf(),
                hf_pretokenizer=ctx.pretokenizer.hf(),
                specials=list(ctx.specials),
                uniform_prefix=use_uniform,
            )

        return backend, {
            **exposure,
            **em_stats,
            "seed_budget": seed_budget,
            "seed_stop_reason": seed_info["stop_reason"],
            "seed_merges": seed_info["n_merges"],
            "seed_removed": seed_info["n_removed"],
            "occupied_vocab": occupied,
            "protected_atoms": len(atoms),
            "initial_unit_coverage": unit_coverage,
            "training_fallback_ids_on_selected_types": fallback_bytes,
            "word_types_used": len(selected),
            "word_mass_covered": sum(c for _, c in selected) / sum(counts.values()),
            "guarded_internal_sites": weighted_sites if self.guard_milli else 0,
            "uniform_prefix": use_uniform,
            "hf_export": hf_report,
        }

    def load(self, out_dir, normalizer):
        return load_mingram_backend(Path(out_dir))

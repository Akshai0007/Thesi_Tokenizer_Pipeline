"""Versioned project encoder for chronological BPE events."""

import json
from itertools import groupby
from pathlib import Path

from tokenizers import Tokenizer

from .atomic import atomic_writer
from .bpe_core import byte_tokens
from .bpe_progress import canonical
from .hybrid_units import initial_units
from .mom_backend import MOMBackend, SurfaceEncoding, encoder_identity, implementation_identity, pretokenizer_split, runtime_identity
from .orthographic_units import merge_domains
from .picky_events import EventProgram

ARTIFACT = "picky-bpe.json"
SCHEMA = "malayalam-event-bpe-v1"


def event_encoder_identity():
    return {**encoder_identity(), **implementation_identity("hybrid_units.py", "picky_backend.py", "picky_events.py")}


class EventBackend(MOMBackend):
    def __init__(self, artifact):
        if (artifact["schema"] != SCHEMA or artifact["runtime"] != runtime_identity()
                or artifact["encoder"] != event_encoder_identity()):
            raise ValueError("Event BPE artifact or Unicode runtime is incompatible")
        self.artifact = artifact
        self.name = "picky_bpe"
        self._transport = Tokenizer.from_str(json.dumps(artifact["preprocessing"], ensure_ascii=False))
        self._program = EventProgram(artifact["initial"], artifact["events"])
        if not set(artifact["atoms"]) <= self._program.initial:
            raise ValueError("Protected atoms are missing from the initial alphabet")
        self._unitize = initial_units(artifact["segmenter"], artifact["atoms"])
        specials = artifact["specials"]
        self.special_ids = {token: i for i, token in enumerate(specials)}
        if len(self.special_ids) != len(specials):
            raise ValueError("Duplicate event BPE special IDs")
        self.unk_id, self.eos_id = (self.special_ids[key] for key in ("<unk>", "</s>"))
        self._byte_start = len(specials)
        lexical_start = self._byte_start + 256
        self.fallback_ids = frozenset(range(self._byte_start, lexical_start))
        vocabulary = self._program.vocabulary()
        self._lexical_ids = {token: i for i, token in enumerate(vocabulary, lexical_start)}
        self._pieces = [*specials, *byte_tokens(), *vocabulary]
        self.vocab_size = artifact["vocab_size"]
        if len(self._pieces) > self.vocab_size:
            raise ValueError("Event BPE artifact exceeds its target vocabulary")

    def encode_offsets(self, text):
        canonical_text = self.expected(text)
        ids, tokens, offsets = [], [], []
        position = 0

        def presplit(surface):
            return pretokenizer_split(self._transport.pre_tokenizer, surface)

        for word in merge_domains(canonical_text, presplit):
            for known, units in groupby(self._unitize(word), self._program.initial.__contains__):
                pieces = self._program.replay(list(units)) if known else units
                for piece in pieces:
                    end = position + len(piece)
                    encoded = [self._lexical_ids[piece]] if known else [self._byte_start + b for b in piece.encode("utf-8")]
                    ids.extend(encoded)
                    tokens.extend(self._pieces[i] for i in encoded)
                    offsets.extend([(position, end)] * len(encoded))
                    position = end
        return SurfaceEncoding(canonical_text, ids, tokens, offsets)

    @classmethod
    def load(cls, out_dir):
        return cls(json.loads((Path(out_dir) / ARTIFACT).read_bytes()))

    def save(self, out_dir):
        path = Path(out_dir) / ARTIFACT
        path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical(self.artifact) + b"\n"
        try:
            with atomic_writer(path, overwrite=False) as writer:
                writer.write(data)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("An incompatible event BPE artifact already exists") from None

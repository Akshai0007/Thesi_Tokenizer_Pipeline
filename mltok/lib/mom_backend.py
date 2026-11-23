"""Versioned MOM-BPE artifact and rank-replay inference."""

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from itertools import groupby
from numbers import Integral
from pathlib import Path

import regex
import tokenizers
from tokenizers import Tokenizer, models

from mltok.contracts import Backend

from .atomic import atomic_writer
from .bpe_core import byte_tokens, merge_units
from .orthographic_units import UNITIZERS, merge_domains

ARTIFACT = "mom-bpe.json"
SCHEMA = "mom-bpe-v1"
# Previous reviewed encoders share the same valid-ID rank replay and grammar.
_REVIEWED_ENCODER = "4b75e5886aea6ae1291e5170f1a3654c66d99fbc7c4711aee503d121079d022a"
_REVIEWED_CORES = {
    "48d2546e2e92cf4ed5b584861654a8bfeed6643a3fd12dde9ffad0c58146876c",
    "8de216db754e184d06c6eccae14fc346aa2110e5d8d5257dac5f6f2dfb5574b8",
}


def runtime_identity():
    return {"regex": regex.__version__, "tokenizers": tokenizers.__version__,
            "unicode": unicodedata.unidata_version}


def implementation_identity(*names):
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in names
    }


def encoder_identity():
    return implementation_identity("mom_backend.py", "bpe_core.py", "orthographic_units.py", "unicode_ml.py")


def compatible_encoder(recorded):
    current = encoder_identity()
    if recorded == current:
        return True
    return (isinstance(recorded, dict) and recorded.keys() == current.keys()
            and recorded["mom_backend.py"] == _REVIEWED_ENCODER
            and recorded["bpe_core.py"] in _REVIEWED_CORES
            and all(recorded[k] == current[k] for k in ("orthographic_units.py", "unicode_ml.py")))


def preprocessing(normalizer, pretokenizer):
    transport = Tokenizer(models.BPE())
    transport.normalizer = normalizer.hf()
    transport.pre_tokenizer = pretokenizer.hf()
    return transport


def pretokenizer_split(pretokenizer, text):
    return [piece for piece, _ in pretokenizer.pre_tokenize_str(text)]


@dataclass
class SurfaceEncoding:
    text: str
    ids: list
    tokens: list
    offsets: list


class MOMBackend(Backend):
    def __init__(self, artifact):
        if (artifact["schema"] != SCHEMA or artifact["runtime"] != runtime_identity()
                or not compatible_encoder(artifact["encoder"])):
            raise ValueError("MOM-BPE artifact schema, encoder, or Unicode runtime is incompatible")
        self.artifact = artifact
        self.name = "mom_bpe"
        self._unitize = UNITIZERS[artifact["segmenter"]]
        self._transport = Tokenizer.from_str(json.dumps(artifact["preprocessing"], ensure_ascii=False))
        specials = artifact["specials"]
        initial = artifact["initial"]
        merges = [tuple(pair) for pair in artifact["merges"]]
        self.special_ids = dict(zip(specials, range(len(specials))))
        if (len(self.special_ids) != len(specials) or len(set(initial)) != len(initial)
                or not all(isinstance(s, str) and s for s in initial)):
            raise ValueError("Invalid MOM-BPE symbol table")
        self.unk_id = self.special_ids["<unk>"]
        self.eos_id = self.special_ids["</s>"]
        self._byte_start = len(specials)
        self._lexical_start = self._byte_start + 256
        self.fallback_ids = frozenset(range(self._byte_start, self._lexical_start))
        learned = list(initial)
        seen = set(initial)
        self._ranks = {}
        for a, b in merges:
            if a not in seen or b not in seen or a + b in seen:
                raise ValueError("MOM-BPE merge refers to invalid or reused symbols")
            self._ranks[a, b] = len(self._ranks)
            learned.append(a + b)
            seen.add(a + b)
        self._lexical_ids = {piece: i for i, piece in enumerate(learned, self._lexical_start)}
        self._initial = frozenset(initial)
        self._pieces = list(specials) + byte_tokens() + learned
        self.vocab_size = artifact["vocab_size"]
        if len(self._pieces) > self.vocab_size:
            raise ValueError("MOM-BPE symbol table exceeds the vocabulary budget")

    def expected(self, text):
        return self._transport.normalizer.normalize_str(text)

    def encode_offsets(self, text):
        text = self.expected(text)
        ids, tokens, offsets = [], [], []
        pos = 0
        def split(s):
            return pretokenizer_split(self._transport.pre_tokenizer, s)

        for word in merge_domains(text, split):
            for known, units in groupby(self._unitize(word), key=self._initial.__contains__):
                pieces = merge_units(list(units), self._ranks) if known else units
                for piece in pieces:
                    end = pos + len(piece)
                    if known:
                        encoded = [self._lexical_ids[piece]]
                    else:
                        encoded = [self._byte_start + b for b in piece.encode("utf-8")]
                    ids.extend(encoded)
                    tokens.extend(self._pieces[i] for i in encoded)
                    offsets.extend([(pos, end)] * len(encoded))
                    pos = end
        return SurfaceEncoding(text, ids, tokens, offsets)

    def encode_batch(self, texts):
        return [self.encode_offsets(t).ids for t in texts]

    def tokens(self, text):
        return self.encode_offsets(text).tokens

    def decode(self, ids):
        """Decode allocated IDs. Reserved controls decode to their exact spelling."""
        raw = bytearray()
        for i in ids:
            if isinstance(i, bool) or not isinstance(i, Integral):
                raise TypeError("MOM-BPE token IDs must be integers")
            i = int(i)
            if not 0 <= i < self.vocab_size:
                raise ValueError("MOM-BPE token ID is out of range")
            if i >= len(self._pieces):
                raise ValueError("MOM-BPE token ID is unallocated")
            if i in self.fallback_ids:
                raw.append(i - self._byte_start)
            else:
                raw.extend(self._pieces[i].encode("utf-8"))
        return raw.decode("utf-8", errors="replace")

    @classmethod
    def load(cls, out_dir):
        return cls(json.loads((Path(out_dir) / ARTIFACT).read_bytes()))

    def save(self, out_dir):
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / ARTIFACT
        data = (json.dumps(self.artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with atomic_writer(path, overwrite=False) as output:
                output.write(data)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FileExistsError("A different MOM-BPE artifact already exists")

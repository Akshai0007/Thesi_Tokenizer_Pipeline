"""Invertible uniform word-prefix transport and EventBackend v2 for lossless M48-Guard.

Invertibility proof for (encode_transport, decode_transport):
Let SP = U+2581 ('▁') and ESC = U+2582 ('▂').
For any canonical input string s in Sigma*:
1. If s == "", encode_transport("") returns "".
2. If s != "", encode_transport(s) prepends a single document-start SP followed by a
   left-to-right prefix-free encoding of each character c in s:
     - c == ESC -> ESC + ESC
     - c == SP  -> ESC + SP
     - c == " " -> SP
     - else     -> c
   Therefore:
   - Every non-empty s produces a non-empty transport starting with SP.
   - Every literal ESC or SP in s is prefixed by an odd-parity ESC, so no literal SP
     can ever appear unescaped in the transport body.
   - Every ASCII space " " in s becomes an unescaped SP in the transport body.
3. decode_transport(t) inverts this bijectively:
   - Empty t maps back to "".
   - Non-empty t strips the single document-start SP (when present) and scans the body
     left-to-right: ESC+ESC -> ESC, ESC+SP -> SP, unescaped SP -> " ", any other c -> c.
Hence decode_transport(encode_transport(s)) == s for every Unicode string s.

Offset projection:
`encode_transport_with_offsets(s)` tracks (orig_start, orig_end) in `s` for every
character position in `t` (with doc-start SP anchored at (0, 0)), so any piece
covering t[a:b] projects cleanly back to (starts[a], ends[b - 1]) in `s`.
"""

import json
from itertools import groupby
from pathlib import Path

from tokenizers import Tokenizer

from .atomic import atomic_writer
from .bpe_core import byte_tokens
from .bpe_progress import canonical
from .hybrid_units import initial_units
from .mom_backend import (
    SurfaceEncoding,
    encoder_identity,
    implementation_identity,
    pretokenizer_split,
    runtime_identity,
)
from .orthographic_units import merge_domains
from .picky_backend import EventBackend, event_encoder_identity
from .picky_events import EventProgram

SP = "\u2581"
ESC = "\u2582"
ARTIFACT = "picky-bpe-v2.json"
SCHEMA = "malayalam-event-bpe-v2"


def encode_transport(text: str) -> str:
    """Map plain text into a 100% lossless uniform-prefix transport string."""
    if not text:
        return ""
    out = [SP]
    for ch in text:
        if ch == ESC:
            out.append(ESC + ESC)
        elif ch == SP:
            out.append(ESC + SP)
        elif ch == " ":
            out.append(SP)
        else:
            out.append(ch)
    return "".join(out)


def encode_transport_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    """Map plain text to transport text plus per-character (start, end) spans in `text`."""
    if not text:
        return "", [], []
    out = [SP]
    starts = [0]
    ends = [0]
    for i, ch in enumerate(text):
        if ch == ESC:
            out.extend((ESC, ESC))
            starts.extend((i, i))
            ends.extend((i + 1, i + 1))
        elif ch == SP:
            out.extend((ESC, SP))
            starts.extend((i, i))
            ends.extend((i + 1, i + 1))
        elif ch == " ":
            out.append(SP)
            starts.append(i)
            ends.append(i + 1)
        else:
            out.append(ch)
            starts.append(i)
            ends.append(i + 1)
    return "".join(out), starts, ends


def decode_transport(transport_text: str) -> str:
    """Invert `encode_transport` back to the exact canonical plain text."""
    if not transport_text:
        return ""
    body = transport_text[1:] if transport_text.startswith(SP) else transport_text
    out = []
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == ESC:
            if i + 1 < n and body[i + 1] in (ESC, SP):
                out.append(body[i + 1])
                i += 2
            else:
                out.append(ch)
                i += 1
        elif ch == SP:
            out.append(" ")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def uniform_prefix_encoder_identity():
    return {
        **event_encoder_identity(),
        **encoder_identity(),
        **implementation_identity("uniform_prefix.py"),
    }


class UniformPrefixEventBackend(EventBackend):
    """Event BPE backend with invertible uniform-prefix transport and exact surface offsets."""

    def __init__(self, artifact):
        if (
            artifact["schema"] != SCHEMA
            or artifact["runtime"] != runtime_identity()
            or artifact["encoder"] != uniform_prefix_encoder_identity()
            or not artifact.get("uniform_prefix", False)
        ):
            raise ValueError("Uniform-prefix Event BPE artifact or Unicode runtime is incompatible")
        self.artifact = artifact
        self.name = "boundary_guarded_hybrid_v2"
        self._transport = Tokenizer.from_str(json.dumps(artifact["preprocessing"], ensure_ascii=False))
        self._program = EventProgram(artifact["initial"], artifact["events"])
        if not set(artifact["atoms"]) <= self._program.initial:
            raise ValueError("Protected atoms are missing from the initial alphabet")
        self._unitize = initial_units(artifact["segmenter"], artifact["atoms"])
        specials = artifact["specials"]
        self.special_ids = {token: i for i, token in enumerate(specials)}
        if len(self.special_ids) != len(specials):
            raise ValueError("Duplicate event BPE special IDs")
        self._control_ids = frozenset(self.special_ids.values())
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
        transport_text, starts, ends = encode_transport_with_offsets(canonical_text)
        ids, tokens, offsets = [], [], []
        t_pos = 0

        def presplit(surface):
            return pretokenizer_split(self._transport.pre_tokenizer, surface)

        for word in merge_domains(transport_text, presplit):
            for known, units in groupby(self._unitize(word), self._program.initial.__contains__):
                pieces = self._program.replay(list(units)) if known else units
                for piece in pieces:
                    t_end = t_pos + len(piece)
                    orig_start = starts[t_pos]
                    orig_end = ends[t_end - 1]
                    encoded = (
                        [self._lexical_ids[piece]]
                        if known
                        else [self._byte_start + b for b in piece.encode("utf-8")]
                    )
                    ids.extend(encoded)
                    tokens.extend(self._pieces[i] for i in encoded)
                    offsets.extend([(orig_start, orig_end)] * len(encoded))
                    t_pos = t_end
        return SurfaceEncoding(canonical_text, ids, tokens, offsets)

    def decode(self, ids):
        """Decode allocated IDs; invert uniform-prefix transport on non-control runs."""
        from numbers import Integral

        out = []
        raw = bytearray()

        def flush_raw():
            if raw:
                out.append(decode_transport(raw.decode("utf-8", errors="replace")))
                raw.clear()

        for i in ids:
            if isinstance(i, bool) or not isinstance(i, Integral):
                raise TypeError("MOM-BPE token IDs must be integers")
            i = int(i)
            if not 0 <= i < self.vocab_size:
                raise ValueError("MOM-BPE token ID is out of range")
            if i >= len(self._pieces):
                raise ValueError("MOM-BPE token ID is unallocated")
            if i in self._control_ids:
                flush_raw()
                out.append(self._pieces[i])
            elif i in self.fallback_ids:
                raw.append(i - self._byte_start)
            else:
                raw.extend(self._pieces[i].encode("utf-8"))
        flush_raw()
        return "".join(out)

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
                raise ValueError("An incompatible uniform-prefix Event BPE artifact already exists") from None

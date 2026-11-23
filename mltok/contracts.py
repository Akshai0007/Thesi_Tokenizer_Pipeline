"""CONTRACTS - the full interface between the engine and every plugin kind.

A plugin may import from: this module, `mltok.registry`, `mltok.lib.*`, and any
third-party library. A plugin must NEVER import another plugin. That rule (checked by
`mltok check` and the test-suite) is what makes every plugin file replaceable on its own.

Conventions shared by all text plugins
--------------------------------------
* The engine, not the plugin, owns the word-start marker "▁" (`mltok.lib.marker`):
  normalizers work on plain text; pre-tokenizers receive text where spaces are already "▁".
* Everything a plugin needs from the outside arrives through ``__init__(**params)``
  (the YAML mapping) or the method arguments. Plugins keep no global state.
* Anything that must be reproducible/cached is derived from the plugin's source file +
  params, so edit the file or the params and exactly the dependent results recompute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


class Plugin:
    """Base class. `name`/`params` are filled in by the registry."""

    name: str = ""
    params: dict = {}


# --------------------------------------------------------------------------- source
class Source(Plugin):
    """WHERE DOCUMENTS COME FROM.

    iter_docs() -> iterator of dicts ``{"text": str, "src": str (optional)}``.
    The engine handles sharding, progress, size limits and resume; the plugin only yields.
    Config: ``sources: [{type: <name>, id: <unique id>, ...params}]``  (``id`` is reserved
    for the engine and is passed to the plugin as ``id``; ``max_docs`` is applied by the engine).
    """

    def __init__(self, id: str = "src", **params):
        self.id = id
        self.params = params

    def iter_docs(self) -> Iterator[dict]:
        raise NotImplementedError


# --------------------------------------------------------------------------- filter
class DocFilter(Plugin):
    """WHICH DOCUMENTS ARE KEPT.  keep(text) -> bool.  Applied in config order."""

    def keep(self, text: str) -> bool:
        raise NotImplementedError


# --------------------------------------------------------------------------- normalizer
class Normalizer(Plugin):
    """TEXT NORMALIZATION, in two equivalent forms.

    normalize(text) -> str                         pure Python (statistics, baselines)
    hf() -> tokenizers.normalizers.Normalizer      shipped inside tokenizer.json

    The two MUST agree on every input (the contract check compares them). Build `hf()`
    only from `tokenizers.normalizers` primitives so the exported tokenizer needs no custom code.
    """

    def normalize(self, text: str) -> str:
        raise NotImplementedError

    def hf(self):
        raise NotImplementedError


# --------------------------------------------------------------------------- pretokenizer
class PreTokenizer(Plugin):
    """PRE-TOKENIZATION: which pieces the algorithm may merge inside.

    split(marked_text) -> list[str]   pieces whose concatenation equals the input exactly
    hf() -> tokenizers.pre_tokenizers.PreTokenizer   same split, as an HF component
    Input text already has spaces replaced by the marker "▁".
    """

    def split(self, marked_text: str) -> list[str]:
        raise NotImplementedError

    def hf(self):
        raise NotImplementedError


# --------------------------------------------------------------------------- backend
class Backend:
    """What every trained tokenizer exposes to metrics / proxy models (engine-facing)."""

    name: str = "tokenizer"
    vocab_size: int = 0
    unk_id: int | None = None
    eos_id: int | None = None

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        raise NotImplementedError

    def tokens(self, text: str) -> list[str] | None:
        """Token strings with "▁" as word-start marker, or None if not available."""
        return None

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def expected(self, text: str) -> str:
        """What decode(encode(text)) should return (usually the normalized text)."""
        return text

    def save(self, out_dir) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------- algorithm
@dataclass
class TrainContext:
    """Everything the engine hands to an algorithm. Algorithms read; they never reach outside."""

    vocab: int
    specials: list[str]
    normalizer: Normalizer
    pretokenizer: PreTokenizer | None
    out_dir: Path
    word_counts: dict[str, int] | None  # marked words -> count (None if needs_word_counts=False)
    shard_paths: list[str]  # cleaned training shards (jsonl[.gz])
    resources: dict = field(default_factory=dict)  # threads, max_word_types, spm_max_mb, boundaries...
    log: Callable[[str], None] = print


class Algorithm(Plugin):
    """TOKENIZER TRAINING ALGORITHM.

    needs_word_counts   engine pre-computes the word table (normalizer + pretokenizer) if True
    uses_pretokenizer   False -> engine ignores the pretokenizer setting for this algorithm
    train(ctx) -> (Backend, info_dict)      writes its artifacts into ctx.out_dir
    load(out_dir, normalizer) -> Backend    re-opens what train() wrote (used by later stages)
    contract_vocab      vocabulary size used by the contract check on a tiny corpus
    """

    needs_word_counts: bool = True
    uses_pretokenizer: bool = True
    contract_vocab: int = 1200

    def train(self, ctx: TrainContext) -> tuple[Backend, dict]:
        raise NotImplementedError

    def load(self, out_dir, normalizer: Normalizer) -> Backend:
        raise NotImplementedError


# --------------------------------------------------------------------------- metric
class Metric(Plugin):
    """INTRINSIC EVALUATION.  compute(backend, docs) -> flat dict of JSON-serialisable values.

    `docs` is a list of (raw_text, source_id). Keys from different metrics are merged, so
    use distinctive key names. Metrics may call backend.encode_batch / tokens / decode.
    """

    def compute(self, backend: Backend, docs: list[tuple[str, str]]) -> dict:
        raise NotImplementedError


# --------------------------------------------------------------------------- proxy model
class ProxyModel(Plugin):
    """ARCHITECTURE OF THE PROXY LANGUAGE MODEL.

    build(vocab_size, seq_len) -> torch.nn.Module with forward(idx[B,T] long) -> logits[B,T,V].
    The engine owns data, optimizer, schedule, mixed precision and bits-per-byte scoring.
    """

    def build(self, vocab_size: int, seq_len: int):
        raise NotImplementedError

    def n_embedding_params(self, model) -> int:
        import torch.nn as nn

        return sum(p.numel() for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters())


# --------------------------------------------------------------------------- selector
class Selector(Plugin):
    """HOW WE DECIDE.  All "lower is better".

    eligible(metrics) -> bool        may this variant be chosen at all?
    score(metrics) -> float          intrinsic ranking (decides who gets a proxy run)
    proxy_score(proxy_result) -> float   final ranking (decides the winner)
    """

    def eligible(self, metrics: dict) -> bool:
        raise NotImplementedError

    def score(self, metrics: dict) -> float:
        raise NotImplementedError

    def proxy_score(self, proxy: dict) -> float:
        raise NotImplementedError


# --------------------------------------------------------------------------- store
class StoreAccessError(RuntimeError):
    """A safe-to-display storage error with no credentials or server response body."""


class Store(Plugin):
    """WHERE CHECKPOINTS GO (and where a restart restores from).

    push(files, message)            files = [(local_path, remote_relpath), ...] uploaded as ONE batch/commit
    pull(remote_prefix, local_dir)  fetch every remote file under `remote_prefix` into `local_dir`
                                    (relative layout preserved, existing local files are NOT overwritten);
                                    returns the number of files fetched
    delete(remote_paths)            optional clean-up of obsolete remote files (default: no-op)

    `remote_relpath` is relative to the store's own root; the store adds its own prefix/bucket/repo.
    Stores must be safe to call often and must never write outside their own prefix.
    The engine treats every store error as non-fatal (logs it, keeps the files queued, retries later).
    Config: ``store: {type: <name>, ...params}``  (default: ``none``).
    """

    #: False for the no-op store (the contract check then skips the round-trip test)
    enabled = True
    # Optional capability: revision(create=False), list_at(revision),
    # fetch_at(relative_path, revision, staging_directory), push_at(files, message, parent).
    verified_snapshots = False
    #: optional dict used by the contract check; "__TMP__" is replaced by a temp directory
    example_params = None

    def push(self, files: list, message: str = "") -> None:
        raise NotImplementedError

    def pull(self, remote_prefix: str, local_dir, skip_existing: bool = True) -> int:
        raise NotImplementedError

    def delete(self, remote_paths: list) -> None:
        return None

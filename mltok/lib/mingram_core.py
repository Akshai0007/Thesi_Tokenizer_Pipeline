"""Core Viterbi decoder, Hard-EM Unigram trainer, MinGram backends, and HF Unigram export."""

import json
import math
from collections import Counter
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, normalizers

from .atomic import atomic_writer
from .backends import save_hf_tokenizer
from .bpe_core import byte_tokens
from .bpe_progress import canonical
from .mom_backend import (
    MOMBackend,
    SurfaceEncoding,
    implementation_identity,
    pretokenizer_split,
    runtime_identity,
)
from .orthographic_units import UNITIZERS, merge_domains
from .unicode_ml import ML_SET, ZWJ, ZWNJ, is_combining
from .uniform_prefix import (
    ESC,
    SP,
    decode_transport,
    encode_transport_with_offsets,
)

ARTIFACT = "ml-mingram.json"
SCHEMA = "malayalam-mingram-v1"
_TRIE_LEAF = None
_INF = float("inf")


def mingram_encoder_identity():
    return implementation_identity(
        "mingram_core.py",
        "mom_backend.py",
        "orthographic_units.py",
        "unicode_ml.py",
        "uniform_prefix.py",
    )


def build_trie(log_probs: dict[str, float]) -> dict:
    """Build a character prefix trie mapping token strings to (token, -log_prob)."""
    trie: dict = {}
    for token, log_prob in log_probs.items():
        if not token:
            continue
        node = trie
        for ch in token:
            nxt = node.get(ch)
            if nxt is None:
                nxt = {}
                node[ch] = nxt
            node = nxt
        node[_TRIE_LEAF] = (token, -float(log_prob))
    return trie


def guarded_akshara_positions(piece: str) -> frozenset[int]:
    """Return character boundary indices strictly inside multi-char Malayalam aksharas."""
    if not piece or not any(c in ML_SET for c in piece):
        return frozenset()
    guarded: list[int] = []
    position = 0
    for orth in UNITIZERS["akshara"](piece):
        span = len(orth)
        if span > 1 and any(c in ML_SET for c in orth):
            guarded.extend(range(position + 1, position + span))
        position += span
    return frozenset(guarded)


def viterbi_segment(
    piece: str,
    log_probs: dict[str, float],
    *,
    trie: dict | None = None,
    forbidden: str | None = None,
    fallback_cost: float = 20.0,
    akshara_guard_milli: int = 1000,
    guarded: frozenset[int] | None = None,
) -> tuple[list[str], float]:
    """Decode `piece` with minimum negative log-likelihood and deterministic tie-breaking.

    Returns (tokens, total_cost) where `"".join(tokens) == piece`. Any character not
    covered by an active single-character lexical entry falls back as a 1-char surface
    unit whose cost scales with its UTF-8 byte length.

    When `akshara_guard_milli >= 1000`, segmentation prefers paths with zero cuts
    strictly inside Malayalam aksharas lexicographically over token NLL, falling back
    to minimum-cut paths only when no akshara-preserving path covers `piece`. When
    `0 < akshara_guard_milli < 1000`, a penalty proportional to `akshara_guard_milli / 1000.0`
    is added per guarded internal akshara boundary crossed.
    """
    if type(akshara_guard_milli) is not int or akshara_guard_milli < 0:
        raise ValueError("akshara_guard_milli must be a non-negative integer")
    n = len(piece)
    if n == 0:
        return [], 0.0
    if trie is None:
        trie = build_trie(log_probs)

    guarded_pos = (
        (guarded_akshara_positions(piece) if guarded is None else guarded)
        if akshara_guard_milli > 0
        else frozenset()
    )
    hard_guard = bool(guarded_pos) and akshara_guard_milli >= 1000
    soft_penalty = (
        (akshara_guard_milli / 1000.0) * fallback_cost
        if guarded_pos and 0 < akshara_guard_milli < 1000
        else 0.0
    )

    dp = [_INF] * (n + 1)
    cuts = [n + 1] * (n + 1) if hard_guard else None
    back: list[tuple[int, str] | None] = [None] * (n + 1)
    dp[0] = 0.0
    if cuts is not None:
        cuts[0] = 0

    for i in range(n):
        base = dp[i]
        if base == _INF:
            continue
        base_cuts = cuts[i] if cuts is not None else 0

        def _relax(end: int, cand_tok: str, step_cost: float) -> None:
            is_cut = end in guarded_pos
            if hard_guard:
                cand_cuts = base_cuts + (1 if is_cut else 0)
                cand_cost = base + step_cost
                prev = back[end]
                if cand_cuts < cuts[end] or (
                    cand_cuts == cuts[end]
                    and (cand_cost < dp[end] or (cand_cost == dp[end] and prev is not None and i < prev[0]))
                ):
                    cuts[end] = cand_cuts
                    dp[end] = cand_cost
                    back[end] = (i, cand_tok)
            else:
                cand = base + step_cost + (soft_penalty if is_cut else 0.0)
                prev = back[end]
                if cand < dp[end] or (cand == dp[end] and prev is not None and i < prev[0]):
                    dp[end] = cand
                    back[end] = (i, cand_tok)

        node = trie
        matched_single = False
        for j in range(i, n):
            node = node.get(piece[j])
            if node is None:
                break
            entry = node.get(_TRIE_LEAF)
            if entry is not None:
                tok, tok_cost = entry
                if tok == forbidden:
                    continue
                if j == i:
                    matched_single = True
                _relax(j + 1, tok, tok_cost)
        if not matched_single:
            ch = piece[i]
            _relax(i + 1, ch, fallback_cost * len(ch.encode("utf-8")))

    tokens = []
    pos = n
    while pos > 0:
        step = back[pos]
        if step is None:
            raise RuntimeError(f"Viterbi failed to cover position {pos} in {piece!r}")
        start, tok = step
        tokens.append(tok)
        pos = start
    tokens.reverse()
    total_cost = dp[n] + (cuts[n] * fallback_cost if cuts is not None else 0.0)
    return tokens, total_cost


def is_malayalam_dangling(token: str) -> bool:
    """True when `token` begins (after marker/space) with a Malayalam combining/joiner mark."""
    stripped = token.lstrip(SP + " ")
    if not stripped:
        return False
    head = stripped[0]
    return (head in ML_SET or head in (ZWJ, ZWNJ)) and is_combining(head)


def _normalize_log_probs(counts: Counter, active: set[str], smoothing: float = 0.5) -> dict[str, float]:
    total = sum(counts.get(tok, 0) for tok in active) + smoothing * len(active)
    log_total = math.log(total if total > 0 else 1.0)
    return {
        tok: math.log(counts.get(tok, 0) + smoothing) - log_total
        for tok in active
    }


def _initial_log_probs(word_items: list[tuple[str, int]], active: set[str]) -> dict[str, float]:
    """Initialize Unigram probabilities from trie-matched substring occurrences in `word_items`."""
    trie = build_trie({tok: -1.0 for tok in active})
    counts: Counter = Counter()
    for piece, freq in word_items:
        n = len(piece)
        for i in range(n):
            node = trie
            for j in range(i, n):
                node = node.get(piece[j])
                if node is None:
                    break
                entry = node.get(_TRIE_LEAF)
                if entry is not None:
                    counts[entry[0]] += freq
    return _normalize_log_probs(counts, active, smoothing=1.0)


def _em_step(
    word_items: list[tuple[str, int]],
    active: set[str],
    log_probs: dict[str, float],
    fallback_cost: float,
    *,
    akshara_guard_milli: int = 1000,
    guarded_map: dict[str, frozenset[int]] | None = None,
) -> tuple[Counter, dict[str, float], float]:
    trie = build_trie(log_probs)
    counts: Counter = Counter()
    total_nll = 0.0
    for piece, freq in word_items:
        tokens, cost = viterbi_segment(
            piece,
            log_probs,
            trie=trie,
            fallback_cost=fallback_cost,
            akshara_guard_milli=akshara_guard_milli,
            guarded=guarded_map.get(piece) if guarded_map is not None else None,
        )
        total_nll += freq * cost
        for tok in tokens:
            if tok in active:
                counts[tok] += freq
    return counts, _normalize_log_probs(counts, active), total_nll


def train_mingram_unigram(
    word_counts: dict[str, int],
    seed_tokens: list[str],
    *,
    target_vocab: int,
    protected: set[str],
    shrink_factor: float = 0.75,
    em_iters: int = 2,
    final_em_iters: int = 2,
    filter_malayalam_dangling: bool = True,
    akshara_guard_milli: int = 1000,
) -> tuple[dict[str, float], dict]:
    """Prune `seed_tokens` down to `target_vocab` via Hard-EM Viterbi Unigram training."""
    if target_vocab < 1:
        raise ValueError("target_vocab must be positive")
    if not 0 < shrink_factor <= 1:
        raise ValueError("shrink_factor must be within (0, 1]")
    if type(akshara_guard_milli) is not int or akshara_guard_milli < 0:
        raise ValueError("akshara_guard_milli must be a non-negative integer")

    protected_set = set(protected)
    if len(protected_set) > target_vocab:
        raise ValueError(f"Protected unit count ({len(protected_set)}) exceeds target vocabulary ({target_vocab})")

    dangling_filtered = 0
    active_list = []
    seen = set()
    for tok in [*protected, *seed_tokens]:
        if not tok or tok in seen:
            continue
        if filter_malayalam_dangling and tok not in protected_set and is_malayalam_dangling(tok):
            dangling_filtered += 1
            continue
        seen.add(tok)
        active_list.append(tok)

    active = set(active_list)
    word_items = sorted(word_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    guarded_map = (
        {piece: pos for piece, _ in word_items if (pos := guarded_akshara_positions(piece))}
        if akshara_guard_milli > 0
        else {}
    )
    tok_guarded = (
        {tok: pos for tok in active if (pos := guarded_akshara_positions(tok))}
        if akshara_guard_milli > 0
        else {}
    )
    log_probs = _initial_log_probs(word_items, active)
    fallback_cost = max(20.0, -min(log_probs.values(), default=-10.0) + 5.0)

    counts = Counter()
    total_nll = 0.0
    for _ in range(max(1, em_iters)):
        counts, log_probs, total_nll = _em_step(
            word_items,
            active,
            log_probs,
            fallback_cost,
            akshara_guard_milli=akshara_guard_milli,
            guarded_map=guarded_map,
        )

    pruning_rounds = 0
    zero_pruned = 0
    loss_pruned = 0

    while len(active) > target_vocab:
        pruning_rounds += 1
        excess = len(active) - target_vocab
        prunable = [tok for tok in active if tok not in protected_set]
        if not prunable:
            break

        zero_tokens = sorted(
            (tok for tok in prunable if counts.get(tok, 0) == 0),
            key=lambda tok: (log_probs[tok], -len(tok), tok),
        )
        if zero_tokens:
            drop_zero = zero_tokens[:excess]
            active.difference_update(drop_zero)
            zero_pruned += len(drop_zero)
            for tok in drop_zero:
                log_probs.pop(tok, None)
            if len(active) == target_vocab:
                break
            excess = len(active) - target_vocab

        prunable = [tok for tok in active if tok not in protected_set]
        if not prunable:
            break

        trie = build_trie(log_probs)
        scored = []
        for tok in prunable:
            cnt = counts.get(tok, 0)
            _, alt_cost = viterbi_segment(
                tok,
                log_probs,
                trie=trie,
                forbidden=tok,
                fallback_cost=fallback_cost,
                akshara_guard_milli=akshara_guard_milli,
                guarded=tok_guarded.get(tok, frozenset()),
            )
            marginal_loss = cnt * (alt_cost - (-log_probs[tok]))
            scored.append((marginal_loss, cnt, -len(tok), tok))
        scored.sort()

        target_drop = min(excess, max(1, int(math.ceil(excess * shrink_factor))))
        to_drop = [row[-1] for row in scored[:target_drop]]
        active.difference_update(to_drop)
        loss_pruned += len(to_drop)
        for tok in to_drop:
            log_probs.pop(tok, None)

        for _ in range(max(1, em_iters)):
            counts, log_probs, total_nll = _em_step(
                word_items,
                active,
                log_probs,
                fallback_cost,
                akshara_guard_milli=akshara_guard_milli,
                guarded_map=guarded_map,
            )

    for _ in range(max(0, final_em_iters)):
        counts, log_probs, total_nll = _em_step(
            word_items,
            active,
            log_probs,
            fallback_cost,
            akshara_guard_milli=akshara_guard_milli,
            guarded_map=guarded_map,
        )

    # Preserve canonical descending probability order (protected units remain included).
    ordered = sorted(active, key=lambda tok: (-log_probs[tok], -len(tok), tok))
    final_scores = {tok: float(log_probs[tok]) for tok in ordered}
    stats = {
        "seed_tokens_input": len(seed_tokens),
        "dangling_malayalam_filtered": dangling_filtered,
        "protected_tokens": len(protected_set),
        "final_lexical_tokens": len(final_scores),
        "pruning_rounds": pruning_rounds,
        "zero_count_pruned": zero_pruned,
        "marginal_loss_pruned": loss_pruned,
        "final_em_nll": total_nll,
        "unused_lexical_train": sum(1 for tok in final_scores if counts.get(tok, 0) == 0),
        "akshara_guard_milli": akshara_guard_milli,
    }
    return final_scores, stats


class MinGramBackend(MOMBackend):
    """Lossless Viterbi Unigram backend over pretokenized `merge_domains`."""

    def __init__(self, artifact):
        if (
            artifact["schema"] != SCHEMA
            or artifact["runtime"] != runtime_identity()
            or artifact["encoder"] != mingram_encoder_identity()
        ):
            raise ValueError("MinGram artifact schema, encoder, or Unicode runtime is incompatible")
        self.artifact = artifact
        self.name = "ml_mingram"
        self._uniform_prefix = bool(artifact.get("uniform_prefix", False))
        self._transport = Tokenizer.from_str(json.dumps(artifact["preprocessing"], ensure_ascii=False))
        specials = artifact["specials"]
        self.special_ids = {token: i for i, token in enumerate(specials)}
        if len(self.special_ids) != len(specials):
            raise ValueError("Duplicate MinGram special IDs")
        self._control_ids = frozenset(self.special_ids.values())
        self.unk_id, self.eos_id = (self.special_ids[key] for key in ("<unk>", "</s>"))
        self._byte_start = len(specials)
        lexical_start = self._byte_start + 256
        self.fallback_ids = frozenset(range(self._byte_start, lexical_start))

        scores = artifact["scores"]
        if not isinstance(scores, dict) or not scores:
            raise ValueError("MinGram artifact must contain nonempty lexical scores")
        self._log_probs = {tok: float(val) for tok, val in scores.items()}
        self._trie = build_trie(self._log_probs)
        self._fallback_cost = float(artifact.get("fallback_cost", 20.0))
        raw_guard = artifact.get("akshara_guard_milli", 1000)
        if type(raw_guard) is not int or raw_guard < 0:
            raise ValueError("MinGram artifact akshara_guard_milli must be a non-negative integer")
        self._akshara_guard_milli = raw_guard
        # Sort deterministically so save() (canonical sort_keys=True) never alters IDs on load().
        vocabulary = sorted(self._log_probs, key=lambda tok: (-self._log_probs[tok], -len(tok), tok))
        self._lexical_ids = {token: i for i, token in enumerate(vocabulary, lexical_start)}
        self._pieces = [*specials, *byte_tokens(), *vocabulary]
        self.vocab_size = artifact["vocab_size"]
        if len(self._pieces) > self.vocab_size:
            raise ValueError("MinGram artifact exceeds its target vocabulary")

    def encode_offsets(self, text):
        canonical_text = self.expected(text)
        ids, tokens, offsets = [], [], []
        position = 0

        def presplit(surface):
            return pretokenizer_split(self._transport.pre_tokenizer, surface)

        for word in merge_domains(canonical_text, presplit):
            pieces, _ = viterbi_segment(
                word,
                self._log_probs,
                trie=self._trie,
                fallback_cost=self._fallback_cost,
                akshara_guard_milli=self._akshara_guard_milli,
            )
            for piece in pieces:
                end = position + len(piece)
                if piece in self._lexical_ids:
                    encoded = [self._lexical_ids[piece]]
                else:
                    encoded = [self._byte_start + b for b in piece.encode("utf-8")]
                ids.extend(encoded)
                tokens.extend(self._pieces[i] for i in encoded)
                offsets.extend([(position, end)] * len(encoded))
                position = end
        return SurfaceEncoding(canonical_text, ids, tokens, offsets)

    @classmethod
    def load(cls, out_dir):
        artifact = json.loads((Path(out_dir) / ARTIFACT).read_bytes())
        if artifact.get("uniform_prefix", False):
            return UniformPrefixMinGramBackend(artifact)
        return cls(artifact)

    def save(self, out_dir):
        path = Path(out_dir) / ARTIFACT
        path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical(self.artifact) + b"\n"
        try:
            with atomic_writer(path, overwrite=False) as writer:
                writer.write(data)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("An incompatible MinGram artifact already exists") from None


class UniformPrefixMinGramBackend(MinGramBackend):
    """MinGram backend using invertible uniform word-prefix transport and exact offsets."""

    def encode_offsets(self, text):
        canonical_text = self.expected(text)
        transport_text, starts, ends = encode_transport_with_offsets(canonical_text)
        ids, tokens, offsets = [], [], []
        t_pos = 0

        def presplit(surface):
            return pretokenizer_split(self._transport.pre_tokenizer, surface)

        for word in merge_domains(transport_text, presplit):
            pieces, _ = viterbi_segment(
                word,
                self._log_probs,
                trie=self._trie,
                fallback_cost=self._fallback_cost,
                akshara_guard_milli=self._akshara_guard_milli,
            )
            for piece in pieces:
                t_end = t_pos + len(piece)
                orig_start = starts[t_pos]
                orig_end = ends[t_end - 1]
                if piece in self._lexical_ids:
                    encoded = [self._lexical_ids[piece]]
                else:
                    encoded = [self._byte_start + b for b in piece.encode("utf-8")]
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


def load_mingram_backend(out_dir) -> MinGramBackend:
    return MinGramBackend.load(out_dir)


_HF_EXPORT_PROBES = [
    "",
    " ",
    "  ",
    "   \n\t  ",
    "\r\n",
    "leading and trailing ",
    "കേരളം India\n കേരളം  മലയാളം",
    "ൻ്റ ൎക്കു് ക് ങ്ങൾ ക്\u200dഷ ക്\u200cഷ സദ്\u200cവാരം അവന്\u200d",
    "▁",
    "▁മലയാളം ▁English  ▁ ",
    "▂ ▂▁ ▁▂ ▂▂▁ ▂▂▂▂▁",
    "<|endoftext|> <bos> <eos> <unk> <pad> <s> </s> <|im_start|> <|im_end|>",
    "漢字 𐍈 ☃ 😀 🎉",
]


def export_hf_unigram_json(
    backend: MinGramBackend,
    out_dir,
    *,
    hf_normalizer,
    hf_pretokenizer,
    specials: list[str],
    uniform_prefix: bool,
) -> dict:
    """Export native HF `models.Unigram` `tokenizer.json` and verify lossless round-trip."""
    min_score = min(backend._log_probs.values())
    byte_score = min_score - 20.0
    special_score = byte_score - 1000.0
    vocab: list[tuple[str, float]] = [(tok, special_score) for tok in specials]
    vocab.extend((tok, byte_score) for tok in byte_tokens())
    ordered_lexical = sorted(backend._log_probs, key=lambda tok: (-backend._log_probs[tok], -len(tok), tok))
    vocab.extend((tok, float(backend._log_probs[tok])) for tok in ordered_lexical)
    unk_id = specials.index("<unk>")

    tok = Tokenizer(models.Unigram(vocab, unk_id=unk_id, byte_fallback=True))
    if uniform_prefix:
        tok.normalizer = normalizers.Sequence([
            hf_normalizer,
            normalizers.Replace(ESC, ESC + ESC),
            normalizers.Replace(SP, ESC + SP),
            normalizers.Prepend(SP),
            normalizers.Replace(" ", SP),
        ])
        tok.pre_tokenizer = hf_pretokenizer
        tok.decoder = decoders.Sequence([
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(content=SP, left=1, right=0),
            decoders.Replace(Regex(rf"(?<=(?:^|[^{ESC}])(?:{ESC}{ESC})*){SP}"), " "),
            decoders.Replace(ESC + SP, SP),
            decoders.Replace(ESC + ESC, ESC),
        ])
    else:
        tok.normalizer = hf_normalizer
        tok.pre_tokenizer = hf_pretokenizer
        tok.decoder = decoders.Sequence([
            decoders.ByteFallback(),
            decoders.Fuse(),
        ])

    save_hf_tokenizer(tok, out_dir, specials)
    loaded = Tokenizer.from_file(str(Path(out_dir) / "tokenizer.json"))
    control_ids = set(range(len(specials)))
    mismatches = []
    for sample in _HF_EXPORT_PROBES:
        enc = loaded.encode(sample, add_special_tokens=False)
        decoded = loaded.decode(enc.ids, skip_special_tokens=False)
        expected = backend.expected(sample)
        if decoded != expected or not control_ids.isdisjoint(enc.ids):
            mismatches.append({"input": sample, "expected": expected, "decoded": decoded})

    return {
        "hf_export_lossless_verified": len(mismatches) == 0,
        "probes_checked": len(_HF_EXPORT_PROBES),
        "mismatches": mismatches[:5],
    }

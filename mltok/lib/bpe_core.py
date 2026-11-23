"""Generic, constraint-aware BPE trainer + exporter to a plain HF tokenizer.

Language-specific knowledge is NOT in here: the algorithm plugin passes
  token_ok(token_str) -> bool      reject merges whose result is invalid
  seed_pairs                       merges applied first, regardless of frequency
  boundaries / boundary_lambda     optional training-time morpheme-boundary penalty
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Callable

from tokenizers import Tokenizer, decoders, models

from .marker import SP

KEY_BITS = 21
KEY_MASK = (1 << KEY_BITS) - 1
BLOCK = KEY_MASK  # id for characters outside the alphabet (never merged; byte fallback at inference)
CONFIDENCE_SCALE = 1000

DEFAULT_SPECIALS = [
    "<pad>", "<s>", "</s>", "<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
]


@dataclass
class BPEResult:
    alphabet: list[str]
    tokens: list[str]
    merges: list[tuple[str, str]]
    info: dict = field(default_factory=dict)


def train_bpe(
    word_counts: dict[str, int],
    *,
    vocab_target: int,
    n_specials: int,
    alphabet: list[str],
    token_ok: Callable[[str], bool] | None = None,
    seed_pairs: list[tuple[str, str]] | None = None,
    seed_mode: str = "all",  # all | seen
    max_token_len: int = 0,
    min_pair_count: int = 2,
    max_word_types: int = 0,
    boundaries: dict[str, tuple[int, ...]] | None = None,
    boundary_lambda: float = 0.0,
    initial_units: Callable[[str], list[str]] | None = None,
    boundary_weights: dict[str, dict[int, int]] | None = None,
    penalty_milli: int = 0,
    progress=None,
    log_every: int = 2000,
    log=print,
) -> BPEResult:
    t0 = time.time()
    alphabet = list(dict.fromkeys(alphabet))
    n_new = vocab_target - n_specials - 256 - len(alphabet)
    if n_new <= 0:
        raise ValueError("vocab_target too small for specials + bytes + alphabet")
    tok_list = list(alphabet)
    tok2id = {t: i for i, t in enumerate(tok_list)}
    n_alpha = len(alphabet)

    items = sorted(word_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if max_word_types:
        items = items[:max_word_types]
    total_mass = sum(word_counts.values()) or 1
    kept_mass = sum(c for _, c in items)

    seqs: list[list[int]] = []
    freqs: list[int] = []
    flags: list[list[int] | None] = []
    merges: list[tuple[str, str]] = []
    if progress is not None and (initial_units is None or seed_pairs):
        raise ValueError("Merge checkpoints require unique-output, seed-free BPE")
    saved = progress.latest() if progress is not None else None
    resumed_merges = 0
    if saved is not None:
        seqs, flags = saved["seqs"], saved["flags"]
        if len(seqs) != len(items) or len(flags) != len(items) or len(saved["merges"]) > n_new:
            raise ValueError("BPE checkpoint does not match the training dimensions")
        for a, b in saved["merges"]:
            if a not in tok2id or b not in tok2id or a + b in tok2id:
                raise ValueError("Invalid BPE merge in checkpoint")
            tok2id[a + b] = len(tok_list)
            tok_list.append(a + b)
            merges.append((a, b))
        for seq, ff in zip(seqs, flags):
            if any(type(i) is not int or (i != BLOCK and not 0 <= i < len(tok_list)) for i in seq):
                raise ValueError("Invalid BPE sequence in checkpoint")
            if ff is not None and (len(ff) != len(seq)
                                   or any(type(x) is not int or not 0 <= x <= CONFIDENCE_SCALE for x in ff)):
                raise ValueError("Invalid BPE boundary weights in checkpoint")
        freqs = [c for _, c in items]
        resumed_merges = len(merges)
    for w, c in (() if saved is not None else items):
        units = list(w) if initial_units is None else initial_units(w)
        seqs.append([tok2id.get(ch, BLOCK) for ch in units])
        freqs.append(c)
        fl = None
        if penalty_milli and boundary_weights and w in boundary_weights:
            offsets = boundary_weights[w]
            pos = 0
            fl = []
            for unit in units:
                fl.append(offsets.get(pos, 0))
                pos += len(unit)
        elif boundaries:
            bset = boundaries.get(w.lstrip(SP))
            if bset:
                shift = len(w) - len(w.lstrip(SP))
                fl = [0] * len(w)
                for o in bset:
                    j = o + shift
                    if 0 < j < len(w):
                        fl[j] = 1
        flags.append(fl)

    valid_cache: dict[int, bool] = {}

    def valid(a: int, b: int) -> bool:
        k = (a << KEY_BITS) | b
        v = valid_cache.get(k)
        if v is None:
            if a == BLOCK or b == BLOCK:
                v = False
            else:
                s = tok_list[a] + tok_list[b]
                v = (token_ok is None or token_ok(s)) and (
                    not max_token_len or len(s.lstrip(SP)) <= max_token_len
                )
            valid_cache[k] = v
        return v

    stats: dict[int, int] = {}
    cross: dict[int, int] = {}
    where: dict[int, set[int]] = {}
    for wi, s in enumerate(seqs):
        f, fl = freqs[wi], flags[wi]
        for j in range(len(s) - 1):
            a, b = s[j], s[j + 1]
            if not valid(a, b):
                continue
            k = (a << KEY_BITS) | b
            stats[k] = stats.get(k, 0) + f
            ws = where.get(k)
            if ws is None:
                where[k] = {wi}
            else:
                ws.add(wi)
            if fl is not None and fl[j + 1]:
                cross[k] = cross.get(k, 0) + f * fl[j + 1]

    lam = boundary_lambda

    def score(k: int) -> float:
        v = stats.get(k, 0)
        if v <= 0:
            return 0
        if penalty_milli:
            return v * CONFIDENCE_SCALE**2 - penalty_milli * cross.get(k, 0)
        if lam and k in cross:
            return v - lam * cross[k]
        return v

    heap = [(-score(k), k) for k in stats if score(k) > 0]
    heapq.heapify(heap)

    def apply(a: int, b: int, nid: int) -> set[int]:
        k = (a << KEY_BITS) | b
        wis = where.pop(k, ())
        touched: set[int] = set()
        for wi in wis:
            s = seqs[wi]
            n = len(s)
            for j in range(n - 1):
                if s[j] == a and s[j + 1] == b:
                    break
            else:
                continue
            f, fl = freqs[wi], flags[wi]
            for j in range(n - 1):
                kk = (s[j] << KEY_BITS) | s[j + 1]
                if kk in stats:
                    stats[kk] -= f
                    if fl is not None and fl[j + 1]:
                        cross[kk] = cross.get(kk, 0) - f * fl[j + 1]
            ns: list[int] = []
            nfl: list[int] | None = [] if fl is not None else None
            j = 0
            while j < n:
                if j + 1 < n and s[j] == a and s[j + 1] == b:
                    ns.append(nid)
                    if nfl is not None:
                        nfl.append(fl[j])
                    j += 2
                else:
                    ns.append(s[j])
                    if nfl is not None:
                        nfl.append(fl[j])
                    j += 1
            seqs[wi] = ns
            flags[wi] = nfl
            for j in range(len(ns) - 1):
                x, y = ns[j], ns[j + 1]
                if not valid(x, y):
                    continue
                kk = (x << KEY_BITS) | y
                stats[kk] = stats.get(kk, 0) + f
                ws = where.get(kk)
                if ws is None:
                    where[kk] = {wi}
                else:
                    ws.add(wi)
                if nfl is not None and nfl[j + 1]:
                    cross[kk] = cross.get(kk, 0) + f * nfl[j + 1]
                if x == nid or y == nid:
                    touched.add(kk)
        stats.pop(k, None)
        cross.pop(k, None)
        return touched

    def do_merge(a: int, b: int) -> None:
        s = tok_list[a] + tok_list[b]
        nid = tok2id.get(s)
        if nid is None:
            nid = len(tok_list)
            tok_list.append(s)
            tok2id[s] = nid
        touched = apply(a, b, nid)
        merges.append((tok_list[a], tok_list[b]))
        for kk in touched:
            sc = score(kk)
            if sc > 0:
                heapq.heappush(heap, (-sc, kk))

    n_seeded = 0
    aset = set(alphabet)
    for x, y in (seed_pairs or []):
        if x not in aset or y not in aset:
            continue
        a, b = tok2id[x], tok2id[y]
        if not valid(a, b):
            continue
        if seed_mode == "seen" and not where.get((a << KEY_BITS) | b):
            continue
        if len(tok_list) - n_alpha >= n_new:
            break
        do_merge(a, b)
        n_seeded += 1

    def pop_best():
        while heap:
            negsc, k = heapq.heappop(heap)
            cur = score(k)
            if cur <= 0:
                continue
            if (initial_units is not None or penalty_milli) and stats[k] < min_pair_count:
                continue
            if initial_units is not None:
                a, b = k >> KEY_BITS, k & KEY_MASK
                # Reusing an earlier symbol can expose an earlier-ranked pair
                # after its training step. Keep the emitted rank replay exact.
                if tok_list[a] + tok_list[b] in tok2id:
                    continue
            if -negsc != cur:
                heapq.heappush(heap, (-cur, k))
                continue
            return k, cur
        return None

    stop_reason = "vocab_reached"
    while len(tok_list) - n_alpha < n_new:
        best = pop_best()
        if best is None:
            stop_reason = "no_pairs_left"
            break
        k, cur = best
        if initial_units is None and not penalty_milli and cur < min_pair_count:
            stop_reason = f"min_pair_count<{min_pair_count}"
            break
        a, b = k >> KEY_BITS, k & KEY_MASK
        do_merge(a, b)
        if progress is not None and len(merges) % progress.interval == 0:
            progress.save(merges, seqs, flags)
        if log_every and len(merges) % log_every == 0:
            log(
                f"  merge {len(merges):>6} {tok_list[a]!r}+{tok_list[b]!r} score={cur:.0f} "
                f"vocab={len(tok_list) + n_specials + 256} t={time.time() - t0:.0f}s"
            )

    info = {
        "n_merges": len(merges),
        "n_seeded": n_seeded,
        "stop_reason": stop_reason,
        "word_types_used": len(items),
        "word_mass_covered": kept_mass / total_mass,
        "train_seconds": round(time.time() - t0, 1),
    }
    if progress is not None:
        progress.save(merges, seqs, flags)
        info["resumed_merges"] = resumed_merges
    return BPEResult(alphabet=alphabet, tokens=tok_list, merges=merges, info=info)


def byte_tokens() -> list[str]:
    return [f"<0x{b:02X}>" for b in range(256)]


def build_tokenizer(
    result: BPEResult,
    *,
    specials: list[str],
    hf_normalizer,
    hf_pretokenizer,
    pad_to_multiple: int = 64,
) -> Tokenizer:
    vocab: dict[str, int] = {}
    for t in specials + byte_tokens() + result.tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
    i = 0
    while pad_to_multiple and len(vocab) % pad_to_multiple:
        vocab[f"<|unused_{i}|>"] = len(vocab)
        i += 1
    tok = Tokenizer(
        models.BPE(
            vocab=vocab,
            merges=[(a, b) for a, b in result.merges],
            unk_token="<unk>",
            byte_fallback=True,
            fuse_unk=True,
        )
    )
    tok.normalizer = hf_normalizer
    tok.pre_tokenizer = hf_pretokenizer
    tok.decoder = decoders.Sequence(
        [
            decoders.Replace(SP, " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(content=" ", left=1, right=0),
        ]
    )
    tok.add_special_tokens(specials)
    return tok


def reference_encode_word(word: str, rank: dict[tuple[str, str], int]) -> list[str]:
    """Slow, obviously-correct BPE encoder used by tests to cross-check HF."""
    return merge_units(list(word), rank)


def merge_units(units: list[str], rank: dict[tuple[str, str], int]) -> list[str]:
    syms = units.copy()
    while True:
        best = None
        for i in range(len(syms) - 1):
            r = rank.get((syms[i], syms[i + 1]))
            if r is not None and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            return syms
        i = best[1]
        syms[i : i + 2] = [syms[i] + syms[i + 1]]

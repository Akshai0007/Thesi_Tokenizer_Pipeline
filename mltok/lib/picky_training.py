"""Incremental pair counts for chronological merge and split-back BPE."""

import heapq
from collections import Counter, defaultdict

from .bpe_core import CONFIDENCE_SCALE
from .picky_events import EventProgram, live_replacement, merge, split


class CorpusCounts:
    def __init__(self, examples, program, penalty):
        self.program = program
        self.penalty = penalty
        self.items = [
            (program.replay(units), frequency, offsets)
            for units, frequency, offsets in examples
        ]
        self.pairs, self.tokens, self.crossings = Counter(), Counter(), Counter()
        self.by_pair, self.by_token = defaultdict(set), defaultdict(set)
        for i in range(len(self.items)):
            self.update_index(i, 1)
        self.heap = [self.key(pair) for pair in self.pairs]
        heapq.heapify(self.heap)

    def update_index(self, i, direction):
        units, frequency, offsets = self.items[i]
        weight = direction * frequency
        for symbol, count in Counter(units).items():
            self.tokens[symbol] += weight * count
            if direction > 0:
                self.by_token[symbol].add(i)
            else:
                self.by_token[symbol].remove(i)
        pairs = Counter(zip(units, units[1:]))
        if offsets:
            position = 0
            last_self_merge = -2
            for j, pair in enumerate(zip(units, units[1:])):
                position += len(pair[0])
                if pair[0] == pair[1]:
                    if last_self_merge == j - 1:
                        continue
                    last_self_merge = j
                self.crossings[pair] += weight * offsets.get(position, 0)
        for pair, count in pairs.items():
            self.pairs[pair] += weight * count
            if direction > 0:
                self.by_pair[pair].add(i)
            else:
                self.by_pair[pair].remove(i)
        return pairs.keys()

    def key(self, pair):
        score = self.pairs[pair] * CONFIDENCE_SCALE**2 - self.penalty * self.crossings[pair]
        return -score, self.program.order[pair[0]], self.program.order[pair[1]], pair

    def replace(self, indices, transform):
        touched = set()
        merged = 0
        for i in sorted(indices):
            touched.update(self.update_index(i, -1))
            units, frequency, offsets = self.items[i]
            new = transform(units)
            merged += frequency * (len(units) - len(new))
            self.items[i] = new, frequency, offsets
            touched.update(self.update_index(i, 1))
        for pair in touched:
            if self.pairs[pair] <= 0:
                del self.pairs[pair], self.by_pair[pair]
                self.crossings.pop(pair, None)
            else:
                heapq.heappush(self.heap, self.key(pair))
        if len(self.heap) > 4 * len(self.pairs) + 1024:
            self.heap = [self.key(pair) for pair in self.pairs]
            heapq.heapify(self.heap)
        return merged

    def best(self, min_count, refine):
        while self.heap:
            trial = heapq.heappop(self.heap)
            left, right = pair = trial[-1]
            if (self.pairs[pair] >= min_count and trial == self.key(pair)
                    and (refine or left + right not in self.program.active)):
                return pair
        return None


def fit(initial, examples, budget, *, threshold=None, penalty=0, min_count=2, progress=None):
    """Train with the reference release's strict threshold and actual merge count."""
    saved = progress.latest() if progress is not None else None
    if saved is not None and (saved["seqs"] or saved["flags"]):
        raise ValueError("Expected an event-only checkpoint")
    program = EventProgram(initial, saved["merges"] if saved is not None else ())
    if len(program.active) > budget:
        raise ValueError("Event checkpoint exceeds the target vocabulary")
    resumed = len(program.events)
    counts = CorpusCounts(examples, program, penalty)
    refine = threshold is not None and threshold < CONFIDENCE_SCALE
    stop = "vocabulary_reached"
    while len(program.active) < budget:
        if len(program.events) >= 8 * budget:
            raise ValueError("Event budget exhausted. Checkpoints remain available.")
        pair = counts.best(min_count, refine)
        if pair is None:
            stop = "no_pairs_left"
            break
        left, right = pair
        before = len(program.events)
        program.append("merge", left, right)
        merged = counts.replace(counts.by_pair[pair], lambda units: merge(units, left, right))
        for symbol in dict.fromkeys(pair) if refine else ():
            if (symbol in program.initial
                    or merged * CONFIDENCE_SCALE <= threshold * (counts.tokens[symbol] + merged)):
                continue
            replacement = live_replacement(symbol, program.parents, program.active)
            program.append("remove", symbol, replacement)
            counts.replace(counts.by_token.get(symbol, ()), lambda units: split(units, symbol, replacement))
        if progress is not None and before // progress.interval != len(program.events) // progress.interval:
            progress.save(program.events, [], [])
    if progress is not None:
        progress.save(program.events, [], [])
    standalone = [counts.tokens[piece] for piece in program.vocabulary()]
    return program, {
        "n_merges": sum(event[0] == "merge" for event in program.events),
        "n_removed": sum(event[0] == "remove" for event in program.events),
        "stop_reason": stop, "resumed_events": resumed, "unused_lexical_train": standalone.count(0),
        "used_once_lexical_train": standalone.count(1), "standalone_train_tokens": sum(standalone),
    }

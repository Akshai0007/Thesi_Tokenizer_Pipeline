"""Chronological BPE merge and removal replay over protected initial symbols."""

from bisect import bisect_left
from collections import defaultdict


def merge(sequence, left, right):
    result = []
    i = 0
    while i < len(sequence):
        if i + 1 < len(sequence) and sequence[i] == left and sequence[i + 1] == right:
            result.append(left + right)
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    return result


def split(sequence, symbol, replacement):
    return [piece for current in sequence for piece in (replacement if current == symbol else (current,))]


def live_replacement(symbol, parents, active):
    leaves, pending = [], list(reversed(parents[symbol]))
    while pending:
        piece = pending.pop()
        if piece in active:
            leaves.append(piece)
        else:
            pending.extend(reversed(parents[piece]))
    return leaves


class EventProgram:
    def __init__(self, initial, events=()):
        if len(set(initial)) != len(initial) or not all(isinstance(s, str) and s for s in initial):
            raise ValueError("Invalid event BPE initial alphabet")
        self.initial = set(initial)
        self.active = set(initial)
        self.order = {symbol: rank for rank, symbol in enumerate(initial)}
        self.parents = {}
        self.events = []
        self.merges = defaultdict(list)
        self.removals = defaultdict(list)
        for kind, symbol, arg in events:
            self.append(kind, symbol, arg)

    def append(self, kind, symbol, arg):
        rank = len(self.events)
        if kind == "merge":
            if symbol not in self.active or arg not in self.active:
                raise ValueError("Event BPE merge has an inactive input")
            merged = symbol + arg
            self.active.add(merged)
            if merged not in self.order:
                self.order[merged] = len(self.order)
                self.parents[merged] = (symbol, arg)
            self.merges[symbol, arg].append(rank)
        elif kind == "remove":
            if (symbol not in self.active or symbol in self.initial or not isinstance(arg, list)
                    or arg != live_replacement(symbol, self.parents, self.active)):
                raise ValueError("Event BPE removal has an invalid split-back")
            arg = list(arg)
            self.active.remove(symbol)
            self.removals[symbol].append(rank)
        else:
            raise ValueError("Unknown event BPE event")
        self.events.append((kind, symbol, arg))

    def replay(self, sequence):
        # Lookup only events applicable to this word after the chronological cursor.
        cursor = 0
        while len(self.events) > cursor:
            best = len(self.events)
            candidates = (self.merges.get(pair, ()) for pair in zip(sequence, sequence[1:]))
            for ranks in candidates:
                at = bisect_left(ranks, cursor)
                if at < len(ranks):
                    best = min(best, ranks[at])
            for symbol in sequence:
                ranks = self.removals.get(symbol, ())
                at = bisect_left(ranks, cursor)
                if at < len(ranks):
                    best = min(best, ranks[at])
            if best == len(self.events):
                break
            kind, symbol, arg = self.events[best]
            sequence = merge(sequence, symbol, arg) if kind == "merge" else split(sequence, symbol, arg)
            cursor = best + 1
        return sequence

    def vocabulary(self):
        return [symbol for symbol in self.order if symbol in self.active]

"""Disk-backed corpus grouping and exact five-gram Jaccard verification."""

import json
import sqlite3
from pathlib import Path

from experiments.mom_bpe.corpus_quality import duplicate_view
from experiments.mom_bpe.diagnose import sha

THRESHOLD = 85
SKETCH_SHA256 = "22e8ba23fb337fb7b6fb8529620a1d35b50dd733a305646736ae322e4a8fe627"


def signatures(text):
    return {text[i:i + 5] for i in range(len(text) - 4)}


def prefix(grams, feature_order=None):
    length = len(grams)
    # AllPairs prefix bound under a shared total ordering of features.
    return sorted(grams, key=feature_order)[:length - (THRESHOLD * length + 99) // 100 + 1]


class DuplicateIndex:
    def __init__(self, path, *, feature_order=None):
        self.feature_order = feature_order
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS documents(
            n INTEGER PRIMARY KEY, identity TEXT NOT NULL UNIQUE, source TEXT NOT NULL, group_key TEXT NOT NULL,
            native_group TEXT,
            raw_hash TEXT NOT NULL, canonical_hash TEXT NOT NULL, view_hash TEXT NOT NULL, gram_count INTEGER NOT NULL,
            parent INTEGER, text TEXT NOT NULL, body TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS provenance ON documents(group_key);
        CREATE INDEX IF NOT EXISTS native ON documents(native_group);
        CREATE INDEX IF NOT EXISTS raw ON documents(raw_hash);
        CREATE INDEX IF NOT EXISTS lexical ON documents(canonical_hash);
        CREATE INDEX IF NOT EXISTS screening ON documents(view_hash);
        CREATE TABLE IF NOT EXISTS postings(term TEXT, n INTEGER, PRIMARY KEY(term,n)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS links(n INTEGER, other INTEGER, kind TEXT, PRIMARY KEY(n,other,kind)) WITHOUT ROWID;
        CREATE TEMP TABLE IF NOT EXISTS candidates(n INTEGER PRIMARY KEY);
        """)
        self.sketch = None
        if feature_order is not None:
            from experiments.mom_bpe.corpus_sketch import GramSketch

            implementation = Path(__file__).with_name("corpus_sketch.py").read_bytes().replace(b"\r\n", b"\n")
            if sha(implementation) != SKETCH_SHA256:
                raise ValueError("The bounded candidate implementation differs from its recorded identity")
            self.sketch = GramSketch(self.db, THRESHOLD)

    def root(self, n):
        path = []
        while True:
            parent, = self.db.execute("SELECT parent FROM documents WHERE n=?", (n,)).fetchone()
            if parent is None:
                break
            path.append(n)
            n = parent
        self.db.executemany("UPDATE documents SET parent=? WHERE n=?", [(n, i) for i in path])
        return n

    def join(self, n, other, kind):
        self.db.execute("INSERT OR IGNORE INTO links VALUES(?,?,?)", (n, other, kind))
        first, second = sorted((self.root(n), self.root(other)))
        if first != second:
            self.db.execute("UPDATE documents SET parent=? WHERE n=?", (first, second))

    def insert(self, record, text):
        view = duplicate_view(text)
        grams = signatures(view)
        values = {"group_key": record["group"], "native_group": record.get("native_group"), "raw_hash": record["sha256"],
                  "canonical_hash": record["canonical_sha256"], "view_hash": sha(view.encode())}
        n = self.db.execute(
            "INSERT INTO documents(identity,source,group_key,native_group,raw_hash,canonical_hash,view_hash,gram_count,text,body)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (record["id"], record["src"], *values.values(), len(grams), text,
             json.dumps(record, ensure_ascii=False, sort_keys=True)),
        ).lastrowid
        for field, kind in (("group_key", "source_group"), ("native_group", "native_ancestry"), ("raw_hash", "raw_exact"),
                            ("canonical_hash", "canonical_exact"), ("view_hash", "screening_exact")):
            if values[field] is None:
                continue
            for other, in self.db.execute(
                f"SELECT min(n) FROM documents WHERE {field}=? AND n<? GROUP BY source", (values[field], n),
            ):
                self.join(n, other, kind)
        if not grams:
            return
        if self.sketch is not None:
            histogram, candidates = self.sketch.candidates(grams)
            # The shared bucket counts bound exact overlap, so every surviving
            # pair is still checked against its complete five-gram set.
            for other in candidates:
                size, old, view_hash = self.db.execute(
                    "SELECT gram_count,text,view_hash FROM documents WHERE n=?", (other,)
                ).fetchone()
                if view_hash != values["view_hash"]:
                    self.compare(n, other, size, old, grams)
            self.sketch.persist(n, len(grams), histogram)
            return
        terms = prefix(grams, self.feature_order)
        self.db.execute("DELETE FROM candidates")
        for start in range(0, len(terms), 180):
            chunk = terms[start:start + 180]
            self.db.execute(
                f"INSERT OR IGNORE INTO candidates SELECT n FROM postings WHERE term IN ({','.join('?' for _ in chunk)})",
                chunk,
            )
        minimum = (THRESHOLD * len(grams) + 99) // 100
        maximum = 100 * len(grams) // THRESHOLD
        for other, size, old in self.db.execute(
            "SELECT d.n,d.gram_count,d.text FROM candidates c JOIN documents d USING(n)"
            " WHERE d.view_hash<>? AND d.gram_count BETWEEN ? AND ?",
            (values["view_hash"], minimum, maximum),
        ):
            self.compare(n, other, size, old, grams)
        self.db.executemany("INSERT INTO postings VALUES(?,?)", ((term, n) for term in terms))

    def compare(self, n, other, size, old, grams):
        comparison = signatures(duplicate_view(old))
        shared = len(comparison & grams)
        if 100 * shared >= THRESHOLD * (len(grams) + size - shared):
            self.join(n, other, "near_duplicate")

    def counts(self):
        # Exact links record one previous witness per source; they are not all pairs.
        return [
            {"source_a": a, "source_b": b, "kind": kind, "links": n}
            for a, b, kind, n in self.db.execute(
                "SELECT min(a.source,b.source),max(a.source,b.source),l.kind,count(*)"
                " FROM links l JOIN documents a ON a.n=l.n JOIN documents b ON b.n=l.other"
                " GROUP BY 1,2,3 ORDER BY 1,2,3"
            )
        ]

    def members(self):
        for n, raw, body in self.db.execute("SELECT n,text,body FROM documents ORDER BY n"):
            root = self.root(n)
            name, = self.db.execute("SELECT identity FROM documents WHERE n=?", (root,)).fetchone()
            yield name, raw, json.loads(body)

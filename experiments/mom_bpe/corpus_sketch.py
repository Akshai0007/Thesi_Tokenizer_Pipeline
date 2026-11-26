"""An exact candidate filter for the local Wikipedia capacity index."""

from zlib import crc32

import numpy as np


class GramSketch:
    """Bound intersection from above by summing shared counts in hash buckets."""

    buckets = 512

    def __init__(self, db, threshold):
        self.db = db
        self.threshold = threshold
        self.length = 0
        self.rows = np.empty(2048, dtype=np.int64)
        self.sizes = np.empty(2048, dtype=np.int64)
        self.histograms = np.empty((2048, self.buckets), dtype="<u4")
        db.execute("CREATE TABLE IF NOT EXISTS gram_sketch(n INTEGER PRIMARY KEY, size INTEGER, histogram BLOB)")
        for n, size, blob in db.execute("SELECT n,size,histogram FROM gram_sketch ORDER BY n"):
            self.add(n, size, np.frombuffer(blob, dtype="<u4"))

    def add(self, n, size, histogram):
        if self.length == len(self.rows):
            def doubled(values):
                resized = np.empty((len(values) * 2, *values.shape[1:]), dtype=values.dtype)
                resized[:self.length] = values
                return resized

            self.rows = doubled(self.rows)
            self.sizes = doubled(self.sizes)
            self.histograms = doubled(self.histograms)
        self.rows[self.length] = n
        self.sizes[self.length] = size
        self.histograms[self.length] = histogram
        self.length += 1

    def candidates(self, grams):
        hashed = np.fromiter((crc32(gram.encode()) % self.buckets for gram in grams),
                             dtype=np.int16, count=len(grams))
        histogram = np.bincount(hashed, minlength=self.buckets).astype("<u4")
        sizes = self.sizes[:self.length]
        positions = np.flatnonzero((self.threshold * len(grams) <= 100 * sizes)
                                   & (self.threshold * sizes <= 100 * len(grams)))
        if not len(positions):
            return histogram, ()
        # Each hash collision can only increase the possible overlap.
        overlap_bound = np.minimum(self.histograms[positions], histogram).sum(axis=1, dtype=np.int64)
        survivors = positions[(100 + self.threshold) * overlap_bound >= self.threshold * (sizes[positions] + len(grams))]
        return histogram, self.rows[survivors].tolist()

    def persist(self, n, size, histogram):
        self.db.execute("INSERT INTO gram_sketch VALUES(?,?,?)", (n, size, histogram.tobytes()))
        self.add(n, size, histogram)

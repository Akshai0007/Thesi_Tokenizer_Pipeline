"""Stage a bounded, revision-pinned Malayalam Wikipedia diagnostic capture."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from experiments.mom_bpe.corpus_acquire import RangeFile, public_session
from experiments.mom_bpe.corpus_build import write_rows_once
from experiments.mom_bpe.diagnose import format_json, immutable, sha
from mltok.snapshots import _lock, file_identity

SOURCE = {
    "source": "mlwiki_snapshot",
    "repo": "wikimedia/wikipedia",
    "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "path": "20231101.ml/train-00000-of-00001.parquet",
    "parquet_size": 183071274,
    "parquet_lfs_sha256": "f2a0f5e7921f5cd56137cfce8f6da05bb885875f83e315b64d6971bb7810041b",
    "license_card": "CC-BY-SA-3.0 AND GFDL",
    "license_evidence": "https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md",
    "terms": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content",
}
TRANSFER_CAP = 75_000_000
GROUP_CAP = 48_000_000
ARTICLE_CAP = 256_000
COLUMNS = ("id", "url", "text")


def article_rows(reader, group):
    for batch in reader.iter_batches(batch_size=128, row_groups=[group], columns=list(COLUMNS), use_threads=False):
        for item in batch.to_pylist():
            text, identifier, url = item["text"], item["id"], item["url"]
            if not isinstance(text, str) or not 0 < len(text.encode()) <= ARTICLE_CAP:
                continue
            parsed = urlsplit(url)
            if (not str(identifier).isascii() or not str(identifier).isdecimal()
                    or parsed.netloc != "ml.wikipedia.org" or parsed.scheme not in {"http", "https"}
                    or not parsed.path.startswith("/wiki/") or parsed.query or parsed.fragment):
                raise ValueError("Malayalam Wikipedia article attribution differs from the pinned schema")
            yield {"src": SOURCE["source"], "row_index": f"{group}:{identifier}", "text": text,
                   "attributes": {"id": str(identifier), "url": url,
                                  "history": f"https://ml.wikipedia.org/w/index.php?curid={identifier}&action=history"}}


def acquire(out, group_count, transfer_cap):
    import pyarrow.parquet as parquet
    import requests

    if not 1 <= group_count <= 86 or not 1 <= transfer_cap <= TRANSFER_CAP:
        raise ValueError("Expected 1–86 groups and a local transfer cap of at most 75,000,000 bytes")
    ledger = {"bytes": 0, "ranges": []}
    url = f"https://huggingface.co/datasets/{SOURCE['repo']}/resolve/{SOURCE['revision']}/{SOURCE['path']}"
    recipe = {**SOURCE, "groups_requested": group_count, "transfer_cap": transfer_cap,
              "code_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))}
    with _lock(out / ".acquire.lock"):
        immutable(out / "recipe.json", format_json(recipe))
        captures = out / "captures"
        captures.mkdir(exist_ok=True)
        with public_session("Malayalam tokenizer thesis bounded corpus capture") as session:
            try:
                reader = parquet.ParquetFile(RangeFile(session, url, SOURCE["parquet_size"], ledger,
                                                       byte_cap=transfer_cap))
            except requests.RequestException:
                raise OSError("Pinned Wikimedia Parquet metadata request failed") from None
            if reader.num_row_groups != 86 or not set(COLUMNS).issubset(reader.schema_arrow.names):
                raise ValueError("Pinned Wikimedia Parquet metadata differs")
            indices = sorted(range(reader.num_row_groups),
                             key=lambda index: sha(f"mlwiki-capacity-v1:{index}".encode()))[:group_count]
            completed = []
            for group in indices:
                stem = f"wiki-{group:03d}"
                receipt_path, payload = captures / f"{stem}.json", captures / f"{stem}.jsonl"
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_bytes())
                    if (any(receipt.get(k) != v for k, v in SOURCE.items())
                            or receipt["shard"] != stem or receipt["file"] != payload.name
                            or file_identity(payload) != (receipt["bytes"], receipt["sha256"])):
                        raise ValueError("Saved Wikimedia capture does not match its receipt")
                    ledger["reserved"] += receipt["range_bytes"]
                else:
                    meta = reader.metadata.row_group(group)
                    if meta.total_byte_size > GROUP_CAP:
                        raise ValueError("Wikimedia row group exceeds the decoded memory bound")
                    expected = sum(meta.column(i).total_compressed_size for i in range(meta.num_columns)
                                   if meta.column(i).path_in_schema in COLUMNS)
                    if ledger["reserved"] + expected > transfer_cap:
                        break
                    start = len(ledger["ranges"])
                    try:
                        result = write_rows_once(payload, article_rows(reader, group))
                    except requests.RequestException:
                        raise OSError("Pinned Wikimedia Parquet payload request failed") from None
                    ranges = ledger["ranges"][start:]
                    receipt = {**SOURCE, "shard": stem, "file": payload.name, "rows": result["rows"],
                               "bytes": result["size"], "sha256": result["sha256"], "ranges": ranges,
                               "range_bytes": sum(r["end"] - r["start"] + 1 for r in ranges)}
                    immutable(receipt_path, format_json(receipt))
                completed.append(receipt_path.name)
            report = {"schema": "mom-wikipedia-capacity-capture-v1", "groups": completed,
                      "groups_requested": group_count, "payload_transfer_bytes": ledger["reserved"],
                      "transfer_cap": transfer_cap, "recipe_sha256": file_identity(out / "recipe.json")[1]}
            immutable(out / "capture-complete.json", format_json(report))
            return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--transfer-cap", type=int, default=TRANSFER_CAP)
    args = parser.parse_args()
    print(json.dumps(acquire(args.out, args.groups, args.transfer_cap), sort_keys=True))

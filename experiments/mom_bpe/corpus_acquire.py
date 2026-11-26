"""Bounded, credential-free sampling of pinned public corpus storage."""

import argparse
import io
import json
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe.diagnose import format_json, immutable, sha
from mltok.lib.bpe_progress import canonical

CAP = 16 * 1024**2
SELECTION = {
    "sangraha": ("verified", "mal", "text"),
    "fineweb2": ("mal_Mlym", "train", "text"),
    "smc": ("default", "train", "sentence"),
    "indiccorp": ("indiccorp_v2", "mal_Mlym", "text"),
}
HPLT_MAP = "https://data.hplt-project.org/three/sorted/mal_Mlym.map"


def public_session(agent):
    import requests

    client = requests.Session()
    # Avoid ambient credentials, including .netrc lookups after redirects.
    client.trust_env = False
    client.headers["User-Agent"] = agent
    return client


class RangeFile(io.RawIOBase):
    def __init__(self, session, url, size, ledger, *, byte_cap=None):
        self.session, self.url, self.size, self.ledger = session, url, size, ledger
        self.byte_cap = byte_cap
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        if base + offset < 0:
            raise OSError("Invalid Parquet seek")
        self.pos = base + offset
        return self.pos

    def read(self, size=-1):
        count = min(self.size - self.pos, size if size >= 0 else self.size)
        if count <= 0:
            return b""
        start, end = self.pos, self.pos + count - 1
        reserved = self.ledger.get("reserved", self.ledger["bytes"]) + count
        if reserved > (CAP if self.byte_cap is None else self.byte_cap):
            raise OSError("Corpus sample transfer budget exhausted")
        # A distinct query per range also prevents intermediary range-cache mixing.
        url = self.url + f"?sample_range={start}-{end}"
        with self.session.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                              timeout=45, stream=True) as response:
            if (response.status_code != 206
                    or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"):
                raise OSError(f"Unexpected response to a pinned Parquet range: HTTP {response.status_code}")
            if response.headers.get("Content-Length", str(count)) != str(count):
                raise OSError("Pinned Parquet range has an inconsistent Content-Length")
            # A read failure may consume some bytes before raising.
            self.ledger["reserved"] = reserved
            data = response.raw.read(count)
            self.ledger["bytes"] += len(data)
        if len(data) != count:
            raise OSError("Truncated Parquet range")
        self.ledger["ranges"].append({"start": start, "end": end, "sha256": sha(data)})
        self.pos += count
        return data


def persist(out, name, records, metadata):
    data = b"".join(canonical(row) + b"\n" for row in records)
    metadata.update(rows=len(records), sha256=sha(data), bytes=len(data))
    out.mkdir(parents=True, exist_ok=True)
    immutable(out / f"{name}.jsonl", data)
    immutable(out / f"{name}.json", format_json(metadata))
    return metadata


def parquet_sample(session, source, probe, viewer, out):
    import pyarrow.parquet as parquet

    config, split, text_field = SELECTION[source]
    hub = probe["hub"]
    expected = viewer["viewer_conversion_revision"]
    if not expected:
        raise ValueError("A pinned Parquet revision is required")
    choices = [row for row in probe["parquet"]["parquet_files"]
               if (row["config"], row["split"]) == (config, split)]
    sizes = viewer["size_metadata"]
    records, files, attempts = [], [], []
    ledger = {"bytes": 0, "ranges": []}
    for item in (choices if len(choices) < 3 else [choices[0], choices[len(choices) // 2], choices[-1]]):
        original = unquote(item["url"])
        path = original.split("/resolve/refs/convert/parquet/")[1]
        url = f"https://huggingface.co/datasets/{hub['id']}/resolve/{expected}/{quote(path)}"
        first_range = len(ledger["ranges"])
        file_info = {"path": path, "size": item["size"], "row_groups_used": []}
        try:
            stream = RangeFile(session, url, item["size"], ledger)
            reader = parquet.ParquetFile(stream)
            if text_field not in reader.schema_arrow.names:
                raise OSError("Pinned Parquet no longer has the expected text field")
            desired = {"id", "doc_id", "url", "sentence_index", "language", "language_score",
                       "minhash_cluster_size", "domain", "source", "file_path", text_field}
            columns = [n for n in reader.schema_arrow.names if n in desired]
            count = reader.num_row_groups
            sampled = sorted(set([0, count // 3, 2 * count // 3, count - 1]))
            # Small groups first leave room for sampling more than one row group.
            def group_bytes(group, attribute):
                row = reader.metadata.row_group(group)
                return sum(getattr(row.column(i), attribute) for i in range(row.num_columns)
                           if row.column(i).path_in_schema in columns)

            for group in sorted(sampled, key=lambda i: (group_bytes(i, "total_compressed_size"), i)):
                if (group_bytes(group, "total_compressed_size") + ledger["bytes"] > CAP
                        or group_bytes(group, "total_uncompressed_size") > 48 * 1024**2):
                    continue
                table = reader.read_row_group(group, columns=columns, use_threads=False)
                if table.num_rows == 0:
                    continue
                indices = sorted({(k * table.num_rows) // 16 for k in range(16)})
                for index in indices:
                    row = {col: table[col][index].as_py() for col in columns}
                    text = row.pop(text_field)
                    if isinstance(text, str) and text and len(text.encode()) <= 300000:
                        records.append({"src": source, "row_index": f"{path}:{group}:{index}",
                                        "text": text, "attributes": row})
                file_info["row_groups_used"].append(group)
        except (OSError, ValueError) as exc:
            attempts.append({"path": path, "failure": str(exc)})
        file_info["ranges"] = ledger["ranges"][first_range:]
        files.append(file_info)
    metadata = {"source": source, "repo": hub["id"], "revision": hub["sha"], "storage_revision": expected,
                "config": config, "split": split, "field": text_field, "size_metadata": sizes,
                "mode": "bounded pointer-pinned Parquet ranges; disjoint positions within decoded row groups",
                "downloaded_bytes": ledger["bytes"], "files": files, "failed_reads": attempts}
    return persist(out, source, records, metadata)


def hplt_sample(session, out):
    import pyarrow as pa

    with session.get(HPLT_MAP, timeout=25, stream=True) as response:
        response.raise_for_status()
        mapping = response.raw.read(32769)
    urls = [x for x in mapping.decode().splitlines() if x.strip()]
    if (not 1 <= len(urls) <= 8 or len(mapping) > 32768
            or any(urlsplit(u).netloc != "data.hplt-project.org" or urlsplit(u).scheme != "https" for u in urls)):
        raise ValueError("Unexpected HPLT shard map")
    records, pages = [], []
    for shard, url in enumerate(urls):
        with session.get(url, headers={"Range": "bytes=0-1048575", "Accept-Encoding": "identity"},
                         timeout=35, stream=True) as response:
            page = {"shard": url, "status": response.status_code, "etag": response.headers.get("ETag"),
                    "content_range": response.headers.get("Content-Range"),
                    "last_modified": response.headers.get("Last-Modified")}
            pages.append(page)
            if (response.status_code not in (200, 206)
                    or (response.status_code == 206 and not (page["content_range"] or "").startswith("bytes 0-"))):
                continue
            data = response.raw.read(1048576)
        page.update(payload_bytes=len(data), sha256=sha(data))
        stream = pa.CompressedInputStream(pa.BufferReader(data), "zstd")
        buffer = b""
        index = 0
        try:
            while index < 14:
                chunk = stream.read(65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer and index < 14:
                    line, buffer = buffer.split(b"\n", 1)
                    row = json.loads(line)
                    text = row.pop("text", None)
                    if isinstance(text, str) and text and len(text.encode()) <= 300000:
                        records.append({"src": "hplt", "row_index": f"{shard}:{index}", "text": text,
                                        "attributes": row})
                    index += 1
                if len(buffer) > 4 * 1024**2:
                    break
        except OSError:
            page["decoder_end"] = "bounded prefix ended"
        page["decoded_records"] = index
    manifest = {"source": "hplt", "repo": "HPLT/HPLT3.0", "revision": "release-3.0:" + sha(mapping),
                "config": "mal_Mlym", "split": "sorted", "field": "text", "map_url": HPLT_MAP, "pages": pages,
                "mode": "up to 14 initial rows per quality-sorted shard; biased to each shard prefix",
                "downloaded_bytes": sum(p.get("payload_bytes", 0) for p in pages)}
    return persist(out, "hplt", records, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--viewer", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", required=True, choices=[*SELECTION, "hplt"])
    args = parser.parse_args()
    with public_session("MOM-BPE thesis small public corpus sample") as client:
        if args.source == "hplt":
            result = hplt_sample(client, args.out)
        else:
            if args.evidence is not None:
                probe = json.loads(args.evidence.read_bytes())["parquet_replay"][args.source]
                viewer = probe["viewer"]
            elif args.probe is not None and args.viewer is not None:
                probe = json.loads((args.probe / f"{args.source}.json").read_bytes())
                viewer = json.loads((args.viewer / f"{args.source}.json").read_bytes())
            else:
                parser.error("Provide committed evidence or the pinned probe and viewer metadata")
            result = parquet_sample(client, args.source, probe, viewer, args.out)
    print(json.dumps({k: result[k] for k in ("source", "rows", "downloaded_bytes")}))

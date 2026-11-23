"""Sample pinned Sangraha columns one data page at a time."""

import argparse
import json
import struct
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mom_bpe.corpus_acquire import RangeFile, persist, public_session
from experiments.mom_bpe.diagnose import sha

REPO = "ai4bharat/sangraha"
REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"


def first_page(schema, stream, footer, field):
    """Create a small single-column Parquet envelope from original page bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from thriftpy2.protocol.compact import TCompactProtocol
    from thriftpy2.transport.memory import TMemoryBuffer

    if len(footer.row_groups) != 1:
        raise ValueError("Unexpected Sangraha row group layout")
    row = footer.row_groups[0]
    column, = [c for c in row.columns if c.meta_data.path_in_schema == [field]]
    metadata = column.meta_data
    start = metadata.dictionary_page_offset if metadata.dictionary_page_offset is not None else metadata.data_page_offset
    stream.seek(start)
    block = stream.read(min(metadata.total_compressed_size, 2 * 1024**2))
    cursor = 0
    decompressed = 0
    values = 0
    rel_dictionary, rel_data = None, None
    while cursor < len(block):
        transport = TMemoryBuffer(block[cursor:])
        header = schema.PageHeader()
        header.read(TCompactProtocol(transport))
        length = transport._pos + header.compressed_page_size
        if length <= 0 or cursor + length > len(block):
            raise ValueError("A complete Sangraha data page did not fit the range")
        decompressed += transport._pos + header.uncompressed_page_size
        if decompressed > 48 * 1024**2:
            raise ValueError("Uncompressed Sangraha page exceeds the local memory limit")
        if header.type == schema.PageType.DICTIONARY_PAGE:
            rel_dictionary = cursor + 4
        elif header.type == schema.PageType.DATA_PAGE:
            rel_data = cursor + 4
            values = header.data_page_header.num_values
        elif header.type == schema.PageType.DATA_PAGE_V2:
            rel_data = cursor + 4
            values = header.data_page_header_v2.num_rows
        else:
            raise ValueError("Unexpected Sangraha page type")
        cursor += length
        if rel_data is not None:
            break
    if values <= 0:
        raise ValueError("No complete Sangraha data page")
    # Footers are reparsed for each projection. Retained pages are byte-identical.
    root = footer.schema[0]
    root.num_children = 1
    footer.schema = [root, next(x for x in footer.schema[1:] if x.name == field)]
    footer.num_rows = row.num_rows = metadata.num_values = values
    footer.key_value_metadata = None
    footer.column_orders = None
    row.columns = [column]
    row.sorting_columns = None
    row.file_offset = 4
    row.total_byte_size = metadata.total_uncompressed_size = decompressed
    row.total_compressed_size = metadata.total_compressed_size = cursor
    column.file_offset = 0
    for key in ("offset_index_offset", "offset_index_length", "column_index_offset", "column_index_length"):
        setattr(column, key, None)
    for key in ("index_page_offset", "bloom_filter_offset", "bloom_filter_length", "statistics", "encoding_stats"):
        setattr(metadata, key, None)
    metadata.dictionary_page_offset = rel_dictionary
    metadata.data_page_offset = rel_data
    encoded = TMemoryBuffer()
    footer.write(TCompactProtocol(encoded))
    b = encoded.getvalue()
    return pq.ParquetFile(pa.BufferReader(b"PAR1" + block[:cursor] + b + struct.pack("<I", len(b)) + b"PAR1")).read()[field]


def acquire(session, schema_path, out):
    import thriftpy2
    from thriftpy2 import _compat

    # The pinned local source distribution uses the pure Python codec.
    _compat.CYTHON = False
    from thriftpy2.protocol.compact import TCompactProtocol
    from thriftpy2.transport.memory import TMemoryBuffer

    binary = schema_path.read_bytes()
    resources = json.loads((schema_path.parent / "resources.json").read_bytes())
    if sha(binary) != next(x["sha256"] for x in resources if x["name"] == "apache-parquet-format"):
        raise ValueError("Apache Parquet schema digest differs")
    schema = thriftpy2.load(str(schema_path), module_name="bounded_parquet_thrift")
    response = session.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/verified/mal", timeout=30)
    response.raise_for_status()
    inventory = sorted((f for f in response.json() if f["type"] == "file"), key=lambda f: f["path"])
    if len(inventory) < 3:
        raise ValueError("Sangraha inventory incomplete")
    ledger = {"bytes": 0, "ranges": []}
    rows, files = [], []
    for item in [inventory[0], inventory[len(inventory) // 2], inventory[-1]]:
        path = item["path"]
        mark = len(ledger["ranges"])
        stream = RangeFile(session, f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(path)}",
                           item["size"], ledger)
        stream.seek(-8, 2)
        tail = stream.read(8)
        length = struct.unpack("<I", tail[:4])[0]
        if tail[4:] != b"PAR1" or length > 256 * 1024:
            raise ValueError("Unrecognized Sangraha Parquet footer")
        stream.seek(-length - 8, 2)
        raw = stream.read(length)
        columns = {}
        for field in ("text", "doc_id", "type"):
            footer = schema.FileMetaData()
            footer.read(TCompactProtocol(TMemoryBuffer(raw)))
            columns[field] = first_page(schema, stream, footer, field)
        size = min(map(len, columns.values()))
        for i in sorted({size * k // 24 for k in range(24)}):
            if size == 0:
                break
            text = columns["text"][i].as_py()
            if isinstance(text, str) and text and len(text.encode()) <= 300000:
                rows.append({"src": "sangraha", "row_index": f"{path}:{i}", "text": text,
                             "attributes": {"doc_id": columns["doc_id"][i].as_py(), "source": columns["type"][i].as_py()}})
        files.append({"path": path, "size": item["size"], "ranges": ledger["ranges"][mark:], "aligned_initial_rows": size})
    return persist(out, "sangraha", rows, {
        "source": "sangraha", "repo": REPO, "revision": REVISION, "storage_revision": REVISION,
        "config": "verified", "split": "mal", "field": "text", "downloaded_bytes": ledger["bytes"],
        "mode": "first independently decodable data page aligned across three fields on each of three pinned shards",
        "files": files, "subset_file_count": len(inventory), "subset_parquet_bytes": sum(f["size"] for f in inventory),
        "parsing_resources": resources,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with public_session("MOM-BPE thesis bounded public Parquet page sample") as client:
        result = acquire(client, args.schema, args.out)
    print(json.dumps({k: result[k] for k in ("rows", "downloaded_bytes")}))

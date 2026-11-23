"""Reference BPB accounting for frozen documents and explicit per-token NLLs."""

import hashlib
import math


def canonical_bpb(backend, rows, nll):
    """Call nll(inputs, targets) once per document. Return one nats value per target."""
    bos, eos = (backend.special_ids[token] for token in ("<s>", "</s>"))
    if bos == eos:
        raise ValueError("BOS and EOS must have distinct IDs")
    control_ids = set(backend.special_ids.values())
    total_bytes = targets_scored = 0
    document_losses = []
    document_ids = set()
    for row in rows:
        identifier = row["id"]
        if identifier in document_ids:
            raise ValueError("Duplicate document in BPB evaluation")
        document_ids.add(identifier)
        enc = backend.encode_offsets(row["text"])
        raw = enc.text.encode("utf-8")
        if len(raw) != row["canonical_bytes"] or hashlib.sha256(raw).hexdigest() != row["canonical_sha256"]:
            raise ValueError("BPB canonical bytes differ from the frozen evaluation record")
        if (backend.decode(enc.ids) != enc.text or control_ids.intersection(enc.ids)
                or any(type(x) is not int or not 0 <= x < backend.vocab_size for x in enc.ids)):
            raise ValueError("BPB requires lossless encoding with valid, non-control text IDs")
        targets = [*enc.ids, eos]
        values = list(nll([bos, *enc.ids], targets))
        if len(values) != len(targets) or any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("BPB requires one finite nonnegative NLL in nats per target")
        document_losses.append(math.fsum(values))
        total_bytes += len(raw)
        targets_scored += len(targets)
    if total_bytes == 0:
        raise ValueError("BPB requires positive canonical source bytes")
    total_nll = math.fsum(document_losses)
    return {"bpb": total_nll / math.log(2) / total_bytes, "canonical_bytes": total_bytes,
            "predicted_tokens": targets_scored, "documents": len(document_ids), "nll_nats": total_nll}

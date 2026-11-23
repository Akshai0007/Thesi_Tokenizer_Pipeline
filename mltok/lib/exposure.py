"""Plan comparable predicted-token and canonical-byte exposure without training."""


def account_exposure(backend, rows, predicted_token_budget=None):
    """Count one predicted EOS per full document and count BOS as context only."""
    if predicted_token_budget is not None and (type(predicted_token_budget) is not int or predicted_token_budget < 1):
        raise ValueError("Expected a positive predicted token budget")
    totals = {"full_documents": 0, "full_canonical_bytes": 0, "full_raw_bytes": 0,
              "predicted_tokens": 0, "predicted_eos": 0, "partial": None}
    for row in rows:
        if totals["predicted_tokens"] == predicted_token_budget:
            break
        enc = backend.encode_offsets(row["text"])
        count = len(enc.ids) + 1
        remaining = None if predicted_token_budget is None else predicted_token_budget - totals["predicted_tokens"]
        if remaining is not None and remaining < count:
            pos = enc.offsets[remaining - 1][1]
            ambiguous = remaining < len(enc.offsets) and enc.offsets[remaining][0] != pos
            totals["partial"] = {
                "document_id": row["id"], "predicted_text_tokens": remaining,
                "canonical_bytes": None if ambiguous else len(enc.text[:pos].encode("utf-8")),
            }
            totals["predicted_tokens"] += remaining
            break
        totals["predicted_tokens"] += count
        totals["full_documents"] += 1
        totals["predicted_eos"] += 1
        totals["full_canonical_bytes"] += len(enc.text.encode("utf-8"))
        totals["full_raw_bytes"] += len(row["text"].encode("utf-8"))
    if predicted_token_budget is not None and totals["predicted_tokens"] != predicted_token_budget:
        raise ValueError("The input stream cannot fill the requested token budget")
    return totals

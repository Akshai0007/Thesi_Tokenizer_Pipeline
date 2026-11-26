"""Reference BPB accounting for frozen documents and explicit per-token NLLs."""

import hashlib
import math


def summarize_bpb(document_losses, canonical_bytes, predicted_tokens):
    """Reduce one summed NLL in nats per document with a shared byte denominator."""
    losses = list(document_losses)
    if canonical_bytes <= 0 or any(not math.isfinite(nll) or nll < 0 for nll in losses):
        raise ValueError("BPB requires finite nonnegative losses and positive canonical source bytes")
    total_nll = math.fsum(losses)
    return {"bpb": total_nll / math.log(2) / canonical_bytes, "canonical_bytes": canonical_bytes,
            "predicted_tokens": predicted_tokens, "documents": len(losses), "nll_nats": total_nll}


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
    return summarize_bpb(document_losses, total_bytes, targets_scored)


class TorchWindowNLL:
    """Reference scorer with right padding and document-local sliding windows.

    A model supplies logits for each position and accepts a binary attention
    mask. Only the final real position in each prefix contributes a loss.
    Positions are local to the truncated window.
    """

    def __init__(self, model, *, context, pad_id, batch_size=32, device="cpu"):
        if type(context) is not int or type(batch_size) is not int or min(context, batch_size) < 1:
            raise ValueError("Context and batch size must be positive integers")
        self.model, self.context = model, context
        self.pad_id, self.batch_size, self.device = pad_id, batch_size, device

    def __call__(self, inputs, targets):
        import torch
        import torch.nn.functional as functional

        if self.model.training or len(inputs) != len(targets):
            raise ValueError("Reference scoring requires evaluation mode and aligned inputs")
        scores = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self.batch_size):
                end = min(start + self.batch_size, len(inputs))
                windows = [inputs[max(0, i + 1 - self.context):i + 1] for i in range(start, end)]
                lengths = torch.tensor([len(w) for w in windows], device=self.device)
                width = max(map(len, windows))
                ids = torch.tensor([w + [self.pad_id] * (width - len(w)) for w in windows], device=self.device)
                mask = torch.arange(width, device=self.device) < lengths[:, None]
                output = self.model(input_ids=ids, attention_mask=mask.long())
                logits = output if isinstance(output, torch.Tensor) else output.logits
                chosen = logits[torch.arange(end - start, device=self.device), lengths - 1].float()
                labels = torch.tensor(targets[start:end], device=self.device)
                scores.extend(functional.cross_entropy(chosen, labels, reduction="none").tolist())
        return scores

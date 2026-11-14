"""Concrete `Backend` implementations (HF tokenizers, SentencePiece, Hub baselines)."""
from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

from ..contracts import Backend
from .bpe_core import DEFAULT_SPECIALS

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def save_hf_tokenizer(tok: Tokenizer, out_dir, specials=None) -> None:
    specials = specials or DEFAULT_SPECIALS
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>",
        "clean_up_tokenization_spaces": False,
        "model_max_length": 1000000000,
        "chat_template": CHAT_TEMPLATE,
        "additional_special_tokens": [s for s in specials if s not in ("<s>", "</s>", "<unk>", "<pad>")],
    }
    (out / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}, indent=2)
    )


class HFBackend(Backend):
    def __init__(self, tok: Tokenizer, normalizer, name: str = "hf", specials=None):
        self.tok, self.normalizer, self.name = tok, normalizer, name
        self.specials = specials or DEFAULT_SPECIALS
        self.vocab_size = tok.get_vocab_size()
        self.unk_id = tok.token_to_id("<unk>")
        self.eos_id = tok.token_to_id("</s>")

    def encode_batch(self, texts):
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def tokens(self, text):
        return self.tok.encode(text, add_special_tokens=False).tokens

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def expected(self, text):
        return self.normalizer.normalize(text)

    def save(self, out_dir):
        save_hf_tokenizer(self.tok, out_dir, self.specials)

    @classmethod
    def load(cls, out_dir, normalizer, name="hf"):
        return cls(Tokenizer.from_file(str(Path(out_dir) / "tokenizer.json")), normalizer, name)


class SPMBackend(Backend):
    def __init__(self, model_path, normalizer, name: str = "spm"):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.path, self.normalizer, self.name = str(model_path), normalizer, name
        self.vocab_size = self.sp.get_piece_size()
        self.unk_id = self.sp.unk_id()
        self.eos_id = self.sp.eos_id()

    def encode_batch(self, texts):
        return self.sp.encode([self.normalizer.normalize(t) for t in texts])

    def tokens(self, text):
        return self.sp.encode(self.normalizer.normalize(text), out_type=str)

    def decode(self, ids):
        return self.sp.decode(ids)

    def expected(self, text):
        return self.normalizer.normalize(text)

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if Path(self.path).resolve() != (out / "spm.model").resolve():
            (out / "spm.model").write_bytes(Path(self.path).read_bytes())


class HubBackend(Backend):
    """Count-only comparison against published tokenizers (needs `transformers`)."""

    def __init__(self, hub_name: str):
        from transformers import AutoTokenizer

        self.tk = AutoTokenizer.from_pretrained(hub_name)
        self.name = hub_name
        self.vocab_size = len(self.tk)
        self.unk_id = getattr(self.tk, "unk_token_id", None)
        self.eos_id = getattr(self.tk, "eos_token_id", None)

    def encode_batch(self, texts):
        return self.tk(texts, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tk.decode(ids)

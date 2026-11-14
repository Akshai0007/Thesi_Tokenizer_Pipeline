"""Hugging Face dataset, streamed (no full download). Params: hf, name, data_dir, data_files, split, text."""
from mltok.contracts import Source
from mltok.registry import register


@register("source", "hf_stream")
class HFStreamSource(Source):
    def __init__(self, id, hf, name=None, data_dir=None, data_files=None, split="train", text="text", **params):
        super().__init__(id=id, **params)
        self.hf, self.split, self.field = hf, split, text
        self.kw = {k: v for k, v in (("name", name), ("data_dir", data_dir), ("data_files", data_files)) if v}

    def iter_docs(self):
        from datasets import load_dataset  # lazy: only needed when this source is used

        ds = load_dataset(self.hf, split=self.split, streaming=True, **self.kw)
        for ex in ds:
            t = ex.get(self.field)
            if t:
                yield {"text": t, "src": self.id}

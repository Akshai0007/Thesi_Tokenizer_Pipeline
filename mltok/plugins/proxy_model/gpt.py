"""Small decoder-only GPT (RMSNorm, SwiGLU, learned positions, tied embeddings).
Params: size = nano|tiny|small|medium, or explicit n_layer / n_head / d_model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mltok.contracts import ProxyModel
from mltok.registry import register

SIZES = {
    "nano": dict(n_layer=2, n_head=2, d_model=64),  # CPU tests only
    "tiny": dict(n_layer=6, n_head=6, d_model=384),  # ~11M non-embedding
    "small": dict(n_layer=12, n_head=12, d_model=768),  # ~85M
    "medium": dict(n_layer=16, n_head=16, d_model=1024),  # ~200M
}


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        hid = int(8 * d / 3 / 64 + 1) * 64
        self.up, self.down = nn.Linear(d, 2 * hid, bias=False), nn.Linear(hid, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C))
        g, u = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class GPT(nn.Module):
    def __init__(self, vocab, seq_len, n_layer, n_head, d_model):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab, d_model), nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.norm = RMSNorm(d_model)
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks:
            x = b(x)
        return F.linear(self.norm(x), self.wte.weight)


@register("proxy_model", "gpt")
class GPTPlugin(ProxyModel):
    def __init__(self, size="tiny", **overrides):
        self.cfg = {**SIZES[size], **overrides}

    def build(self, vocab_size, seq_len):
        return GPT(vocab_size, seq_len, **self.cfg)

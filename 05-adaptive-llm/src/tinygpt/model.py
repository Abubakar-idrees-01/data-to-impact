"""A small GPT (decoder-only transformer), written to be read.

    tokens ─► embedding + position ─► [block] × n_layer ─► norm ─► linear ─► next-token scores
    block:   x = x + attention(norm(x))         tokens look at earlier tokens (causal)
             x = x + feedforward(norm(x))       each token is transformed on its own

Pre-norm (normalise before each sub-layer) trains stably without warm-up tricks; the output layer shares its weights
with the token embedding ("weight tying"), which saves vocab × d parameters and usually helps small models.

The feed-forward sub-layer is built by `make_ffn`, so phase A3 can swap in a Mixture-of-Experts layer without
touching anything else.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    ctx: int = 256              # tokens of context the model can see
    d: int = 256                # width: size of each token's vector
    n_layer: int = 6
    n_head: int = 8
    ff_mult: int = 4            # feed-forward hidden size = ff_mult × d
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        assert c.d % c.n_head == 0
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.out = nn.Linear(c.d, c.d, bias=False)
        self.dropout = c.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        # (B, T, D) -> (B, heads, T, D/heads): each head attends separately
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        # softmax(q·kᵀ / √d_head) · v, with a causal mask so token t only sees tokens ≤ t
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).contiguous().view(B, T, D))


class FeedForward(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.up = nn.Linear(c.d, c.ff_mult * c.d, bias=False)
        self.down = nn.Linear(c.ff_mult * c.d, c.d, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.gelu(self.up(x))))


def make_ffn(c: GPTConfig) -> nn.Module:
    return FeedForward(c)


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.d)
        self.attn = Attention(c)
        self.norm2 = nn.LayerNorm(c.d)
        self.ffn = make_ffn(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab_size, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight                       # weight tying
        self.apply(self._init)
        for name, p in self.named_parameters():                  # GPT-2: scale residual outputs by depth
            if name.endswith(("out.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def n_params(self, embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())            # the tied head is counted once
        return n if embeddings else n - self.tok.weight.numel() - self.pos.weight.numel()

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.config.ctx:
            raise ValueError(f"sequence of {T} tokens is longer than the context ({self.config.ctx})")
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float = 0.8, top_k: int | None = 40,
                 stop: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample up to max_new tokens after idx (shape 1 × T). Stops early at the `stop` token."""
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.config.ctx:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            if stop is not None and nxt.item() == stop:
                break
        return idx

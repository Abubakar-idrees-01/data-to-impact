"""A small GPT (decoder-only transformer), written to be read.

    tokens ─► embedding + position ─► [block] × n_layer ─► norm ─► linear ─► next-token scores
    block:   x = x + attention(norm(x))         tokens look at earlier tokens (causal)
             x = x + feedforward(norm(x))       each token is transformed on its own

Pre-norm (normalise before each sub-layer) trains stably without warm-up tricks; the output layer shares its weights
with the token embedding ("weight tying"), which saves vocab × d parameters and usually helps small models.

The feed-forward sub-layer is built by `make_ffn`: a dense MLP, or (n_experts > 0) a Mixture-of-Experts layer where
a router sends each token to top_k of n_experts smaller MLPs, so only part of the layer runs for each token.
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
    n_experts: int = 0          # 0 = dense feed-forward; > 0 = Mixture-of-Experts
    top_k: int = 1              # experts each token is sent to
    expert_mult: int = 4        # each expert's hidden size = expert_mult × d
    aux_coef: float = 0.01      # weight of the load-balancing loss (added by the training loop)
    exit_weight: float = 0.0    # > 0: every layer gets an exit; loss += exit_weight × mean loss of the early exits

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


class MoE(nn.Module):
    """Mixture-of-Experts feed-forward: a router picks top_k of n_experts MLPs per token.

    router:   p = softmax(x · W_r)            one probability per expert, per token
    choose:   the top_k experts by p; with top_k ≥ 2 their weights are renormalised to sum to 1 (Mixtral);
              with top_k = 1 the weight is the raw p (Switch Transformer). Renormalising one weight gives p/p = 1,
              a constant, and the router would get no gradient from the language-model loss at all.
    output:   Σ weight_e · expert_e(x)        only the chosen experts run for this token

    Load balancing (Switch Transformer): with f_e = share of tokens whose first choice is expert e and P_e = mean
    router probability of e, aux = n_experts · Σ f_e · P_e. It is 1.0 when routing is perfectly even and grows when a
    few experts take most tokens; the training loop adds aux_coef × aux to the loss.
    """

    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_experts, self.top_k = c.n_experts, c.top_k
        self.router = nn.Linear(c.d, c.n_experts, bias=False)
        self.experts = nn.ModuleList(
            FeedForward(GPTConfig(d=c.d, ff_mult=c.expert_mult, dropout=c.dropout)) for _ in range(c.n_experts))
        self.aux_loss = torch.zeros(())
        self.last_counts = torch.zeros(c.n_experts)        # tokens per expert in the last forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        probs = F.softmax(self.router(flat), dim=-1)                    # (N, E)
        weights, chosen = torch.topk(probs, self.top_k, dim=-1)         # (N, k)
        if self.top_k > 1:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = torch.where(chosen == e)                       # tokens that picked expert e
            if rows.numel():
                out.index_add_(0, rows, expert(flat[rows]) * weights[rows, slot, None])
        first = F.one_hot(chosen[:, 0], self.n_experts).float()
        self.last_counts = first.sum(0).detach()
        self.aux_loss = self.n_experts * (first.mean(0) * probs.mean(0)).sum()
        return out.view(B, T, D)


def make_ffn(c: GPTConfig) -> nn.Module:
    return MoE(c) if c.n_experts else FeedForward(c)


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
        # early exits: one norm per layer except the last (which uses self.norm), all sharing the output layer
        self.exit_norms = nn.ModuleList(nn.LayerNorm(c.d) for _ in range(c.n_layer - 1)) if c.exit_weight else None
        self.exit_loss = torch.zeros(())
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

    def n_active_params(self) -> int:
        """Parameters used for one token: an MoE layer counts its router and only top_k of its experts."""
        n = self.n_params()
        for b in self.blocks:
            if isinstance(b.ffn, MoE):
                per_expert = sum(p.numel() for p in b.ffn.experts[0].parameters())
                n -= (b.ffn.n_experts - b.ffn.top_k) * per_expert
        return n

    def aux_loss(self) -> torch.Tensor:
        """Sum of the MoE load-balancing losses from the last forward pass (0 for a dense model)."""
        return sum((b.ffn.aux_loss for b in self.blocks if isinstance(b.ffn, MoE)), torch.zeros(()))

    def n_params(self, embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())            # the tied head is counted once
        return n if embeddings else n - self.tok.weight.numel() - self.pos.weight.numel()

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.config.ctx:
            raise ValueError(f"sequence of {T} tokens is longer than the context ({self.config.ctx})")
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        early = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.exit_norms is not None and i < len(self.exit_norms) and targets is not None:
                early.append(self.head(self.exit_norms[i](x)))
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if early:
                self.exit_loss = torch.stack([F.cross_entropy(e.reshape(-1, e.size(-1)), targets.reshape(-1))
                                              for e in early]).mean()
        return logits, loss

    @torch.no_grad()
    def exit_logits(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """Next-token scores after every layer. With trained exits, each layer's own norm; without them, the final
        norm applied to the intermediate state (the "logit lens": what an untrained early exit would say)."""
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        out = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            last = i == len(self.blocks) - 1
            norm = self.norm if last or self.exit_norms is None else self.exit_norms[i]
            out.append(self.head(norm(x)))
        return out

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

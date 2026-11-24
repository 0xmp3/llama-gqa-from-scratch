"""LLaMA-style transformer, written out layer by layer.

Nothing here imports a prebuilt model: RMSNorm, SwiGLU, RoPE and grouped-query
attention are spelled out explicitly so every shape and every rotation is
visible in the code. The one thing delegated to PyTorch is the attention
kernel itself (``scaled_dot_product_attention``), which covers flash attention,
memory-efficient attention and the maths fallback.

Incremental decoding is supported through a KV cache: ``forward`` accepts the
cache returned by the previous call plus the absolute ``start_pos`` of the new
tokens, so generated tokens cost O(context) instead of re-running the whole
prefix.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .config import ModelConfig

# (k, v) each shaped [batch, num_kv_heads, seq_len, head_dim]
KVCache = Tuple[torch.Tensor, torch.Tensor]

# Kernels we would like to land on, in order of preference. If the installed
# build has neither (CPU-only wheels, for example) we fall back to whatever
# PyTorch picks by default instead of raising.
_PREFERRED_BACKENDS = (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    try:
        with sdpa_kernel(list(_PREFERRED_BACKENDS)):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
    except (RuntimeError, AssertionError):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)


class RMSNorm(nn.Module):
    """Root-mean-square normalisation: no mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Gated feed-forward: ``w2(silu(w1 x) * w3 x)``."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings, computed for the positions actually used.

    Frequencies are generated on the fly rather than precomputed into a fixed
    table, which means sequences longer than the training window still work
    (with the usual extrapolation caveat) and a cached decode can ask for any
    absolute offset.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self,
        seq_len: int,
        start_pos: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            start_pos, start_pos + seq_len, dtype=torch.float32, device=self.inv_freq.device
        )
        freqs = torch.outer(positions, self.inv_freq)  # [seq_len, head_dim/2]
        # Duplicate so the two halves of the head rotate as pairs.
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the head dimension in half and rotate: ``[a, b] -> [-b, a]``."""
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to query-head count by repetition (the GQA trick)."""
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """Multi-head attention where K and V have fewer heads than Q."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.n_rep = cfg.kv_group_size
        self.head_dim = cfg.head_dim

        self.q_proj = nn.Linear(cfg.dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, cfg.dim, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, KVCache]:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Rotate at absolute positions so cached keys keep the phase they were
        # written with.
        cos, sin = self.rotary.cos_sin(seq_len, start_pos, dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k = torch.cat((kv_cache[0], k), dim=2)
            v = torch.cat((kv_cache[1], v), dim=2)
        present: KVCache = (k, v)

        # Expanded after caching, so the cache only ever stores KV heads.
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        # Training and prefill see the whole sequence at once, so they need the
        # causal mask. Single-token decoding is already causal by construction:
        # the only query sits at the end and may attend to every cached key.
        is_causal = kv_cache is None and seq_len > 1
        out = _attention(q, k_rep, v_rep, is_causal)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(out), present


class TransformerBlock(nn.Module):
    """Pre-norm block: attention then feed-forward, both residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim)
        self.mlp = SwiGLU(cfg.dim, cfg.mlp_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, KVCache]:
        attn_out, present = self.attn(self.attn_norm(x), kv_cache, start_pos)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


class LLaMA(nn.Module):
    """The full model: embeddings, N blocks, final norm, unembedding."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.padded_vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.padded_vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[Optional[KVCache], ...]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[KVCache, ...]]:
        """Return ``(logits, loss, present_caches)``.

        ``kv_cache`` is the per-layer tuple returned by the previous call, or
        ``None`` for a fresh sequence. ``start_pos`` is the absolute position of
        the first token in ``tokens``.
        """
        _, seq_len = tokens.shape

        h = self.tok_embeddings(tokens)

        present = []
        for i, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache[i]
            h, layer_present = layer(h, layer_cache, start_pos)
            present.append(layer_present)

        logits = self.output(self.norm(h))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.padded_vocab_size), targets.reshape(-1)
            )

        return logits, loss, tuple(present)


def count_parameters(model: nn.Module, non_embedding: bool = False) -> int:
    total = sum(p.numel() for p in model.parameters())
    if non_embedding:
        total -= model.tok_embeddings.weight.numel()
        total -= model.output.weight.numel()
    return total


def kv_cache_bytes(cfg: ModelConfig, seq_len: int, batch_size: int = 1) -> int:
    """Bytes held by a full KV cache, useful for reasoning about VRAM."""
    per_token = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * 2  # fp16/bf16
    return per_token * seq_len * batch_size

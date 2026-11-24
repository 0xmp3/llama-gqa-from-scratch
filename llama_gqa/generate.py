"""Sampling with an incremental KV cache.

Each new token costs one forward pass over the token itself, because the keys
and values of everything before it are reused from the cache. The previous
version of this file re-ran the entire prefix for every token, which is
quadratic in the sequence length.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .model import KVCache, LLaMA


def _sample_next(logits: torch.Tensor, temperature: float, top_k: Optional[int]) -> torch.Tensor:
    """One token per batch row from the final position's logits."""
    if temperature == 0.0:  # greedy
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _trim(cache: Tuple[KVCache, ...], max_len: int) -> Tuple[KVCache, ...]:
    """Drop the oldest entries of every layer's cache to cap memory."""
    trimmed = []
    for k, v in cache:
        if k.size(2) > max_len:
            trimmed.append((k[:, :, -max_len:], v[:, :, -max_len:]))
        else:
            trimmed.append((k, v))
    return tuple(trimmed)


@torch.no_grad()
def generate(
    model: LLaMA,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Autoregressively extend ``idx`` and return the full token sequence.

    The context is a sliding window of ``model.cfg.max_seq_len`` tokens. Past
    that, the oldest cached entries are dropped while absolute positions keep
    increasing, which is the standard trade: bounded memory, positions beyond
    the trained range.
    """
    model.eval()
    window = model.cfg.max_seq_len

    logits, _, cache = model(idx, start_pos=0)
    tokens = [idx]
    pos = idx.size(1)

    for _ in range(max_new_tokens):
        next_token = _sample_next(logits[:, -1, :], temperature, top_k)
        tokens.append(next_token)

        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

        cache = _trim(cache, window - 1)
        logits, _, cache = model(next_token, kv_cache=cache, start_pos=pos)
        pos += 1

    return torch.cat(tokens, dim=1)

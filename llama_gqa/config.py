"""Architecture hyperparameters for the from-scratch LLaMA.

The config lives in its own object because it is serialised into every
checkpoint: training and inference read the same source of truth instead of
keeping two copies of the numbers that drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

# Unembedding rows are padded to this multiple so the final projection stays
# aligned for tensor-core shaped matmuls (50,257 -> 50,304 for GPT-2 BPE).
VOCAB_ALIGNMENT = 128

# SwiGLU hidden width is rounded to this multiple, as in the LLaMA reference.
MLP_ALIGNMENT = 256


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass
class ModelConfig:
    vocab_size: int = 50257  # GPT-2 BPE vocabulary, before padding
    dim: int = 768
    num_layers: int = 6
    num_heads: int = 12
    num_kv_heads: int = 4  # GQA: every 3 query heads share one KV head
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    mlp_hidden_dim: Optional[int] = None  # None => 8/3*dim, rounded up

    def __post_init__(self) -> None:
        if self.dim % self.num_heads:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.mlp_hidden_dim is None:
            self.mlp_hidden_dim = round_up(8 * self.dim // 3, MLP_ALIGNMENT)

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def padded_vocab_size(self) -> int:
        return round_up(self.vocab_size, VOCAB_ALIGNMENT)

    @property
    def kv_group_size(self) -> int:
        """How many query heads are served by one KV head."""
        return self.num_heads // self.num_kv_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        return cls(**values)

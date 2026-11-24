import pytest
import torch

from llama_gqa import LLaMA, ModelConfig


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    """Small enough to run a full suite on CPU in seconds."""
    return ModelConfig(
        vocab_size=256,
        dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
    )


@pytest.fixture
def tiny_model(tiny_cfg: ModelConfig) -> LLaMA:
    torch.manual_seed(0)
    model = LLaMA(tiny_cfg)
    model.eval()
    return model

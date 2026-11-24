"""Shape, size and causal-masking checks for the model."""

import pytest
import torch

from llama_gqa import LLaMA, ModelConfig, count_parameters, kv_cache_bytes
from llama_gqa.model import RotaryEmbedding, repeat_kv, rotate_half


def test_vocab_padding_and_mlp_width():
    cfg = ModelConfig()
    assert cfg.padded_vocab_size == 50304  # 50257 rounded up to a multiple of 128
    assert cfg.mlp_hidden_dim == 2048  # 8/3 * 768, already a multiple of 256
    assert cfg.head_dim == 64
    assert cfg.kv_group_size == 3  # 12 query heads over 4 KV heads


def test_invalid_head_split_rejected():
    with pytest.raises(ValueError):
        ModelConfig(dim=768, num_heads=10)  # 768 is not divisible by 10
    with pytest.raises(ValueError):
        ModelConfig(dim=768, num_heads=12, num_kv_heads=5)  # 12 not divisible by 5


def test_parameter_count_of_shipped_config():
    model = LLaMA(ModelConfig())
    assert count_parameters(model) == 115_025_664
    # Embedding and unembedding are untied, so each is 38,633,472 parameters.
    assert count_parameters(model, non_embedding=True) == 37_758_720


def test_forward_shapes(tiny_model):
    tokens = torch.randint(0, tiny_model.cfg.vocab_size, (3, 10))
    logits, loss, present = tiny_model(tokens)
    assert logits.shape == (3, 10, tiny_model.cfg.padded_vocab_size)
    assert loss is None
    assert len(present) == tiny_model.cfg.num_layers
    k, v = present[0]
    assert k.shape == (3, tiny_model.cfg.num_kv_heads, 10, tiny_model.cfg.head_dim)
    assert v.shape == k.shape


def test_loss_matches_manual_cross_entropy(tiny_model):
    tokens = torch.randint(0, tiny_model.cfg.vocab_size, (2, 8))
    targets = torch.randint(0, tiny_model.cfg.vocab_size, (2, 8))
    logits, loss, _ = tiny_model(tokens, targets)
    manual = torch.nn.functional.cross_entropy(
        logits.view(-1, tiny_model.cfg.padded_vocab_size), targets.view(-1)
    )
    assert torch.allclose(loss, manual)


def test_attention_is_causal(tiny_model):
    """Changing the last token must not change earlier positions' logits."""
    tokens = torch.randint(0, tiny_model.cfg.vocab_size, (2, 9))
    perturbed = tokens.clone()
    perturbed[:, -1] = (tokens[:, -1] + 7) % tiny_model.cfg.vocab_size

    with torch.no_grad():
        before, _, _ = tiny_model(tokens)
        after, _, _ = tiny_model(perturbed)

    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6)
    assert not torch.allclose(before[:, -1], after[:, -1], atol=1e-3)


def test_repeat_kv_groups():
    x = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    out = repeat_kv(x, 3)
    assert out.shape == (2, 6, 3, 4)
    # Head 0..2 are copies of KV head 0, head 3..5 are copies of KV head 1.
    assert torch.equal(out[:, 1], x[:, 0])
    assert torch.equal(out[:, 2], x[:, 0])
    assert torch.equal(out[:, 4], x[:, 1])


def test_rotate_half_is_a_quarter_turn():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rope_offsets_match_direct_computation():
    """cos/sin for a window must equal the same rows of a longer window."""
    rope = RotaryEmbedding(head_dim=8)
    cos_full, sin_full = rope.cos_sin(12, start_pos=0)
    cos_offset, sin_offset = rope.cos_sin(4, start_pos=8)
    assert torch.allclose(cos_offset, cos_full[8:12], atol=1e-6)
    assert torch.allclose(sin_offset, sin_full[8:12], atol=1e-6)
    # Position 0 is the identity rotation.
    assert torch.allclose(cos_full[0], torch.ones(8), atol=1e-6)
    assert torch.allclose(sin_full[0], torch.zeros(8), atol=1e-6)


def test_kv_cache_size_accounting():
    cfg = ModelConfig()
    per_token = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * 2
    assert kv_cache_bytes(cfg, seq_len=256) == per_token * 256
    # Storing 4 KV heads instead of 12 is what keeps this small.
    assert cfg.num_kv_heads * 3 == cfg.num_heads

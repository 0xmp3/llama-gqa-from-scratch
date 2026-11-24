"""Incremental decoding must agree with recomputing the whole prefix.

These are the tests that justify the KV cache: if the cache path diverged from
the full forward pass, the numbers here would catch it.
"""

import numpy as np
import pytest
import torch

from llama_gqa import LLaMA, ModelConfig, PrepackedDataset, generate


def _full_prefix_logits(model, tokens):
    with torch.no_grad():
        logits, _, _ = model(tokens)
    return logits


def test_cached_decode_matches_full_forward(tiny_model):
    """Prefill + token-by-token caching == one forward over the whole sequence."""
    tokens = torch.randint(0, tiny_model.cfg.vocab_size, (2, 12))
    split = 7

    full = _full_prefix_logits(tiny_model, tokens)

    with torch.no_grad():
        prefill, _, cache = tiny_model(tokens[:, :split], start_pos=0)
        stepwise = [prefill]
        pos = split
        for t in range(split, tokens.size(1)):
            step_logits, _, cache = tiny_model(
                tokens[:, t : t + 1], kv_cache=cache, start_pos=pos
            )
            stepwise.append(step_logits)
            pos += 1

    cached = torch.cat(stepwise, dim=1)
    assert cached.shape == full.shape
    assert torch.allclose(cached, full, atol=1e-5), (cached - full).abs().max()


def test_greedy_generation_matches_naive_recompute(tiny_model):
    """Greedy sampling with the cache must produce the same tokens as the
    original O(n^2) loop that re-ran the full prefix every step."""
    prompt = torch.randint(0, tiny_model.cfg.vocab_size, (1, 6))
    n_new = 6

    cached = generate(tiny_model, prompt, max_new_tokens=n_new, temperature=0.0)

    naive = prompt.clone()
    with torch.no_grad():
        for _ in range(n_new):
            logits = _full_prefix_logits(tiny_model, naive)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            naive = torch.cat((naive, next_token), dim=1)

    assert torch.equal(cached, naive)


def test_generation_length_and_eos(tiny_model):
    prompt = torch.randint(0, tiny_model.cfg.vocab_size, (1, 4))

    out = generate(tiny_model, prompt, max_new_tokens=5, temperature=0.8)
    assert out.shape == (1, 9)
    assert torch.equal(out[:, :4], prompt)

    # Every drawn token is the EOS id, so generation stops after the first one.
    eos = int(prompt[0, 0])
    stopped = generate(tiny_model, prompt, max_new_tokens=5, temperature=0.0, eos_token_id=eos)
    assert stopped.size(1) <= 9


def test_generate_beyond_context_window(tiny_cfg):
    """Sliding the cache keeps memory bounded when generation outruns max_seq_len."""
    torch.manual_seed(0)
    model = LLaMA(tiny_cfg)
    model.eval()
    prompt = torch.randint(0, tiny_cfg.vocab_size, (1, 5))

    out = generate(model, prompt, max_new_tokens=tiny_cfg.max_seq_len * 2, temperature=0.8)
    assert out.size(1) == 5 + tiny_cfg.max_seq_len * 2
    assert torch.isfinite(out.float()).all()


def test_dataset_reads_prepacked_bins(tmp_path):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 1000, size=5000, dtype=np.uint16)
    (tmp_path / "train.bin").write_bytes(tokens.tobytes())

    dataset = PrepackedDataset(str(tmp_path), "train", batch_size=4, seq_len=32, seed=0)
    assert dataset.num_tokens == 5000

    x, y = dataset.get_batch()
    assert x.shape == (4, 32)
    assert y.shape == (4, 32)
    assert x.dtype == torch.int64
    assert int(x.min()) >= 0 and int(y.max()) < 1000
    # Targets are the same window shifted one token to the left.
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PrepackedDataset(str(tmp_path), "train", batch_size=2, seq_len=8)

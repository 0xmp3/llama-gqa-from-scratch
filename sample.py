"""Generate text from a trained checkpoint.

The architecture comes out of the checkpoint itself, so there is no second copy
of the hyperparameters to keep in sync.

Usage:
    python sample.py --ckpt checkpoints/best.pt --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from llama_gqa import LLaMA, ModelConfig, generate


def main() -> None:
    parser = argparse.ArgumentParser(description="sample from a checkpoint")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--prompt", default="Once upon a time, a girl named Lily found a shiny blue key in her garden.")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(checkpoint["config"])
    model = LLaMA(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    prompt = torch.tensor([ids], dtype=torch.long, device=device)

    out = generate(
        model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    print(text)
    new_tokens = out.size(1) - prompt.size(1)
    print(f"\n[{new_tokens} tokens generated]")


if __name__ == "__main__":
    main()

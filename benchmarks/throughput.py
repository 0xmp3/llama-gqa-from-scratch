"""Measure training step time and token throughput on whatever device is present.

This reports the same two numbers quoted in the README (milliseconds per
optimizer step and tokens per second) so the claim can be checked on other
hardware.

Usage:
    python benchmarks/throughput.py --steps 20
    python benchmarks/throughput.py --steps 20 --batch-size 32 --grad-accum 4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_gqa import LLaMA, ModelConfig, count_parameters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="training throughput benchmark")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    use_amp = device == "cuda" and amp_dtype != torch.float32

    cfg = ModelConfig(
        dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        max_seq_len=args.seq_len,
    )
    model = LLaMA(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))

    tokens_per_step = args.batch_size * cfg.max_seq_len * args.grad_accum
    takes = []

    model.train()
    for step in range(args.steps + args.warmup):
        x = torch.randint(0, cfg.vocab_size, (args.batch_size, cfg.max_seq_len), device=device)
        y = torch.randint(0, cfg.vocab_size, (args.batch_size, cfg.max_seq_len), device=device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss, _ = model(x, y)
                (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if step >= args.warmup:
            takes.append(elapsed)

    median = statistics.median(takes)
    peak_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else float("nan")
    )

    print(f"device            {device}")
    print(f"parameters        {count_parameters(model):,}")
    print(f"tokens per step   {tokens_per_step:,}")
    print(f"step time (median) {median * 1000:.1f} ms")
    print(f"throughput        {tokens_per_step / median:,.0f} tok/s")
    if device == "cuda":
        print(f"peak VRAM         {peak_gb:.1f} GB")


if __name__ == "__main__":
    main()

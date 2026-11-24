"""Training loop for the from-scratch LLaMA on TinyStories.

Same hyperparameters as the measured run: 6 layers of width 768, 12 query
heads over 4 KV heads, 256-token windows, micro-batch 32 with 4 accumulation
steps (32,768 tokens per optimizer step), AdamW with cosine decay, bf16
autocast, 2,000 steps.

Usage:
    python tokenize_data.py          # writes data/train.bin, data/validation.bin
    python train.py                  # trains, writes checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch

from llama_gqa import LLaMA, ModelConfig, PrepackedDataset, count_parameters


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return base_lr * min_ratio
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * min_ratio + coeff * base_lr * (1.0 - min_ratio)


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float = 0.1) -> torch.optim.Optimizer:
    """Decay 2-D weights only; norms and biases keep weight_decay 0."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = [p for _, p in named if p.dim() >= 2]
    no_decay = [p for _, p in named if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def pick_amp_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = pick_amp_dtype(device)
    amp_enabled = device == "cuda"

    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        max_seq_len=args.seq_len,
    )
    model = LLaMA(cfg).to(device)

    print(f"device            {device}")
    print(f"amp dtype         {amp_dtype}" if amp_enabled else "amp               disabled")
    print(f"parameters        {count_parameters(model):,} ({count_parameters(model, non_embedding=True):,} non-embedding)")
    print(f"vocab             {cfg.vocab_size:,} padded to {cfg.padded_vocab_size:,}")

    train_set = PrepackedDataset(args.data_dir, "train", args.batch_size, cfg.max_seq_len, seed=args.seed)
    val_set = PrepackedDataset(args.data_dir, "validation", args.batch_size, cfg.max_seq_len, seed=args.seed + 1)
    print(f"tokens            train {train_set.num_tokens:,} / val {val_set.num_tokens:,}")

    optimizer = build_optimizer(model, args.lr)
    tokens_per_step = args.batch_size * cfg.max_seq_len * args.grad_accum
    print(f"tokens per step   {tokens_per_step:,} ({args.batch_size} x {cfg.max_seq_len} x {args.grad_accum})")
    print(f"planned tokens    {tokens_per_step * args.max_steps:,} over {args.max_steps} steps\n")

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    last_val = float("nan")  # final.pt must not claim the best checkpoint's score
    model.train()

    for step in range(args.max_steps):
        lr = cosine_lr(step, args.max_steps, args.lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0

        for _ in range(args.grad_accum):
            x, y = train_set.get_batch(device)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                _, loss, _ = model(x, y)
                scaled = loss / args.grad_accum
            loss_sum += loss.detach().item()
            scaled.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if step % args.log_every == 0:
            throughput = tokens_per_step / elapsed
            print(
                f"step {step:5d} | loss {loss_sum / args.grad_accum:.4f} | lr {lr:.2e} "
                f"| grad_norm {grad_norm:.3f} | {elapsed * 1000:.0f} ms | {throughput:,.0f} tok/s"
            )

        if args.eval_every and step > 0 and step % args.eval_every == 0:
            val_loss = evaluate(model, val_set, device, amp_dtype, amp_enabled, args.eval_batches)
            marker = ""
            last_val = val_loss
            if val_loss < best_val:
                best_val = val_loss
                marker = " (best)"
                save_checkpoint(model, cfg, optimizer, step, val_loss, os.path.join(args.out_dir, "best.pt"), best_val)
            print(f"step {step:5d} | val loss {val_loss:.4f}{marker}")

    save_checkpoint(model, cfg, optimizer, args.max_steps, last_val, os.path.join(args.out_dir, "final.pt"), best_val)
    print(f"\nfinal checkpoint written to {os.path.join(args.out_dir, 'final.pt')}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: PrepackedDataset,
    device: str,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    batches: int,
) -> float:
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = dataset.get_batch(device)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            _, loss, _ = model(x, y)
        total += loss.item()
    model.train()
    return total / batches


def save_checkpoint(
    model, cfg: ModelConfig, optimizer, step: int, val_loss: float, path: str, best_val: float
) -> None:
    """Weights plus the config, so inference cannot drift from training.

    ``val_loss`` is the most recent validation measurement at the time of saving
    (NaN when no evaluation ran); ``best_val`` is the best seen anywhere in the
    run. They are different numbers whenever a later checkpoint fails to improve.
    """
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "best_val": best_val,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train the from-scratch LLaMA on TinyStories")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

"""Tokenise TinyStories into flat uint16 binaries.

Writes each batch straight to disk instead of accumulating a Python list of
integers: ~474M boxed ints would cost several GB of RAM on their own.

Usage:
    python tokenize_data.py --out-dir data
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

DTYPE = np.uint16
EOS_APPENDED_PER_STORY = 1


def tokenize_split(tokenizer, dataset_split, out_path: str) -> int:
    total = 0
    with open(out_path, "wb") as handle:
        for batch in tqdm(dataset_split.iter(batch_size=1024), desc=os.path.basename(out_path)):
            encoded = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
            for ids in encoded:
                # One EOS per story so documents do not run together.
                row = np.asarray(ids + [tokenizer.eos_token_id], dtype=DTYPE)
                handle.write(row.tobytes())
                total += row.size
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="pre-tokenise TinyStories")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--tokenizer", default="gpt2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = load_dataset(args.dataset)

    # Tokenising per batch and appending, so the tokeniser warning about long
    # stories is expected and harmless (they are truncated at its 1024 limit).
    for split in ("train", "validation"):
        path = os.path.join(args.out_dir, f"{split}.bin")
        tokens = tokenize_split(tokenizer, dataset[split], path)
        print(f"{split}: {tokens:,} tokens -> {path}")


if __name__ == "__main__":
    main()

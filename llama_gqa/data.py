"""Reader for the pre-tokenised ``.bin`` files produced by ``tokenize_data.py``.

The corpus is a flat array of uint16 token ids. Batches are random windows,
copied straight out of the memory map, so the process resident set stays small
regardless of corpus size.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch

DTYPE = np.uint16  # max 65,535 covers the GPT-2 vocabulary with room to spare


class PrepackedDataset:
    def __init__(
        self,
        data_dir: str,
        split: str,
        batch_size: int,
        seq_len: int,
        seed: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

        path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing {path}; run tokenize_data.py first (it writes train.bin and validation.bin)"
            )

        self.data = np.memmap(path, dtype=DTYPE, mode="r")
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"{path} holds {len(self.data)} tokens, too few for seq_len={seq_len}")

    @property
    def num_tokens(self) -> int:
        return int(self.data.shape[0])

    def __len__(self) -> int:
        return self.num_tokens

    def get_batch(self, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``batch_size`` contiguous windows and shift them by one token."""
        high = self.num_tokens - (self.seq_len + 1)
        starts = self.rng.integers(0, high, size=self.batch_size)

        # One gather for the whole batch instead of a Python loop per window.
        index = starts[:, None] + np.arange(self.seq_len + 1)[None, :]
        block = self.data[index]

        x = torch.from_numpy(block[:, :-1].astype(np.int64))
        y = torch.from_numpy(block[:, 1:].astype(np.int64))

        if device.startswith("cuda") and torch.cuda.is_available():
            return (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        return x.to(device), y.to(device)

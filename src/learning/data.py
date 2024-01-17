from typing import Tuple, Iterable

import numpy as np
import torch
from torch.utils import data


class Data(data.Dataset):
    """Default dataset for Torch."""

    def __init__(self, x: Iterable, y: Iterable):
        assert len(x) == len(y), f"Data should have the same length, but len(x) = {len(x)} and len(y) = {len(y)}"
        self.x: torch.Tensor = torch.tensor(np.array(x), dtype=torch.float32)
        self.y: torch.Tensor = torch.tensor(np.expand_dims(y, axis=-1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class HGR:
    """Interface for an object that computes the HGR correlation differentiable way."""

    @abstractmethod
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Computes the correlation between two tensors <a> and <b> in a differentiable way."""
        pass

    def correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        """Computes the correlation between two numpy vectors <a> and <b>."""
        a = torch.tensor(a, dtype=torch.float)
        b = torch.tensor(b, dtype=torch.float)
        return self(a=a, b=b).numpy(force=True).item()

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch

from src.serializable import Serializable


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class HGR(Serializable):
    """Interface for an object that computes the HGR correlation differentiable way."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the HGR metric."""
        pass

    @abstractmethod
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Computes the correlation between two tensors <a> and <b> in a differentiable way."""
        pass

    @abstractmethod
    def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        """Computes the correlation between two numpy vectors <a> and <b> and returns a dictionary of type
        {correlation: <float>, **additional_results}."""
        pass

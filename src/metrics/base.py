from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class Metric:
    """Interface for a metric."""

    name: str = field()
    """The name of the metric."""

    @abstractmethod
    def __call__(self, x, y: np.ndarray, p: np.ndarray) -> float:
        """Computes the metric value respectively to the input (x), the ground truths (y), and the predictions (p)."""
        pass


@dataclass(frozen=True)
class CorrelationMetric(Metric):
    """Interface for a correlation metric."""

    feature: int = field()
    """The index of the feature to inspect."""

    @abstractmethod
    def correlation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Computes the correlation between two tensors <a> and <b> in a differentiable way."""
        pass

    def __call__(self, x, y: np.ndarray, p: np.ndarray) -> float:
        x = torch.tensor(np.array(x)[:, self.feature], dtype=torch.float)
        p = torch.tensor(np.array(p), dtype=torch.float)
        return self.correlation(a=x, b=p).item()

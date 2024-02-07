from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any

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
    def correlation(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Computes the correlation between two numpy vectors <a> and <b> and returns the dictionary along with a
        dictionary of additional results."""
        pass

    @abstractmethod
    def __call__(self, a: torch.Tensor, b: torch.Tensor, kwargs: Dict[str, Any]) -> torch.Tensor:
        """Computes the correlation between two tensors <a> and <b> in a differentiable way.
        Additionally, kwargs are used both for additional input parameters and additional output storage."""
        pass


class KernelsHGR(HGR):
    """Interface for an HGR object that also allows to inspect kernels."""

    @abstractmethod
    def _kernels(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the f(a) and g(b) kernels given the two input vectors and the result of the experiments."""
        pass

    def kernels(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[float, np.ndarray, np.ndarray]:
        """Returns the f(a) and g(b) kernels, along with the computed correlation, given the two input vectors and the
        result of the experiments."""
        assert self == experiment.metric, f'Unexpected metric {experiment.metric} when computing kernels'
        fa, gb = self._kernels(a=a, b=b, experiment=experiment)
        fa = (fa - fa.mean()) / fa.std(ddof=0)
        gb = (gb - gb.mean()) / gb.std(ddof=0)
        return abs(np.mean(fa * gb)), fa, gb

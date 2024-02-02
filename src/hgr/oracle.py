from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr

from src.datasets import Deterministic
from src.hgr import KernelsHGR
from src.serializable import Serializable


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Oracle(KernelsHGR):
    """A metric that knows the exact relationship of a deterministic dataset. The factory class trick is used not to
    break compatibility with DoE implementation, since an oracle depends on the dataset which is being used."""

    @dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
    class _Instance(KernelsHGR):
        dataset: Deterministic = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=None)

        @property
        def name(self) -> str:
            return 'oracle'

        @property
        def configuration(self) -> Dict[str, Any]:
            return dict(name=self.name, dataset=self.dataset.configuration)

        @property
        def key(self) -> str:
            return f'{self.name}-{self.dataset.key}'

        def _kernels(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
            assert self.dataset == experiment.dataset, f'Unexpected dataset {experiment.dataset} when computing kernels'
            return self.dataset.f(a), self.dataset.g(b)

        def correlation(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, Dict[str, Any]]:
            dataset = self.dataset
            correlation, _ = pearsonr(dataset.f(a), dataset.g(b))
            return abs(float(correlation)), dict()

        def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            raise AssertionError("Oracle metric does not provide gradients")

    # noinspection PyMethodMayBeStatic
    def instance(self, dataset: Deterministic) -> _Instance:
        return Oracle._Instance(dataset=dataset)

    def _kernels(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
        return experiment.dataset.f(a), experiment.dataset.g(b)

    def __eq__(self, other: Serializable) -> bool:
        return isinstance(other, Oracle._Instance)

    @property
    def name(self) -> str:
        raise AssertionError("Oracle is a factory object, please call method '.instance()' to get a valid metric")

    @property
    def configuration(self) -> Dict[str, Any]:
        raise AssertionError("Oracle is a factory object, please call method '.instance()' to get a valid metric")

    @property
    def key(self) -> str:
        raise AssertionError("Oracle is a factory object, please call method '.instance()' to get a valid metric")

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        raise AssertionError("Oracle is a factory object, please call method '.instance()' to get a valid metric")

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Oracle is a factory object, please call method '.instance()' to get a valid metric")

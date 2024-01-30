from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr

from src.datasets.deterministic import Deterministic
from src.hgr import HGR


def init():
    raise AssertionError("Oracle constructor should be passed a deterministic dataset as argument")


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Oracle(HGR):
    dataset: Deterministic = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default_factory=init)

    @property
    def name(self) -> str:
        return 'oracle'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, dataset=self.dataset.configuration)

    @property
    def key(self) -> str:
        return f'{self.name}-{self.dataset.key}'

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        dataset = self.dataset
        correlation, _ = pearsonr(dataset.f(a), dataset.g(b))
        return abs(float(correlation)), dict()

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Oracle metric does not provide gradients")

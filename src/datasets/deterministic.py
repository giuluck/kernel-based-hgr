from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import pi

import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset


@dataclass(frozen=True, kw_only=True)
class Deterministic(Dataset, ABC):
    size: int = field(kw_only=True, default=101)
    """The size of the dataset."""

    noise: float = field(kw_only=True, default=0.0)
    """The amount of noise to be introduced in the target data."""

    linspace: bool = field(kw_only=True, default=True)
    """Whether to build the protected data from a linear space, or sample it uniformly."""

    seed: int = field(kw_only=True, default=0)
    """The random seed for generating the dataset."""

    def _load(self) -> pd.DataFrame:
        rng = np.random.default_rng(seed=self.seed)
        # take x within the interval [-1, 1] then duplicate it in order to have swapped signs for y if necessary
        s = np.linspace(0, 1, num=self.size, endpoint=True) if self.linspace else rng.uniform(0, 1, size=self.size)
        x = np.concatenate((s, s[::-1]))
        # build y according to the function then normalize it in order to add a correct relative amount of noise
        y = self.function(x)
        y = (y - y.min()) / (y.max() - y.min()) + rng.normal(loc=0.0, scale=self.noise, size=len(y))
        # build the dataframe with standardized input and normalized output
        return pd.DataFrame({
            'x': (x - x.mean()) / x.std(ddof=0),
            'y': (y - y.min()) / (y.max() - y.min())
        })

    @abstractmethod
    def function(self, x: np.ndarray) -> np.ndarray:
        """Builds the target data given the protected (input) data."""
        pass

    @property
    def classification(self) -> bool:
        return False

    @property
    def continuous(self) -> bool:
        return True

    @property
    def excluded_name(self) -> str:
        return 'x'

    @property
    def target_name(self) -> str:
        return 'y'


@dataclass(frozen=True, kw_only=True)
class Polynomial(Deterministic):
    degree_x: int = field(kw_only=True, default=1)
    """The degree of the protected data in the deterministic relationship."""

    degree_y: int = field(kw_only=True, default=1)
    """The degree of the target data in the deterministic relationship."""

    @property
    def name(self) -> str:
        return f'deterministic {self.degree_x}-{self.degree_y}'

    def function(self, x: np.ndarray) -> np.ndarray:
        # rescale x from [0, 1] to [-1, 1]
        x = 2 * x - 1
        # build y so that the following relationship holds: x^dx + y^dy = 1
        # in order to do that, compute y as: y = (1 - x^dx) ^ (1 / dy)
        # then, if the degree of y is odd swap the signs of the first half, otherwise take positive signs only
        sign = np.array([-1 if self.degree_y % 2 == 0 else 1] * self.size + [1] * self.size)
        return -sign * np.power(1 - x ** self.degree_x, 1.0 / self.degree_y)


@dataclass(frozen=True, kw_only=True)
class NonLinear(Deterministic):
    name: str = field(kw_only=True, default='relu')
    """The name of non-linear relationship (one in 'sign', 'relu', 'sin', 'tanh')."""

    def function(self, x: np.ndarray) -> np.ndarray:
        if self.name == 'sign':
            # apply the function to the input vector rescaled from [0, 1] to [-1, 1]
            return np.sign(2 * x - 1)
        elif self.name == 'relu':
            # apply the function to the input vector rescaled from [0, 1] to [-1, 1]
            return np.maximum(0, 2 * x - 1)
        elif self.name == 'sin':
            # apply the function to the input vector rescaled from [0, 1] to [0, 2 * pi]
            return np.sin(2 * pi * x)
        elif self.name == 'tanh':
            # apply the function to the input vector rescaled from [0, 1] to [-10, 10]
            return np.tanh(20 * x - 10)
        else:
            raise AssertionError(f"Unknown non-linear function name '{self.name}'")

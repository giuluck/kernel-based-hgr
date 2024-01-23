from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import pi
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset

SIZE: int = 101
"""The size of the dataset."""

LINSPACE: bool = True
"""Whether to build the protected data from a linear space, or sample it uniformly."""

SEED: int = 0
"""The random seed for generating the dataset."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Deterministic(Dataset, ABC):
    noise: float = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=0.0)
    """The amount of noise to be introduced in the target data."""

    def _load(self) -> pd.DataFrame:
        rng = np.random.default_rng(seed=SEED)
        # take x within the interval [-1, 1] then duplicate it in order to have swapped signs for y if necessary
        s = np.linspace(0, 1, num=SIZE, endpoint=True) if LINSPACE else rng.uniform(0, 1, size=SIZE)
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

    def plot(self, ax: plt.Axes, **kwargs):
        ax.plot(self.excluded(backend='numpy'), self.target(backend='numpy'), **kwargs)


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Polynomial(Deterministic):
    degree_x: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=1)
    """The degree of the protected data in the deterministic relationship."""

    degree_y: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=1)
    """The degree of the target data in the deterministic relationship."""

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, degree_x=self.degree_x, degree_y=self.degree_y, noise=self.noise)

    @property
    def name(self) -> str:
        return 'poly'

    def function(self, x: np.ndarray) -> np.ndarray:
        # rescale x from [0, 1] to [-1, 1]
        x = 2 * x - 1
        # build y so that the following relationship holds: x^dx + y^dy = 1
        # in order to do that, compute y as: y = (1 - x^dx) ^ (1 / dy)
        # then, if the degree of y is odd swap the signs of the first half, otherwise take positive signs only
        sign = np.array([-1 if self.degree_y % 2 == 0 else 1] * SIZE + [1] * SIZE)
        return -sign * np.power(1 - x ** self.degree_x, 1.0 / self.degree_y)


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class NonLinear(Deterministic):
    fn: str = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default='relu')
    """The name of non-linear relationship (one in 'sign', 'relu', 'sin', 'tanh')."""

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, fn=self.fn, noise=self.noise)

    @property
    def name(self) -> str:
        return 'nonlinear'

    def function(self, x: np.ndarray) -> np.ndarray:
        if self.fn == 'sign':
            # apply the function to the input vector rescaled from [0, 1] to [-1, 1]
            return np.sign(2 * x - 1)
        elif self.fn == 'relu':
            # apply the function to the input vector rescaled from [0, 1] to [-1, 1]
            return np.maximum(0, 2 * x - 1)
        elif self.fn == 'sin':
            # apply the function to the input vector rescaled from [0, 1] to [0, 2 * pi]
            return np.sin(2 * pi * x)
        elif self.fn == 'tanh':
            # apply the function to the input vector rescaled from [0, 1] to [-10, 10]
            return np.tanh(20 * x - 10)
        else:
            raise AssertionError(f"Unknown non-linear function name '{self.fn}'")

    def plot(self, ax: plt.Axes, **kwargs):
        # custom plot for 'sign' function to show the non-continuity
        if self.fn == 'sign':
            x = self.excluded(backend='numpy')
            y = self.target(backend='numpy')
            ax.plot(x[x < 0], y[x < 0], **kwargs)
            ax.plot(x[x > 0], y[x > 0], **kwargs)
        else:
            super().plot(ax=ax, **kwargs)

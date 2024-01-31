from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import pi
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset

SIZE: int = 1001
"""The size of the dataset."""

LINSPACE: bool = True
"""Whether to build the protected data from a linear space, or sample it uniformly."""

SEED: int = 10
"""The random seed for generating the dataset."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Deterministic(Dataset, ABC):
    noise: float = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=0.0)
    """The amount of noise to be introduced in the target data."""

    def from_seed(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns an alternative version of the same dataset generated using a given random seed."""
        rng = np.random.default_rng(seed=seed)
        # take x within the interval [-1, 1] then duplicate it in order to have swapped signs for y if necessary
        s = np.linspace(-1, 1, num=SIZE, endpoint=True) if LINSPACE else rng.uniform(0, 1, size=SIZE)
        x = np.concatenate((s, s[::-1]))
        # build y according to the function and add noise (proportional to the standard deviation of the data)
        y = self._function(x=x)
        y = y + rng.normal(loc=0.0, scale=self.noise * y.std(ddof=0), size=len(y))
        # return the data
        return x, y

    def _load(self) -> pd.DataFrame:
        x, y = self.from_seed(seed=SEED)
        return pd.DataFrame({'x': x, 'y': y})

    @abstractmethod
    def _function(self, x: np.ndarray) -> np.ndarray:
        """Computes the target data given the protected (input) data and the training (input) data."""
        pass

    @abstractmethod
    def f(self, a: np.ndarray) -> np.ndarray:
        """Maps the protected (input) data in the correlation space using the optimal f kernel."""
        pass

    @abstractmethod
    def g(self, b: np.ndarray) -> np.ndarray:
        """Maps the target data in the correlation space using the optimal g kernel."""
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
        # use lineplot in case the noise is null
        if self.noise == 0.0:
            ax.plot(self.excluded(backend='numpy'), self.target(backend='numpy'), **kwargs)
        else:
            super(Deterministic, self).plot(ax=ax, **kwargs)


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

    def _function(self, x: np.ndarray) -> np.ndarray:
        # build y so that the following relationship holds: x^dx + y^dy = 1
        # in order to do that, compute y as: y = (1 - x^dx) ^ (1 / dy)
        # then, if the degree of y is odd swap the signs of the first half, otherwise take positive signs only
        sign = np.array([-1 if self.degree_y % 2 == 0 else 1] * SIZE + [1] * SIZE)
        return -sign * np.power(1 - x ** self.degree_x, 1.0 / self.degree_y)

    def f(self, a: np.ndarray) -> np.ndarray:
        return 1 - a ** self.degree_x

    def g(self, b: np.ndarray) -> np.ndarray:
        return (-b) ** self.degree_y


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

    def _function(self, x: np.ndarray) -> np.ndarray:
        if self.fn == 'sign':
            return np.sign(x)
        elif self.fn == 'relu':
            return np.maximum(0, x)
        elif self.fn == 'sin':
            # apply the function to the input vector rescaled by a factor of \pi
            return np.sin(pi * x)
        elif self.fn == 'tanh':
            # apply the function to the input vector rescaled by a factor of 10
            return np.tanh(10 * x)
        else:
            raise AssertionError(f"Unknown non-linear function name '{self.fn}'")

    def f(self, a: np.ndarray) -> np.ndarray:
        return self._function(a)

    def g(self, b: np.ndarray) -> np.ndarray:
        return b

    def plot(self, ax: plt.Axes, **kwargs):
        # use custom plot for 'sign' function without noise to show the non-continuity
        if self.fn == 'sign' and self.noise == 0.0:
            x = self.excluded(backend='numpy')
            y = self.target(backend='numpy')
            ax.plot(x[x < 0], y[x < 0], **kwargs)
            ax.plot(x[x > 0], y[x > 0], **kwargs)
        else:
            super(NonLinear, self).plot(ax=ax, **kwargs)

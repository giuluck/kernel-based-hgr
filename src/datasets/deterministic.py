from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import pi
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets.dataset import Dataset


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Deterministic(Dataset, ABC):
    seed: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=0)
    """The random seed for generating the dataset."""

    noise: float = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=0.0)
    """The amount of noise to be introduced in the target data."""

    size: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=1001)
    """The size of the dataset."""

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
    def units(self) -> List[int]:
        return []

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
        return dict(
            name=self.name,
            degree_x=self.degree_x,
            degree_y=self.degree_y,
            noise=self.noise,
            seed=self.seed,
            size=self.size
        )

    @property
    def name(self) -> str:
        return 'poly'

    def _load(self) -> pd.DataFrame:
        rng = np.random.default_rng(seed=self.seed)
        # build the input space (either as a linspace if the noise is null, otherwise sample from uniform distribution)
        if self.noise == 0.0:
            s = np.linspace(-1, 1, num=self.size, endpoint=True)
        else:
            s = rng.uniform(-1, 1, size=self.size)
        # select the strategy depending on the direction of the dependency:
        #   - if degree_y == 1, then y = f(x) so we add noise only on y
        #   - if degree_x == 1, then x = g(y) so we add noise only on x
        #   - otherwise, x and y are co-dependent, so we add noise on both
        if self.degree_y == 1:
            x = s
            y = self.f(x)
            y = y + rng.normal(loc=0.0, scale=self.noise * y.std(ddof=0), size=len(y))
        elif self.degree_x == 1:
            y = s
            x = -self.g(y)
            x = x + rng.normal(loc=0.0, scale=self.noise * x.std(ddof=0), size=len(x))
        else:
            x = np.concatenate((s, s[::-1]))
            sign = np.array([-1 if self.degree_y % 2 == 0 else 1] * self.size + [1] * self.size)
            y = sign * np.power(1 - x ** self.degree_x, 1.0 / self.degree_y)
            x = x + rng.normal(loc=0.0, scale=self.noise * x.std(ddof=0), size=len(x))
            y = y + rng.normal(loc=0.0, scale=self.noise * y.std(ddof=0), size=len(y))
        # return the data
        return pd.DataFrame({'x': x, 'y': y})

    def f(self, a: np.ndarray) -> np.ndarray:
        return -a ** self.degree_x

    def g(self, b: np.ndarray) -> np.ndarray:
        return b ** self.degree_y


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class NonLinear(Deterministic):
    fn: str = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default='relu')
    """The name of non-linear relationship (one in 'sign', 'relu', 'sin', 'tanh')."""

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, fn=self.fn, noise=self.noise, seed=self.seed, size=self.size)

    @property
    def name(self) -> str:
        return 'nonlinear'

    def _load(self) -> pd.DataFrame:
        rng = np.random.default_rng(seed=self.seed)
        x = np.linspace(-1, 1, num=self.size, endpoint=True)
        y = self.f(x)
        y = y + rng.normal(loc=0.0, scale=self.noise * y.std(ddof=0), size=len(y))
        # return the data
        return pd.DataFrame({'x': x, 'y': y})

    def f(self, a: np.ndarray) -> np.ndarray:
        if self.fn == 'sign':
            return np.sign(a)
        elif self.fn == 'relu':
            return np.maximum(0, a)
        elif self.fn == 'sin':
            # apply the function to the input vector rescaled by a factor of \pi
            return np.sin(pi * a)
        elif self.fn == 'tanh':
            # apply the function to the input vector rescaled by a factor of 10
            return np.tanh(10 * a)
        else:
            raise AssertionError(f"Unknown non-linear function name '{self.fn}'")

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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import pi

import numpy as np

from src.datasets.dataset import Dataset


@dataclass(frozen=True)
class Deterministic(Dataset, ABC):
    size: int = field(kw_only=True, default=101)
    """The size of the dataset."""

    noise_x: float = field(kw_only=True, default=0.0)
    """The amount of noise to be introduced in the protected data."""

    noise_y: float = field(kw_only=True, default=0.0)
    """The amount of noise to be introduced in the target data."""

    linspace: bool = field(kw_only=True, default=True)
    """Whether to build the protected data from a linear space, or sample it uniformly."""

    seed: int = field(kw_only=True, default=0)
    """The random seed for generating the dataset."""

    def __post_init__(self):
        rng = np.random.default_rng(seed=self.seed)
        # take x within the interval [-1, 1] then duplicate it in order to have swapped signs for y if necessary
        s = np.linspace(0, 1, num=self.size, endpoint=True) if self.linspace else rng.uniform(0, 1, size=self.size)
        x = np.array([*s, *s])
        # build y according to the function
        y = self.function(x)
        # add noise in protected and target data
        x += rng.normal(loc=0.0, scale=self.noise_x, size=len(x))
        y += rng.normal(loc=0.0, scale=self.noise_y, size=len(y))
        # standardize input and normalize target, then append to the internal dataframe
        self._data['x'] = (x - x.mean()) / x.std(ddof=0)
        self._data['y'] = (y - y.mean()) / (y.max() - y.min())

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


@dataclass(frozen=True)
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
        return sign * np.power(1 - x ** self.degree_x, 1.0 / self.degree_y)


class NonLinear(Deterministic):
    name: str = field(kw_only=True)
    """The name of non-linear relationship (one in 'sin', 'cos', 'tan', 'log', 'exp')."""

    def function(self, x: np.ndarray) -> np.ndarray:
        ub = 1000.0
        if self.name == 'exp':
            # apply the function to the input vector rescaled from [0, 1] to [-ub, ub]
            return np.exp(2 * ub * x - ub)
        elif self.name == 'log':
            # apply the function to the input vector rescaled from [0, 1] to [0, ub]
            return np.log(ub * x)
        elif self.name in ['sin', 'cos', 'tan']:
            # retrieve the numpy function based on the name
            fn = getattr(np, self.name)
            # apply the function to the input vector rescaled from [0, 1] to [0, 2 * pi]
            return fn(2 * pi * x)
        else:
            raise AssertionError(f"Unknown non-linear function name '{self.name}'")

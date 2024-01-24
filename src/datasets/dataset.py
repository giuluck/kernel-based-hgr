from abc import abstractmethod
from dataclasses import dataclass
from typing import Union, Literal, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.serializable import Serializable

BackendType = Literal['numpy', 'pandas', 'torch']
"""The possible backend types."""

BackendOutput = Union[np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]
"""The output backend types."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Dataset(Serializable):

    @property
    def _data(self) -> pd.DataFrame:
        """Internal data representation."""
        return self._lazy_initialization(attribute='data', function=self._load)

    @abstractmethod
    def _load(self) -> pd.DataFrame:
        """Internal abstract function to load the data."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The dataset name."""
        pass

    @property
    @abstractmethod
    def classification(self) -> bool:
        """Whether this is a classification or a regression task."""
        pass

    @property
    @abstractmethod
    def continuous(self) -> bool:
        """Whether this is a classification or a regression task."""
        pass

    @property
    def input_names(self) -> List[str]:
        return [column for column in self._data.dataframe.columns if column != self.target_name]

    @property
    @abstractmethod
    def target_name(self) -> str:
        """The name of the target feature."""
        pass

    @property
    @abstractmethod
    def excluded_name(self) -> str:
        """The name of the excluded feature."""
        pass

    @property
    def excluded_index(self) -> int:
        """The index of the excluded feature within the input matrix."""
        return self.input_names.index(self.excluded_name)

    @property
    def data(self) -> pd.DataFrame:
        """The dataset data."""
        return self._data.copy()

    def input(self, backend: BackendType = 'pandas') -> BackendOutput:
        """The input features matrix."""
        return Dataset._to_backend(v=self._data.drop(columns=self.target_name), backend=backend)

    def target(self, backend: BackendType = 'pandas') -> BackendOutput:
        """The output target vector."""
        return Dataset._to_backend(v=self._data[self.target_name], backend=backend)

    def excluded(self, backend: BackendType = 'numpy') -> BackendOutput:
        """The protected feature vector."""
        return Dataset._to_backend(v=self._data[self.excluded_name], backend=backend)

    @abstractmethod
    def plot(self, ax: plt.Axes, **kwargs):
        """Plots the excluded and the target feature in the given ax with the given arguments."""
        pass

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _to_backend(v: Union[pd.Series, pd.DataFrame], backend: BackendType) -> BackendOutput:
        if backend == 'pandas':
            return v
        elif backend == 'numpy':
            return v.values
        elif backend == 'torch':
            return torch.tensor(v.values, dtype=torch.float32)
        else:
            raise AssertionError(f"Unknown backend '{backend}'")

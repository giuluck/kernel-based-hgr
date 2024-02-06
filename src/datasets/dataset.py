from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Union, Literal, List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

from src.serializable import Cacheable

BackendType = Literal['numpy', 'pandas', 'torch']
"""The possible backend types."""

BackendOutput = Union[np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]
"""The output backend types."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Dataset(Cacheable):
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
    def units(self) -> List[int]:
        """The number of hidden units in the neural model trained on the dataset."""
        pass

    @property
    def input_names(self) -> List[str]:
        return [column for column in self._data.columns if column != self.target_name]

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

    def data(self, folds: int, seed: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Returns a list of tuples <train, val> (if folds == 1, splits between 70% train and 30% test)."""
        data = self._data
        if folds == 1:
            stratify = data[self.target_name] if self.classification else None
            idx = [train_test_split(data.index, test_size=0.3, stratify=stratify, random_state=seed)]
        else:
            kf = StratifiedKFold if self.classification else KFold
            idx = kf(n_splits=folds, shuffle=True, random_state=seed).split(X=data.index, y=data[self.target_name])
        return [(data.iloc[tr], data.iloc[ts]) for tr, ts in idx]

    def input(self, backend: BackendType = 'numpy') -> BackendOutput:
        """The input features matrix."""
        return Dataset._to_backend(v=self._data.drop(columns=self.target_name), backend=backend)

    def target(self, backend: BackendType = 'numpy') -> BackendOutput:
        """The output target vector."""
        return Dataset._to_backend(v=self._data[self.target_name], backend=backend)

    def excluded(self, backend: BackendType = 'numpy') -> BackendOutput:
        """The protected feature vector."""
        return Dataset._to_backend(v=self._data[self.excluded_name], backend=backend)

    def plot(self, ax: plt.Axes, **kwargs):
        """Plots the excluded and the target feature in the given ax with the given arguments."""
        ax.scatter(self.excluded(backend='numpy'), self.target(backend='numpy'), **kwargs)

    def __len__(self) -> int:
        return len(self._data)

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


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class RealDataset(Dataset):
    features: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=10)
    """The number of features to select using KBest feature selector."""

    @abstractmethod
    def _from_csv(self) -> pd.DataFrame:
        """Loads all the data from a csv file."""
        pass

    @property
    @abstractmethod
    def surrogate_name(self) -> str:
        """The name of the discrete surrogate."""
        pass

    @property
    def surrogate_index(self) -> int:
        """The index of the discrete surrogate within the input matrix."""
        return self.input_names.index(self.surrogate_index)

    def surrogate(self, backend: BackendType = 'numpy') -> BackendOutput:
        """The discrete surrogate of the excluded feature."""
        return Dataset._to_backend(v=self._data[self.surrogate_name], backend=backend)

    def _load(self) -> pd.DataFrame:
        data = self._from_csv()
        # if features is -1 do not perform selection
        if self.features != -1:
            x = data.drop(columns=[self.excluded_name, self.surrogate_name, self.target_name])
            y = data[self.target_name]
            k_best = SelectKBest(k=self.features)
            k_best.fit(x, y)
            features = [*k_best.get_feature_names_out(), self.excluded_name, self.surrogate_name, self.target_name]
            data = data[features]
        return data

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, features=self.features)

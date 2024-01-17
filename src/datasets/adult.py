import importlib.resources
from dataclasses import dataclass, field

import pandas as pd

from src.datasets.dataset import Dataset


@dataclass(frozen=True)
class Adult(Dataset):
    continuous: bool = field(kw_only=True)

    def __post_init__(self):
        with importlib.resources.path('data', 'adult.csv') as filepath:
            data = pd.read_csv(filepath).astype(float)
        # standardize the numerical features while keep the other as they are (categorical binary)
        numerical_features = {'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'}
        for column, values in data.items():
            if column in numerical_features:
                self._data[column] = (values - values.mean()) / values.std(ddof=0)
            else:
                self._data[column] = values

    @property
    def name(self) -> str:
        return f"adult {'continuous' if self.continuous else 'categorical'}"

    @property
    def classification(self) -> bool:
        return True

    @property
    def excluded_name(self) -> str:
        return 'age' if self.continuous else 'sex'

    @property
    def target_name(self) -> str:
        return 'income'

import importlib.resources
from dataclasses import dataclass, field

import pandas as pd

from src.datasets.dataset import Dataset


@dataclass(frozen=True, kw_only=True)
class Adult(Dataset):
    continuous: bool = field(kw_only=True, default=True)

    def _load(self) -> pd.DataFrame:
        with importlib.resources.path('data', 'adult.csv') as filepath:
            data = pd.read_csv(filepath).astype(float)
        # standardize the numerical features while keep the other as they are (categorical binary)
        for column in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
            values = data[column]
            data[column] = (values - values.mean()) / values.std(ddof=0)
        return data

    @property
    def name(self) -> str:
        return f"adult-{'continuous' if self.continuous else 'categorical'}"

    @property
    def classification(self) -> bool:
        return True

    @property
    def excluded_name(self) -> str:
        return 'age' if self.continuous else 'sex'

    @property
    def target_name(self) -> str:
        return 'income'

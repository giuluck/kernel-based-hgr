import importlib.resources
from dataclasses import dataclass, field

import pandas as pd

from src.datasets.dataset import Dataset


@dataclass(frozen=True)
class Communities(Dataset):
    continuous: bool = field(kw_only=True)

    def __post_init__(self):
        with importlib.resources.path('data', 'communities.csv') as filepath:
            data = pd.read_csv(filepath)
        # standardize all features but race (already binary) and target (normalized)
        for column, values in data.items():
            if column == 'race':
                self._data[column] = values
            elif column == self.target_name:
                self._data[column] = (values - values.mean()) / (values.max() - values.min())
            else:
                self._data[column] = (values - values.mean()) / values.std(ddof=0)

    @property
    def name(self) -> str:
        return f"communities {'continuous' if self.continuous else 'categorical'}"

    @property
    def classification(self) -> bool:
        return False

    @property
    def excluded_name(self) -> str:
        return 'pctBlack' if self.continuous else 'race'

    @property
    def target_name(self) -> str:
        return 'violentPerPop'

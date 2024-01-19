import importlib.resources
from dataclasses import dataclass, field

import pandas as pd

from src.datasets.dataset import Dataset


@dataclass(frozen=True, kw_only=True)
class Communities(Dataset):
    continuous: bool = field(kw_only=True, default=True)

    def _load(self) -> pd.DataFrame:
        with importlib.resources.path('data', 'communities.csv') as filepath:
            data = pd.read_csv(filepath)
        # standardize all features but race (already binary) and target (target to normalize)
        for column, values in data.items():
            if column == 'violentPerPop':
                data[column] = (values - values.mean()) / (values.max() - values.min())
            elif column != 'race':
                data[column] = (values - values.min()) / values.std(ddof=0)
        return data

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

import importlib.resources
from dataclasses import dataclass, field
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd

from src.datasets.dataset import Dataset


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Communities(Dataset):
    continuous: bool = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=True)

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
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, excluded='continuous' if self.continuous else 'categorical')

    @property
    def name(self) -> str:
        return 'communities'

    @property
    def classification(self) -> bool:
        return False

    @property
    def excluded_name(self) -> str:
        return 'pctBlack' if self.continuous else 'race'

    @property
    def target_name(self) -> str:
        return 'violentPerPop'

    def plot(self, ax: plt.Axes, **kwargs):
        ax.scatter(self.excluded(backend='numpy'), self.excluded(backend='numpy'), **kwargs)

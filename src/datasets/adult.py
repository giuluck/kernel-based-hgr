import importlib.resources
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import pandas as pd

from src.datasets.dataset import Dataset


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Adult(Dataset):
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
        return 'adult'

    @property
    def classification(self) -> bool:
        return True

    @property
    def units(self) -> List[int]:
        return [32, 32]

    @property
    def excluded_name(self) -> str:
        return 'age'

    @property
    def surrogate_name(self) -> Optional[str]:
        return 'marital-status_Never-married'

    @property
    def target_name(self) -> str:
        return 'income'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name)

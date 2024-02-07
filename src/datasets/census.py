import importlib.resources
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd

from src.datasets.dataset import Dataset


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Census(Dataset):

    def _load(self) -> pd.DataFrame:
        with importlib.resources.path('data', 'census.csv') as filepath:
            data = pd.read_csv(filepath)
        data = data.drop(columns=['CensusTract', 'County', 'Poverty', 'Men']).dropna()
        for column, values in data.items():
            # standardize all features but state (categorical) and ChildPoverty (target to normalize)
            if column == 'ChildPoverty':
                data[column] = (values - values.min()) / (values.max() - values.min())
            elif column != 'State':
                data[column] = (values - values.mean()) / values.std(ddof=0)
        return data.reset_index(drop=True).pipe(pd.get_dummies).astype(float)

    @property
    def name(self) -> str:
        return 'census'

    @property
    def classification(self) -> bool:
        return False

    @property
    def units(self) -> List[int]:
        raise NotImplementedError()

    @property
    def excluded_name(self) -> str:
        return 'Income'

    @property
    def target_name(self) -> str:
        return 'ChildPoverty'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name)

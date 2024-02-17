import importlib.resources
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.datasets.dataset import Dataset


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Students(Dataset):
    def _load(self) -> pd.DataFrame:
        surrogates = ['mother_occupation', 'father_occupation', 'mother_education', 'father_education', 'books']
        with importlib.resources.path('data', 'students.csv') as filepath:
            if filepath.exists():
                data = pd.read_csv(filepath)
            else:
                raise ImportError("The students dataset is private. If you have access to it, "
                                  "please import it in the 'experiments.results' package in order to load it.")
        # cache surrogate attributes for evaluation
        self._cache['surrogates'] = data[surrogates]
        # split train (year == 2016) and test (year != 2016)
        self._cache['train_mask'] = np.array(data['year'] == 2016)
        # standardize ESCS (continuous), normalize scoreMAT (output), and process from start_schooling_age (multi-class)
        data = data.drop(columns=surrogates + ['year'])
        for column, values in data.items():
            if column == 'ESCS':
                data[column] = (values - values.mean()) / values.std(ddof=0)
            elif column == 'scoreMAT':
                data[column] = (values - values.min()) / (values.max() - values.min())
            elif column == 'start_schooling_age':
                data[column] = values.astype('category')
        return data.pipe(pd.get_dummies).astype(float)

    @property
    def _mask(self) -> pd.Series:
        return self._cache['train_mask']

    def data(self, folds: int, seed: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        # if cross-validation, take folds from test data only
        data = self._data
        mask = self._mask
        train, test = data[mask], data[~mask]
        if folds == 1:
            return [(train.index, test.index)]
        else:
            kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
            idx = kf.split(X=train.index, y=train[self.target_name])
            return [(train.iloc[tr], train.iloc[ts]) for tr, ts in idx]

    @property
    def name(self) -> str:
        return 'students'

    @property
    def classification(self) -> bool:
        return False

    @property
    def units(self) -> List[int]:
        raise NotImplementedError()

    @property
    def threshold(self) -> float:
        return 0.2

    @property
    def excluded_name(self) -> str:
        return 'ESCS'

    @property
    def target_name(self) -> str:
        return 'scoreMAT'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name)

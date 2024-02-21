import importlib.resources
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.datasets.dataset import SurrogateDataset

COLUMNS: Dict[str, str] = dict(
    year='year',
    ESCS='ESCS',
    books='books',
    mother_occupation='mother_occupation',
    father_occupation='father_occupation',
    mother_education='mother_education',
    father_education='father_education',
    a160k='english_perceived_good',
    a162k='english_difficult_students',
    a163k='english_difficult_subjects',
    score_ING='english_score'
)
"""Data columns used during the preprocessing step."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Students(SurrogateDataset):
    def _load(self) -> pd.DataFrame:
        surrogates = dict(
            books='N. Books',
            mother_education='Mother Ed.',
            father_education='Father Ed.',
            mother_occupation='Mother Occ.',
            father_occupation='Father Occ.'
        )
        with importlib.resources.path('data', 'students.csv') as filepath:
            if filepath.exists():
                data = pd.read_csv(filepath)
            else:
                raise ImportError("The students dataset is private. If you have access to it, "
                                  "please import it in the 'data' package in order to load it.")
        # cache surrogate attributes for evaluation
        # self._cache['surrogates'] = data[list(surrogates)].rename(columns=surrogates)
        self._cache['surrogates'] = data[list(surrogates)].rename(columns=surrogates)
        # split train (year == 2018) and test (year != 2018)
        self._cache['train_mask'] = np.array(data['year'] == 2018)
        # standardize ESCS (continuous), normalize scoreING (output), and convert multi-class features
        data = data.drop(columns='year')
        for column, values in data.items():
            if column == 'ESCS':
                data[column] = (values - values.mean()) / values.std(ddof=0)
            elif column == 'english_score':
                data[column] = (values - values.min()) / (values.max() - values.min())
            else:
                data[column] = values.astype('category')
        return data.pipe(pd.get_dummies).astype(float)

    @property
    def _mask(self) -> pd.Series:
        return self._cache['train_mask']

    def data(self, folds: int, seed: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        # if cross-validation, take folds from test data only
        data = self._data
        mask = self._mask
        train = data[mask]
        if folds == 1:
            return [(train, data[~mask])]
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
        return [16, 16]

    @property
    def batch(self) -> int:
        return -1

    @property
    def threshold(self) -> float:
        return 0.2

    @property
    def excluded_name(self) -> str:
        return 'ESCS'

    @property
    def surrogate_name(self) -> str:
        return 'books_4.0'

    @property
    def target_name(self) -> str:
        return 'english_score'

    @property
    def surrogates(self) -> pd.DataFrame:
        return self._cache['surrogates'].copy()

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name)

import os.path
import pickle
from argparse import Namespace
from typing import Union, Dict, Any, Optional, List, Set

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger
from tqdm import tqdm


class InternalLogger(Logger):
    def __init__(self):
        self._results: List[Dict[str, float]] = []

    @property
    def results(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in pd.DataFrame(self._results).items()}

    @property
    def name(self) -> Optional[str]:
        return 'internal_logger'

    @property
    def version(self) -> Optional[Union[int, str]]:
        return 0

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if len(self._results) == step:
            self._results.append(metrics)
        else:
            self._results[step].update(metrics)

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any):
        pass


class History(pl.Callback):
    def __init__(self, key: str, folder: str):
        """
        :param key:
            The key of the experiment.

        :param folder:
            The folder where to store the results.
        """
        self._key: str = key
        self._folder: str = folder
        self._external: Set[str] = set()

    @property
    def subfolder(self) -> str:
        return os.path.join('learning', self._key)

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, Any],
                           batch: Any,
                           batch_idx: int):
        results = dict(
            train_predictions=pl_module(trainer.train_dataloader.dataset.x),
            val_predictions=pl_module(trainer.val_dataloaders.dataset.x),
            model_state=pl_module.state_dict()
        )
        name = f'{self._key}_step-{trainer.global_step - 1}.pkl'
        filepath = os.path.join(self._folder, 'results', self.subfolder, name)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as file:
            pickle.dump(results, file=file)
        self._external.add(name)

    def get(self, item: int, folder: str) -> Dict[str, Any]:
        name = f'{self._key}_step-{item}.pkl'
        assert name in self._external, f"External results for experiment {self._key} are not available at step {item}"
        with open(os.path.join(folder, 'results', self.subfolder, name), 'rb') as file:
            return pickle.load(file=file)


class Progress(pl.Callback):
    def __init__(self):
        self._pbar: Optional[tqdm] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._pbar = tqdm(total=trainer.max_steps, desc='Model Training', unit='step')

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, Any],
                           batch: Any,
                           batch_idx: int):
        desc = 'Model Training (' + ' - '.join([f'{k}: {v:.4f}' for k, v in trainer.logged_metrics.items()]) + ')'
        self._pbar.set_description(desc=desc, refresh=True)
        self._pbar.update(n=1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._pbar.close()
        self._pbar = None

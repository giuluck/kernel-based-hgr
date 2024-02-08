import importlib.resources
import os.path
import pickle
from argparse import Namespace
from typing import Union, Dict, Any, Optional, List, Set

import pandas as pd
import pytorch_lightning as pl
import torch
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
        epoch = metrics.pop('epoch')
        if len(self._results) == epoch:
            self._results.append(metrics)
        else:
            self._results[epoch].update(metrics)

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any):
        pass


class History(pl.Callback):
    def __init__(self, key: str):
        """
        :param key:
            The key of the experiment.
        """
        self._key: str = key
        self._external: Set[str] = set()

    @property
    def folder(self) -> str:
        return os.path.join('learning', self._key)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        results = dict(
            train_predictions=pl_module(trainer.train_dataloader.dataset.x),
            val_predictions=pl_module(trainer.val_dataloaders.dataset.x),
            model_state=pl_module.state_dict()
        )
        with importlib.resources.files('experiments.results') as folder:
            name = f'{self._key}_epoch-{trainer.current_epoch}.pkl'
            filepath = os.path.join(folder, self.folder, name)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as file:
                pickle.dump(results, file=file)
        self._external.add(name)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        name = f'{self._key}_epoch-{item}.pkl'
        assert name in self._external, f"External results for experiment {self._key} are not available at epoch {item}"
        with importlib.resources.files('experiments.results') as folder:
            with open(os.path.join(folder, self.folder, name), 'rb') as file:
                return pickle.load(file=file)


class Progress(pl.Callback):
    def __init__(self):
        self._pbar: Optional[tqdm] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._pbar = tqdm(total=trainer.max_epochs * trainer.num_training_batches, desc='Model Training', unit='step')

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, torch.Tensor],
                           batch: Any,
                           batch_idx: int) -> None:
        self._pbar.update(n=1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._pbar.close()
        self._pbar = None

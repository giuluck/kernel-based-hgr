import time
from typing import List, Dict, Any, Optional

import pytorch_lightning as pl
import torch
import wandb

PROJECT: str = 'kernel-based-hgr'
"""The name of the wandb project."""


class Callback(pl.Callback):
    def __init__(self, experiment: Any):
        """
        :param experiment:
            The experiment instance.
        """
        self._experiment: Any = experiment
        self._time: Optional[float] = None
        self._cache: List[Dict[str, Any]] = []
        self._results: List[Dict[str, Any]] = []

    @property
    def results(self) -> Dict[str, list]:
        keys = [] if len(self._results) == 0 else self._results[0].keys()
        results = {key: [] for key in keys}
        for res in self._results:
            for key in keys:
                results[key].append(res[key])
        return results

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # start wandb if necessary
        if self._experiment.entity is not None:
            wandb.init(
                project=PROJECT,
                entity=self._experiment.entity,
                name=self._experiment.key,
                config=self._experiment.configuration
            )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._time = time.time()

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, torch.Tensor],
                           batch: Any,
                           batch_idx: int):
        # append batch output on cache
        self._cache.append(outputs)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # compute mean of cached outputs
        keys = [] if len(self._cache) == 0 else self._cache[0].keys()
        cache = {key: 0.0 for key in keys}
        for output in self._cache:
            for key in keys:
                cache[key] += output[key]
        # build results dictionary and store it
        results = {
            **{key: value / len(self._cache) for key, value in cache.items()},
            'time': time.time() - self._time,
            'model': pl_module
        }
        self._results.append(results)
        if self._experiment.entity is not None:
            wandb.log(results)
        # empty cache and time
        self._time = None
        self._cache = []

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # close wandb if necessary
        if self._experiment.entity:
            wandb.finish()

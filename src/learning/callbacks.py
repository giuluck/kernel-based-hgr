import copy
import time
from typing import List, Dict, Any, Optional

import pytorch_lightning as pl
import torch
import wandb


class Callback(pl.Callback):
    PROJECT: str = 'kernel-based-hgr'
    """The name of the wandb project."""

    def __init__(self, experiment: Any):
        """
        :param experiment:
            The experiment instance.
        """
        self._experiment: Any = experiment
        self._time: Optional[float] = None
        self._epoch_cache: List[Dict[str, Any]] = []
        self._train_cache: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # include general information to the results
        self.results['train_inputs'] = trainer.train_dataloader.dataset.x
        self.results['train_target'] = trainer.train_dataloader.dataset.y
        self.results['val_inputs'] = trainer.val_dataloaders.dataset.x
        self.results['val_target'] = trainer.val_dataloaders.dataset.y
        self.results['epochs'] = trainer.max_epochs
        # start and log on wandb if necessary
        if self._experiment.entity is not None:
            wandb.init(
                project=Callback.PROJECT,
                entity=self._experiment.entity,
                name=self._experiment.key,
                config=self._experiment.configuration
            )
            wandb.log(self.results)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._time = time.time()

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, torch.Tensor],
                           batch: Any,
                           batch_idx: int):
        # append batch output on cache
        self._epoch_cache.append(outputs)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # compute mean of cached outputs
        keys = [] if len(self._epoch_cache) == 0 else self._epoch_cache[0].keys()
        cache = {key: 0.0 for key in keys}
        for output in self._epoch_cache:
            for key in keys:
                cache[key] += output[key]
        # build results dictionary and store it
        results = {
            **{key: value / len(self._epoch_cache) for key, value in cache.items()},
            'time': time.time() - self._time,
            'model': copy.deepcopy(pl_module),
            'train_predictions': pl_module(self.results['train_inputs']),
            'val_predictions': pl_module(self.results['val_inputs'])
        }
        self._train_cache.append(results)
        if self._experiment.entity is not None:
            wandb.log(results)
        # empty cache and time
        self._time = None
        self._epoch_cache = []

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # close wandb if necessary
        if self._experiment.entity:
            wandb.finish()
        # build results
        keys = [] if len(self._train_cache) == 0 else self._train_cache[0].keys()
        results = {key: [] for key in keys}
        for res in self._train_cache:
            for key in keys:
                results[key].append(res[key])
        self.results.update(results)
        self._train_cache = []

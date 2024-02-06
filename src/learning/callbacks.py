import time
from typing import List, Dict, Any, Optional

import pandas as pd
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
        self.results: Dict[str, Any] = {}

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # start and log on wandb if necessary
        if self._experiment.entity is not None:
            wandb.init(
                project=Callback.PROJECT,
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
        self._epoch_cache.append({key: value.numpy(force=True).item() for key, value in outputs.items()})

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # compute mean of cached outputs and use it to build a dictionary with other results
        logs = {
            **pd.DataFrame(self._epoch_cache).mean().to_dict(),
            'train_predictions': pl_module(trainer.train_dataloader.dataset.x),
            'val_predictions': pl_module(trainer.val_dataloaders.dataset.x),
            'time': time.time() - self._time
        }
        for key, value in logs.items():
            result_list = self.results.get(key, [])
            result_list.append(value)
            self.results[key] = result_list
        # log if necessary, then empty caches
        if self._experiment.entity is not None:
            wandb.log(logs)
        self._epoch_cache = []
        self._time = None

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # include general information to the results
        logs = dict(
            train_inputs=trainer.train_dataloader.dataset.x,
            train_target=trainer.train_dataloader.dataset.y,
            val_inputs=trainer.val_dataloaders.dataset.x,
            val_target=trainer.val_dataloaders.dataset.y,
            epochs=trainer.max_epochs,
            model=pl_module
        )
        self.results.update(logs)
        # store model and close wandb if necessary
        if self._experiment.entity:
            wandb.log(logs)
            wandb.finish()

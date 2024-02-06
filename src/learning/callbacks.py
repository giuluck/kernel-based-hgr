import sys
import time
from typing import List, Dict, Any, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar


class ResultsCallback(pl.Callback):
    def __init__(self):
        self._time: Optional[float] = None
        self._epoch_cache: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._time = time.time()

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, torch.Tensor],
                           batch: Any,
                           batch_idx: int):
        self._epoch_cache.append({key: value.numpy(force=True).item() for key, value in outputs.items()})

    def on_validation_batch_end(self,
                                trainer: pl.Trainer,
                                pl_module: pl.LightningModule,
                                outputs: Dict[str, torch.Tensor],
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0):
        self._epoch_cache.append({key: value.numpy(force=True).item() for key, value in outputs.items()})

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log(train_predictions=pl_module(trainer.train_dataloader.dataset.x), time=time.time() - self._time)
        self._time = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log(val_predictions=pl_module(trainer.val_dataloaders.dataset.x))

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # include general information to the results
        self.results['train_inputs'] = trainer.train_dataloader.dataset.x
        self.results['train_target'] = trainer.train_dataloader.dataset.y
        self.results['val_inputs'] = trainer.val_dataloaders.dataset.x
        self.results['val_target'] = trainer.val_dataloaders.dataset.y
        self.results['epochs'] = trainer.max_epochs
        self.results['model'] = pl_module

    def _log(self, **kwargs):
        # compute mean of cached outputs and use it to build a dictionary with other results
        logs = {**pd.DataFrame(self._epoch_cache).mean().to_dict(), **kwargs}
        for key, value in logs.items():
            result_list = self.results.get(key, [])
            result_list.append(value)
            self.results[key] = result_list
        self._epoch_cache = []


class ProgressCallback(TQDMProgressBar):
    # patch to have a better integration of the progress bar when running from pycharm
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

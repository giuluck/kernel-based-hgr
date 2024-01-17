from typing import Optional, List, Dict, Tuple

import lightning.pytorch as pl
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from torch import Tensor


class Callback(pl.Callback):
    def __init__(self, verbose: bool = True):
        self.verbose: bool = verbose
        self.history: Optional[pd.DataFrame] = None
        self._history: List[dict] = []

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           module: pl.LightningModule,
                           outputs: Dict[str, Tensor],
                           batch: Tuple[Tensor, Tensor],
                           batch_idx: int):
        inp, out = batch
        pred = outputs.pop('pred')
        outputs = {k: v.numpy(force=True).item() for k, v in outputs.items()}
        outputs = {
            'batch': batch_idx,
            'epoch': trainer.current_epoch + 1,
            'r2': r2_score(y_true=out.numpy(force=True), y_pred=pred.numpy(force=True)),
            'mse': mean_squared_error(y_true=out.numpy(force=True), y_pred=pred.numpy(force=True)),
            **outputs
        }
        if self.verbose:
            for k, v in outputs.items():
                print(k, '-->', v)
            print()
        self._history.append(outputs)

    def on_train_end(self, trainer, pl_module):
        self.history = pd.DataFrame(self._history)

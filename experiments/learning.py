import importlib.resources
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from experiments.experiment import Experiment
from src.datasets import Dataset
from src.hgr import HGR
from src.learning import MultiLayerPerceptron, Data, Callback

SEED: int = 0
"""The random seed used in the experiment."""

FOLDS: int = 5
"""The number of folds for k-fold cross-validation."""

MINI_BATCH: int = 128
"""The size of a mini batch."""

MINI_EPOCHS: int = 500
"""The number of epochs used during training with mini batches."""

FULL_EPOCHS: int = 200
"""The number of epochs used during training full batch."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class LearningExperiment(Experiment):
    """An experiment where a neural network is constrained so that the correlation between a protected variable and the
    target is reduced."""

    dataset: Dataset = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The dataset used in the experiment."""

    fold: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The fold that is used for training the model."""

    penalty: Optional[HGR] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The HGR metric used as penalizer."""

    _alpha: Optional[float] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The alpha value for the penalizer."""

    _warm_start: bool = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """Whether to start from a pretrained non-constrained network."""

    full_batch: bool = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """Whether to train the model full batch or with mini batches."""

    entity: Optional[str] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """Either the Weights & Biases entity, or None for no logging."""

    def _compute(self) -> Experiment.Result:
        pl.seed_everything(SEED, workers=True)
        external = f'{self.key}.pkl'
        # retrieve train and validation data from splits and set parameters
        trn, val = self.dataset.data(folds=FOLDS, seed=SEED)[self.fold]
        trn_data = Data(x=trn[self.dataset.input_names], y=trn[self.dataset.target_name])
        val_data = Data(x=val[self.dataset.input_names], y=val[self.dataset.target_name])
        epochs = FULL_EPOCHS if self.full_batch else MINI_EPOCHS
        batch = len(trn) if self.full_batch else MINI_BATCH
        # build model
        model = MultiLayerPerceptron(
            units=[len(self.dataset.input_names), *self.dataset.units],
            classification=self.dataset.classification,
            feature=self.dataset.excluded_index,
            penalty=self.penalty,
            alpha=self.alpha
        )
        # if warm start is true, retrieve the pretrained unconstrained model
        if self.warm_start:
            # build the pretrained experiment by considering penalty = None, alpha = None, warm_start = False
            pretrained = LearningExperiment(
                dataset=self.dataset,
                fold=self.fold,
                penalty=None,
                _alpha=None,
                _warm_start=False,
                full_batch=self.full_batch,
                entity=self.entity
            )
            # check if the pretrained experiment was already run and the external file is in the results package
            # if so, load the results, otherwise compute them
            with importlib.resources.path('experiments.results', f'{pretrained.key}.pkl') as path:
                if path.exists():
                    with open(path, 'rb') as file:
                        result = pickle.load(file=file)
                else:
                    result = pretrained.result
            # load the last state of the pretrained model as initial state for this model
            model.load_state_dict(result['model'][epochs - 1].state_dict())
        # build trainer and callback
        callback = Callback(experiment=self)
        trainer = pl.Trainer(
            min_epochs=epochs,
            max_epochs=epochs,
            check_val_every_n_epoch=epochs + 1,
            callbacks=[callback],
            deterministic=True,
            enable_progress_bar=True,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=False
        )
        # run fitting
        start = time.time()
        trainer.fit(
            model=model,
            train_dataloaders=DataLoader(trn_data, batch_size=batch, num_workers=4, persistent_workers=True),
            val_dataloaders=DataLoader(val_data, batch_size=len(val), num_workers=4, persistent_workers=True)
        )
        gap = time.time() - start
        # store external results
        with importlib.resources.path('experiments.results', external) as path:
            assert not path.exists(), f"File '{self.key}' is already present in package 'experiments.results'"
        with open(path, 'wb') as file:
            pickle.dump(callback.results, file=file)
        # return experiments
        return Experiment.Result(timestamp=start, execution=gap, external=external)

    @property
    def alpha(self) -> Optional[float]:
        """The alpha value for the penalizer (defaults to None when no penalty is used)."""
        return None if self.penalty is None else self._alpha

    @property
    def warm_start(self) -> Optional[float]:
        """Whether to start from a pretrained non-constrained network (defaults to False when no penalty is used)."""
        return False if self.penalty is None else self._warm_start

    @property
    def name(self) -> str:
        return 'learning'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(
            experiment=self.name,
            dataset=self.dataset.configuration,
            fold=self.fold,
            penalty=None if self.penalty is None else self.penalty.configuration,
            alpha=self.alpha,
            full_batch=self.full_batch,
            warm_start=self.warm_start
        )

    @property
    def key(self) -> str:
        penalty = None if self.penalty is None else self.penalty.key
        return f'{self.name}_{self.dataset.key}_{self.fold}_{penalty}_{self.alpha}_{self.full_batch}_{self.warm_start}'

    @staticmethod
    def learning(datasets: Dict[str, Dataset],
                 penalties: Dict[str, HGR],
                 alpha: Optional[float] = None,
                 full_batch: bool = True,
                 warm_start: bool = False,
                 entity: Optional[str] = None):
        penalties = {'UNC': None, **penalties}
        LearningExperiment.doe(
            file_name='learning',
            save_time=0,
            verbose=True,
            dataset=datasets,
            penalty=penalties,
            fold=list(range(FOLDS)),
            _alpha=alpha,
            _warm_start=warm_start,
            full_batch=full_batch,
            entity=entity
        )

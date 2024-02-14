import importlib.resources
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Iterable, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.experiment import Experiment
from src.datasets import SurrogateDataset
from src.hgr import HGR
from src.learning import MultiLayerPerceptron, Data, Loss, Accuracy, Metric, InternalLogger, Progress, History, \
    Correlation
from src.learning.metrics import DIDI

PALETTE: List[str] = [
    '#000000',
    '#377eb8',
    '#ff7f00',
    '#4daf4a',
    '#f781bf',
    '#a65628',
    '#984ea3',
    '#999999',
    '#e41a1c',
    '#dede00'
]
"""The color palette for plotting data."""

SEED: int = 0
"""The random seed used in the experiment."""

ALPHA: Optional[float] = None
"""The alpha value used in the experiment."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class LearningExperiment(Experiment):
    """An experiment where a neural network is constrained so that the correlation between a protected variable and the
    target is reduced."""

    dataset: SurrogateDataset = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The dataset used in the experiment."""

    fold: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The fold that is used for training the model."""

    folds: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The number of folds for k-fold cross-validation."""

    metric: Optional[HGR] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The metric to be used as penalty, or None for unconstrained model."""

    hidden: Optional[Iterable[int]] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The number of hidden units used to build the neural model, or None to use the dataset default value."""

    batches: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The number of batches used during training (e.g., 1 for full batch)."""

    steps: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The number of training steps."""

    wandb_project: Optional[str] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The name of the Weights & Biases project for logging, or None for no logging."""

    @property
    def units(self) -> List[int]:
        hidden = self.dataset.hidden if self.hidden is None else self.hidden
        return [len(self.dataset.input_names), *hidden]

    def _compute(self) -> Experiment.Result:
        pl.seed_everything(SEED, workers=True)
        # retrieve train and validation data from splits and set parameters
        trn, val = self.dataset.data(folds=self.folds, seed=SEED)[self.fold]
        trn_data = Data(x=trn[self.dataset.input_names], y=trn[self.dataset.target_name])
        val_data = Data(x=val[self.dataset.input_names], y=val[self.dataset.target_name])
        # build model
        model = MultiLayerPerceptron(
            units=self.units,
            classification=self.dataset.classification,
            feature=self.dataset.excluded_index,
            metric=self.metric,
            alpha=None if self.metric is None else ALPHA
        )
        # build trainer and callback
        progress = Progress()
        logger = InternalLogger()
        history = History(key=self.key)
        if self.wandb_project is not None:
            wandb_logger = WandbLogger(project=self.wandb_project, name=self.key, log_model='all')
            wandb_logger.experiment.config.update(self.configuration)
            loggers = [logger, wandb_logger]
        else:
            loggers = [logger]
        trainer = pl.Trainer(
            deterministic=True,
            min_steps=self.steps,
            max_steps=self.steps,
            logger=loggers,
            callbacks=[history, progress],
            num_sanity_val_steps=0,
            val_check_interval=1,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False
        )
        # run fitting
        batch_size = int(math.ceil(len(trn_data) / self.batches))
        start = time.time()
        trainer.fit(
            model=model,
            train_dataloaders=DataLoader(trn_data, batch_size=batch_size, shuffle=True),
            val_dataloaders=DataLoader(val_data, batch_size=len(val), shuffle=False)
        )
        gap = time.time() - start
        # close wandb in case it was used in the logger
        if self.wandb_project is not None:
            wandb.finish()
        # store external files and return result
        external = os.path.join('learning', f'{self.key}.pkl')
        with importlib.resources.files('experiments.results') as folder:
            filepath = os.path.join(folder, external)
            assert not os.path.exists(filepath), f"Experiment '{self.key}' is already present in 'experiments.results'"
            with open(filepath, 'wb') as file:
                pickle.dump({
                    'train_inputs': trn_data.x,
                    'train_target': trn_data.y,
                    'val_inputs': val_data.x,
                    'val_target': val_data.y
                }, file=file)
        return Experiment.Result(timestamp=start, execution=gap, history=history, external=external, **logger.results)

    @property
    def name(self) -> str:
        return 'learning'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(
            experiment=self.name,
            dataset=self.dataset.configuration,
            metric={'name': 'unc'} if self.metric is None else self.metric.configuration,
            units=self.units,
            steps=self.steps,
            batches=self.batches,
            folds=self.folds,
            fold=self.fold,
        )

    @property
    def key(self) -> str:
        mtr = None if self.metric is None else self.metric.key
        return f'{self.name}_{self.dataset.key}_{mtr}_{self.units}_{self.steps}_{self.batches}_{self.folds}_{self.fold}'

    @staticmethod
    def calibration(datasets: Dict[str, SurrogateDataset],
                    batches: Iterable[int] = (1, 5, 20),
                    hiddens: Iterable[Iterable[int]] = ((32,), (256,), (32,) * 2, (256,) * 2, (32,) * 3, (256,) * 3),
                    steps: int = 1000,
                    folds: int = 3,
                    wandb_project: Optional[str] = None,
                    formats: Iterable[str] = ('png',),
                    plot: bool = False):
        def configuration(ds, bt, hd, fl):
            classification = datasets[ds].classification
            return dict(dataset=ds, batch=bt, hidden=hd, fold=fl), [
                Loss(classification=classification),
                Accuracy(classification=classification)
            ]

        hiddens = [list(h) for h in hiddens]
        experiments = LearningExperiment.doe(
            file_name='learning',
            save_time=0,
            verbose=True,
            dataset=datasets,
            batches={b: b for b in batches},
            hidden={str(h): h for h in hiddens},
            fold=list(range(folds)),
            folds=folds,
            steps=steps,
            metric=None,
            wandb_project=wandb_project
        )
        # get metric results and add time
        results = LearningExperiment._metrics(experiments=experiments, configuration=configuration)
        times = []
        for index, experiment in experiments.items():
            info, _ = configuration(*index)
            times += [{
                **info,
                'kpi': 'Time',
                'split': 'Train',
                'step': step,
                'value': experiment.result['time'][step]
            } for step in range(experiment.steps)]
        results = pd.concat((results, pd.DataFrame(times)))
        # plot results
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        for (dataset, kpi), data in results.groupby(['dataset', 'kpi']):
            cl = len(hiddens)
            rw = len(batches)
            fig, axes = plt.subplots(rw, cl, figsize=(5 * cl, 4 * rw), sharex='all', sharey='all', tight_layout=True)
            # used to index the axes in case either or both hidden units and batches have only one value
            axes = np.array(axes).reshape(rw, cl)
            for i, batch in enumerate(batches):
                for j, hidden in enumerate(hiddens):
                    sns.lineplot(
                        data=data[np.logical_and(data['batch'] == batch, data['hidden'] == str(hidden))],
                        x='step',
                        y='value',
                        hue='split',
                        style='split',
                        estimator='mean',
                        errorbar='sd',
                        linewidth=2,
                        palette=['black'] if kpi == 'Time' else PALETTE[1:3],
                        ax=axes[i, j]
                    )
                    axes[i, j].set_ylabel(kpi)
                    axes[i, j].set_ylim((0, 1 if kpi in ['R2', 'ACC'] else data[data['step'] > 20]['value'].max()))
                    axes[i, j].set_title(f"Batch Size: {'Full' if batch == -1 else batch} - Hidden: {hidden}")
            # store, print, and plot if necessary
            for extension in formats:
                name = f'calibration_{kpi}_{dataset}.{extension}'
                with importlib.resources.path('experiments.exports', name) as file:
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Calibration {kpi} for {dataset.title()}")
                fig.show()
            plt.close(fig)

    @staticmethod
    def history(datasets: Dict[str, SurrogateDataset],
                metrics: Dict[str, Optional[HGR]],
                batches: Iterable[int] = (1, 10),
                steps: int = 600,
                folds: int = 3,
                wandb_project: Optional[str] = None,
                formats: Iterable[str] = ('png',),
                plot: bool = False):
        def configuration(ds, mt, fl):
            d = datasets[ds]
            # include the DIDI on surrogate excluded index only if present
            s = [] if d.surrogate_name is None else [DIDI(
                excluded=d.surrogate_index,
                classification=d.classification,
                name=f'Surrogate DIDI'
            )]
            # return a list of metrics for loss, accuracy, correlation, and optionally surrogate fairness
            return dict(dataset=ds, metric=mt, fold=fl), [
                Loss(classification=d.classification),
                Accuracy(classification=d.classification),
                Correlation(excluded=d.excluded_index, algorithm='sk', name=f'Protected HGR'),
                *s
            ]

        sns.set_context('notebook')
        sns.set_style('whitegrid')
        # iterate over dataset and batches
        for name, dataset in datasets.items():
            for batch in batches:
                # use dictionaries for dataset to retrieve correct configuration
                experiments = LearningExperiment.doe(
                    file_name='learning',
                    save_time=0,
                    verbose=True,
                    dataset={name: dataset},
                    metric=metrics,
                    fold=list(range(folds)),
                    folds=folds,
                    hidden=None,
                    steps=steps,
                    batches=batch,
                    wandb_project=wandb_project
                )
                # get and plot metric results
                group = LearningExperiment._metrics(experiments=experiments, configuration=configuration)
                kpis = group['kpi'].unique()
                col = len(kpis) + 1
                fig, axes = plt.subplots(2, col, figsize=(5 * col, 8), sharex='all', sharey=None, tight_layout=True)
                for i, sp in enumerate(['Train', 'Val']):
                    for j, kpi in enumerate(kpis):
                        sns.lineplot(
                            data=group[np.logical_and(group['split'] == sp, group['kpi'] == kpi)],
                            x='step',
                            y='value',
                            estimator='mean',
                            errorbar='sd',
                            linewidth=2,
                            hue='metric',
                            style='metric',
                            palette=PALETTE[:len(metrics)],
                            ax=axes[i, j]
                        )
                        axes[i, j].set_title(f"{kpi} ({sp.lower()})")
                        if i == 1:
                            ub = axes[1, j].get_ylim()[1] if kpi == 'MSE' or kpi == 'BCE' or 'DIDI' in kpi else 1
                            axes[0, j].set_ylim((0, ub))
                            axes[1, j].set_ylim((0, ub))
                # get and plot history results (alpha and time)
                history = []
                to_remove = []
                for (_, mtr, fld), experiment in experiments.items():
                    result = experiment.result
                    times = result['time']
                    if experiment.metric is None:
                        alphas = [np.nan] * len(result['alpha'])
                        to_remove.append(mtr)
                    else:
                        alphas = result['alpha']
                    history.extend([{
                        'metric': mtr,
                        'fold': fld,
                        'step': step,
                        'Training Time (s)': times[step],
                        'Lambda Weight': alphas[step]
                    } for step in range(experiment.steps)])
                history = pd.DataFrame(history)
                for i, col in enumerate(['Training Time (s)', 'Lambda Weight']):
                    sns.lineplot(
                        data=history,
                        x='step',
                        y=col,
                        estimator='mean',
                        errorbar='sd',
                        linewidth=2,
                        hue='metric',
                        style='metric',
                        palette=PALETTE[:len(metrics)],
                        ax=axes[i, -1]
                    )
                    axes[i, -1].set_title(col)
                    axes[i, -1].set_ylabel(None)
                # QUICK PATCH TO REMOVE UNCONSTRAINED EXPERIMENTS FROM ALPHA LEGEND
                leg = axes[1, -1].legend()
                for handle, text in zip(leg.legendHandles, leg.texts):
                    # noinspection PyProtectedMember
                    if handle._label in to_remove and text._text in to_remove:
                        handle.set_visible(False)
                        text.set_visible(False)
                leg.set_title('metric')
                # store, print, and plot if necessary
                for extension in formats:
                    filename = f'history_{name}_{batch}.{extension}'
                    with importlib.resources.path('experiments.exports', filename) as file:
                        fig.savefig(file, bbox_inches='tight')
                if plot:
                    fig.suptitle(f"Learning History for {name.title()} (batches={batch})")
                    fig.show()
                plt.close(fig)

    @staticmethod
    def _metrics(experiments: Dict[Any, 'LearningExperiment'],
                 configuration: Callable[[tuple], Tuple[Dict[str, Any], Iterable[Metric]]]) -> pd.DataFrame:
        results = []
        for index, experiment in experiments.items():
            with importlib.resources.files('experiments.results') as folder:
                with open(os.path.join(folder, experiment.result.external), 'rb') as file:
                    ext = pickle.load(file=file)
            # retrieve input data
            xtr = ext['train_inputs'].numpy(force=True)
            ytr = ext['train_target'].numpy(force=True).flatten()
            xvl = ext['val_inputs'].numpy(force=True)
            yvl = ext['val_target'].numpy(force=True).flatten()
            # compute metrics for each step
            info, metrics = configuration(*index)
            outputs = {
                **{f'train_{mtr.name}': ext.get(f'train_{mtr.name}', []) for mtr in metrics},
                **{f'val_{mtr.name}': ext.get(f'val_{mtr.name}', []) for mtr in metrics},
            }
            # if the metrics are already pre-computed, load them
            if np.all([len(v) == experiment.steps for v in outputs.values()]):
                print(f'Fetching Metrics for {experiment.key}')
                df = pd.DataFrame(outputs).melt()
                df['split'] = df['variable'].map(lambda v: v.split('_')[0].title())
                df['kpi'] = df['variable'].map(lambda v: v.split('_')[1])
                df['step'] = list(range(experiment.steps)) * len(outputs)
                for key, value in info.items():
                    df[key] = value
                df = df.drop(columns='variable').to_dict(orient='records')
                results.extend(df)
            # otherwise, compute and re-serialize them
            else:
                if not np.all([len(v) == 0 for v in outputs.values()]):
                    print(f"WARNING: recomputing metrics for {experiment.key} due to possible serialization errors")
                for step in tqdm(range(experiment.steps), desc=f'Computing Metrics for {experiment.key}'):
                    info['step'] = step
                    hst = experiment.result['history'][step]
                    ptr = hst['train_predictions'].numpy(force=True).flatten()
                    pvl = hst['val_predictions'].numpy(force=True).flatten()
                    for mtr in metrics:
                        for split, (x, y, p) in zip(['train', 'val'], [(xtr, ytr, ptr), (xvl, yvl, pvl)]):
                            value = mtr(x=x, y=y, p=p)
                            outputs[f'{split}_{mtr.name}'].append(value)
                            results.append({**info, 'kpi': mtr.name, 'split': split.title(), 'value': value})
                ext.update(outputs)
                with importlib.resources.files('experiments.results') as folder:
                    filepath = os.path.join(folder, experiment.result.external)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    with open(filepath, 'wb') as file:
                        pickle.dump(ext, file=file)
        return pd.DataFrame(results)

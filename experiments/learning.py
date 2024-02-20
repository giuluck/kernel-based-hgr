import importlib.resources
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
from src.datasets import SurrogateDataset, Students
from src.hgr import HGR, SingleKernelHGR
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

    _units: Optional[Iterable[int]] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The number of hidden units used to build the neural model, or None to use the dataset default value."""

    _batch: Optional[int] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The batch size used for training (-1 for full batch), or None to use the dataset default value."""

    _alpha: Optional[float] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The alpha value used in the experiment."""

    _threshold: Optional[float] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The penalty threshold used during training, or None to use the dataset default value."""

    steps: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The number of training steps."""

    wandb_project: Optional[str] = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The name of the Weights & Biases project for logging, or None for no logging."""

    @property
    def units(self) -> Iterable[int]:
        return self.dataset.units if self._units is None else self._units

    @property
    def batch(self) -> int:
        return self.dataset.batch if self._batch is None else self._batch

    @property
    def alpha(self) -> Optional[float]:
        return None if self.metric is None else self._alpha

    @property
    def threshold(self) -> float:
        if self.metric is None:
            return 0.0
        elif self._threshold is None:
            return self.dataset.threshold
        else:
            return self._threshold

    def _compute(self) -> Experiment.Result:
        pl.seed_everything(SEED, workers=True)
        # retrieve train and validation data from splits and set parameters
        trn, val = self.dataset.data(folds=self.folds, seed=SEED)[self.fold]
        trn_data = Data(x=trn[self.dataset.input_names], y=trn[self.dataset.target_name])
        val_data = Data(x=val[self.dataset.input_names], y=val[self.dataset.target_name])
        # build model
        model = MultiLayerPerceptron(
            units=[len(self.dataset.input_names), *self.units],
            classification=self.dataset.classification,
            feature=self.dataset.excluded_index,
            metric=self.metric,
            alpha=self.alpha,
            threshold=self.threshold
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
        batch_size = len(trn_data) if self.batch == -1 else self.batch
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
            steps=self.steps,
            units=self.units,
            batch=self.batch,
            alpha=self.alpha,
            threshold=self.threshold,
            folds=self.folds,
            fold=self.fold,
        )

    @property
    def key(self) -> str:
        metric = 'unc' if self.metric is None else self.metric.key
        return (f'{self.name}_{self.dataset.key}_{metric}_{self.steps}_{self.units}_{self.batch}_{self.alpha}_'
                f'{self.threshold}_{self.folds}_{self.fold}')

    @staticmethod
    def calibration(datasets: Dict[str, SurrogateDataset],
                    batches: Iterable[int] = (512, 4096, -1),
                    units: Iterable[Iterable[int]] = ((32,), (256,), (32,) * 2, (256,) * 2, (32,) * 3, (256,) * 3),
                    steps: int = 1000,
                    folds: int = 5,
                    wandb_project: Optional[str] = None,
                    formats: Iterable[str] = ('png',),
                    plot: bool = False):
        def configuration(ds, bt, ut, fl):
            classification = datasets[ds].classification
            return dict(dataset=ds, batch=bt, units=ut, fold=fl), [
                Loss(classification=classification),
                Accuracy(classification=classification)
            ]

        units = [list(u) for u in units]
        experiments = LearningExperiment.doe(
            file_name='learning',
            save_time=0,
            verbose=True,
            dataset=datasets,
            _batch={b: b for b in batches},
            _units={str(u): u for u in units},
            _threshold=0.0,
            _alpha=None,
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
            cl = len(units)
            rw = len(batches)
            fig, axes = plt.subplots(rw, cl, figsize=(5 * cl, 4 * rw), sharex='all', sharey='all', tight_layout=True)
            # used to index the axes in case either or both hidden units and batches have only one value
            axes = np.array(axes).reshape(rw, cl)
            for i, batch in enumerate(batches):
                for j, unit in enumerate(units):
                    sns.lineplot(
                        data=data[np.logical_and(data['batch'] == batch, data['units'] == str(unit))],
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
                    axes[i, j].set_ylim((0, 1 if kpi in ['R2', 'AUC'] else data[data['step'] > 20]['value'].max()))
                    axes[i, j].set_title(f"Batch Size: {'Full' if batch == -1 else batch} - Units: {unit}")
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
                metrics: Dict[str, HGR],
                steps: int = 500,
                folds: int = 5,
                units: Optional[Iterable[int]] = None,
                batch: Optional[int] = None,
                alpha: Optional[float] = None,
                threshold: Optional[float] = None,
                wandb_project: Optional[str] = None,
                formats: Iterable[str] = ('png',),
                plot: bool = False):
        metrics = {'//': None, **metrics}

        def configuration(ds, mt, fl):
            d = datasets[ds]
            # return a list of metrics for loss, accuracy, correlation, and optionally surrogate fairness
            return dict(Dataset=ds, Penalizer=mt, fold=fl), [
                Accuracy(classification=d.classification),
                Correlation(excluded=d.excluded_index, algorithm='sk', name=f'Protected HGR'),
                DIDI(excluded=d.surrogate_index, classification=d.classification, name=f'Surrogate DIDI')
            ]

        sns.set(context='poster', style='whitegrid')
        # iterate over dataset and batches
        for name, dataset in datasets.items():
            # use dictionaries for dataset to retrieve correct configuration
            experiments = LearningExperiment.doe(
                file_name='learning',
                save_time=0,
                verbose=True,
                dataset={name: dataset},
                metric=metrics,
                fold=list(range(folds)),
                folds=folds,
                _units=units,
                _batch=batch,
                _alpha=alpha,
                _threshold=threshold,
                steps=steps,
                wandb_project=wandb_project
            )
            # get and plot metric results
            group = LearningExperiment._metrics(experiments=experiments, configuration=configuration)
            kpis = group['kpi'].unique()
            col = len(kpis) + 1
            fig, axes = plt.subplots(2, col, figsize=(5 * col, 8), sharex='all', sharey=None, tight_layout=True)
            for i, sp in enumerate(['Train', 'Val']):
                for j, kpi in enumerate(kpis):
                    j += 1
                    sns.lineplot(
                        data=group[np.logical_and(group['split'] == sp, group['kpi'] == kpi)],
                        x='step',
                        y='value',
                        estimator='mean',
                        errorbar='sd',
                        linewidth=2,
                        hue='Penalizer',
                        style='Penalizer',
                        palette=PALETTE[:len(metrics)],
                        ax=axes[i, j]
                    )
                    axes[i, j].set_title(f"{kpi} ({sp.lower()})")
                    axes[i, j].get_legend().remove()
                    axes[i, j].set_ylabel(None)
                    if i == 1:
                        ub = axes[1, j].get_ylim()[1] if kpi == 'MSE' or kpi == 'BCE' or 'DIDI' in kpi else 1
                        axes[0, j].set_ylim((0, ub))
                        axes[1, j].set_ylim((0, ub))
            # get and plot lambda history
            lambdas = []
            for (_, mtr, fld), experiment in experiments.items():
                result = experiment.result
                alphas = ([np.nan] * len(result['alpha'])) if experiment.metric is None else result['alpha']
                lambdas.extend([{
                    'Penalizer': mtr,
                    'fold': fld,
                    'step': step,
                    'lambda': alphas[step]
                } for step in range(experiment.steps)])
            sns.lineplot(
                data=pd.DataFrame(lambdas),
                x='step',
                y='lambda',
                estimator='mean',
                errorbar='sd',
                linewidth=2,
                hue='Penalizer',
                style='Penalizer',
                palette=PALETTE[:len(metrics)],
                ax=axes[1, 0]
            )
            axes[1, 0].get_legend().remove()
            axes[1, 0].set_title('$\lambda$')
            axes[1, 0].set_ylabel(None)
            # plot legend
            handles, labels = axes[1, 0].get_legend_handles_labels()
            axes[0, 0].legend(handles, labels, title='PENALIZER', loc='center left', labelspacing=1.2, frameon=False)
            axes[0, 0].spines['top'].set_visible(False)
            axes[0, 0].spines['right'].set_visible(False)
            axes[0, 0].spines['bottom'].set_visible(False)
            axes[0, 0].spines['left'].set_visible(False)
            axes[0, 0].set_xticks([])
            axes[0, 0].set_yticks([])
            # store, print, and plot if necessary
            for extension in formats:
                filename = f'history_{name}.{extension}'
                with importlib.resources.path('experiments.exports', filename) as file:
                    fig.savefig(file, bbox_inches='tight')
            if plot:
                fig.suptitle(f"Learning History for {name.title()}")
                fig.show()
            plt.close(fig)

    @staticmethod
    def results(datasets: Dict[str, SurrogateDataset],
                metrics: Dict[str, HGR],
                steps: int = 500,
                folds: int = 5,
                units: Optional[Iterable[int]] = None,
                batch: Optional[int] = None,
                alpha: Optional[float] = None,
                threshold: Optional[float] = None,
                wandb_project: Optional[str] = None,
                formats: Iterable[str] = ('csv',)):
        # run experiments
        metrics = {'//': None, **metrics}
        experiments = LearningExperiment.doe(
            file_name='learning',
            save_time=0,
            verbose=True,
            dataset=datasets,
            metric=metrics,
            fold=list(range(folds)),
            folds=folds,
            _units=units,
            _batch=batch,
            _alpha=alpha,
            _threshold=threshold,
            steps=steps,
            wandb_project=wandb_project
        )
        group = []
        # retrieve results
        for (ds, mt, fl), experiment in tqdm(experiments.items(), desc='Computing KPIs'):
            dataset = datasets[ds]
            kpis = [
                Accuracy(classification=dataset.classification, name='SCORE'),
                Correlation(excluded=dataset.excluded_index, algorithm='kb', name=f'HGR'),
                DIDI(excluded=dataset.surrogate_index, classification=dataset.classification, name=f'DIDI')
            ]
            configuration = dict(Dataset=ds, Penalizer=mt)
            results = experiment.result(external=True)
            history = results['history'][experiment.steps - 1]
            group.append({**configuration, 'split': 'train', 'kpi': 'Time', 'value': results['execution']})
            for split in ['train', 'val']:
                x = results[f'{split}_inputs'].numpy(force=True)
                y = results[f'{split}_target'].numpy(force=True).flatten()
                p = history[f'{split}_predictions'].numpy(force=True).flatten()
                for kpi in kpis:
                    group.append({**configuration, 'split': split, 'kpi': kpi.name, 'value': kpi(x=x, y=y, p=p)})
        group = pd.DataFrame(group)
        group = group.groupby(['Dataset', 'Penalizer', 'split', 'kpi'], as_index=False).agg(['mean', 'std'])
        group.columns = ['Dataset', 'Penalizer', 'split', 'kpi', 'mean', 'std']
        group['text'] = [f"{row['mean']:.3f} ± {row['std']:.2f}" for _, row in group.iterrows()]
        group = group.pivot(index=['Dataset', 'Penalizer'], columns=['kpi', 'split']).reorder_levels([1, 2, 0], axis=1)
        group = group.reindex(index=[(d, m) for d in datasets.keys() for m in metrics.keys()])
        columns = [(kpi, split, agg)
                   for kpi in ['SCORE', 'HGR', 'DIDI']
                   for split in ['train', 'val']
                   for agg in ['mean', 'std', 'text']]
        group = group.reindex(columns=columns + [('Time', 'train', agg) for agg in ['mean', 'std', 'text']])
        if len(datasets) == 1:
            group = group.droplevel(0)
        if 'csv' in formats:
            df = group[(c for c in group.columns if c[2] != 'text')]
            with importlib.resources.path('experiments.exports', f'results.csv') as filepath:
                df.to_csv(filepath, header=True, index=True)
        if 'tex' in formats:
            df = group[(c for c in group.columns if c[2] == 'text')].droplevel(2, axis=1)
            with importlib.resources.path('experiments.exports', f'results.tex') as filepath:
                df.to_latex(filepath, multicolumn=True, multirow=False, multicolumn_format='c')

    @staticmethod
    def usecase(steps: int = 500,
                units: Optional[Iterable[int]] = None,
                batch: Optional[int] = None,
                alpha: Optional[float] = None,
                threshold: Optional[float] = None,
                wandb_project: Optional[str] = None,
                formats: Iterable[str] = ('png',),
                plot: bool = False):
        # run experiments
        dataset = Students()
        metrics = {'Unconstrained': None, 'Constrained': SingleKernelHGR()}
        experiments = LearningExperiment.doe(
            file_name='learning',
            save_time=0,
            verbose=True,
            dataset=dataset,
            metric=metrics,
            fold=0,
            folds=1,
            _units=units,
            _batch=batch,
            _alpha=alpha,
            _threshold=threshold,
            steps=steps,
            wandb_project=wandb_project
        )
        # retrieve results
        r2 = Accuracy(classification=False)
        hgr = Correlation(excluded=dataset.excluded_index, algorithm='kb')
        (xtr, ytr), (xts, yts) = [
            (data.drop(columns=dataset.target_name), data[dataset.target_name])
            for data in dataset.data(folds=1, seed=SEED)[0]
        ]
        surrogates = {name: (val.loc[xtr.index], val.loc[xts.index]) for name, val in dataset.surrogates.items()}
        outputs = []
        for metric, experiment in experiments.items():
            history = experiment.result['history'][experiment.steps - 1]
            for i, (x, y), split in zip([0, 1], [(xtr.values, ytr.values), (xts.values, yts.values)], ['train', 'val']):
                p = history[f'{split}_predictions'].numpy(force=True).flatten()
                outputs.append({'penalizer': metric, 'split': split, 'kpi': 'R2', 'value': r2(x=x, y=y, p=p)})
                outputs.append({'penalizer': metric, 'split': split, 'kpi': 'HGR', 'value': hgr(x=x, y=y, p=p)})
                for name, values in surrogates.items():
                    x = values[i].values.reshape((-1, 1))
                    # noinspection PyUnresolvedReferences
                    kpi = f"DIDI\n{name}"
                    mtr = DIDI(excluded=0, classification=False)
                    outputs.append({'penalizer': metric, 'split': split, 'kpi': kpi, 'value': mtr(x=x, y=y, p=p)})
        outputs = pd.DataFrame(outputs)
        # build plots
        sns.set(context='poster', style='whitegrid', font_scale=2)
        figures = {split: plt.figure(figsize=(26, 10), tight_layout=True) for split in ['train', 'val']}
        for split, fig in figures.items():
            ax = fig.gca()
            sns.barplot(
                data=outputs[outputs['split'] == split],
                x='kpi',
                y='value',
                hue='penalizer',
                linewidth=3,
                palette=PALETTE[:len(metrics)],
                ax=ax
            )
            ax.legend(*ax.get_legend_handles_labels(), title=None)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        # store, print, and plot if necessary
        for extension in formats:
            for split, fig in figures.items():
                filename = f'usecase_{split}.{extension}'
                with importlib.resources.path('experiments.exports', filename) as file:
                    fig.savefig(file, bbox_inches='tight')
        if plot:
            for split, fig in figures.items():
                fig.suptitle(f"Use Case Results ({split.title()})")
                fig.show()
        for fig in figures.values():
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

import importlib.resources
import itertools
import json
import math
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, List

import pytorch_lightning as pl
from tqdm import tqdm

from src.datasets import Dataset
from src.hgr import HGR
from src.serializable import Serializable


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Experiment(Serializable):
    """Interface for an experiment."""

    dataset: Dataset = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The dataset used in the experiment."""

    metric: HGR = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The HGR metric used in the experiment."""

    seed: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=0)
    """The random seed used in the experiment."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The experiment name."""
        pass

    @abstractmethod
    def _compute(self) -> Dict[str, Any]:
        """Computes the results of the experiment."""
        pass

    @property
    def _result(self) -> Dict[str, Any]:
        """Returns the result of the experiment. If the results were not cached yet, runs the experiment."""

        def function():
            pl.seed_everything(seed=self.seed, workers=True)
            start = time.time()
            result = self._compute()
            execution = time.time() - start
            return {**result, 'execution': execution, 'timestamp': start}

        return self._lazy_initialization(attribute='result', function=function)

    @property
    def result(self) -> Dict[str, Any]:
        """The result of the experiment."""
        return self._result.copy()

    @property
    def output(self) -> Dict[str, Any]:
        """The full output of the experiment, i.e., configuration and result."""
        return {**self.configuration, 'result': self._result}

    @property
    def configuration(self) -> Dict[str, Any]:
        return {
            'experiment': self.name,
            'dataset': self.dataset.configuration,
            'metric': self.metric.configuration,
            'seed': self.seed
        }

    @property
    def key(self) -> str:
        return f'{self.name}_{self.dataset.key}_{self.metric.key}_{self.seed}'

    @classmethod
    def doe(cls, file_name: str, save_time: int, **configuration: Iterable) -> List['Experiment']:
        """Runs a combinatorial design of experiments (DoE) with the given characteristics. If possible, loads results
        from the given file which must be stored in the 'results' sub-package. When experiments are running, stores
        their results in the given file every <save_time> seconds."""
        # retrieve the path of the results and load the json dictionary if the file exists
        with importlib.resources.path('experiments.results', file_name) as path:
            pass
        if path.exists():
            with open(path, 'r') as file:
                results = json.load(fp=file)
        else:
            results = {}
        # create the experiments and run the combinatorial design
        gap = time.time()
        keys = configuration.keys()
        values = configuration.values()
        parameters = itertools.product(*values)
        experiments = []
        for params in tqdm(parameters, total=math.prod([len(v) for v in values])):
            # build the input configuration and use it to create an instance of the experiment
            config = {k: v for k, v in zip(keys, params)}
            experiment = cls(**config)
            experiments.append(experiment)
            # check if the experiment output is already in the dictionary based on its key
            key = experiment.key
            out = results.get(key)
            # if the experiment is not in the dictionary, store its output (it computes the results automatically)
            # otherwise, check that the data is correct and inject the stored result
            if out is None:
                results[key] = experiment.output
            else:
                out = out.copy()
                for k, exp in experiment.configuration.items():
                    ref = out.pop(k)
                    assert exp == ref, f"Wrong attribute '{k}' loaded for '{key}', expected {exp}, got {ref}"
                experiment._cache['result'] = out.pop('result')
                assert len(out) == 0, f"Output has additional keys {out.keys()} which are not expected for '{key}'"
            # whenever the gap is larger than the expected time, flush the results in the file
            if time.time() - gap >= save_time:
                with open(path, 'w') as file:
                    json.dump(results, fp=file, indent=2)
                gap = time.time()
        # flush the results again at the end of the doe, then return the list of experiments
        with open(path, 'w') as file:
            json.dump(results, fp=file, indent=2)
        return experiments

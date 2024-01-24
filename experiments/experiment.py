import importlib.resources
import itertools
import pickle
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

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
    def doe(cls, file_name: str, save_time: int, **configuration: Any) -> Dict[Any, 'Experiment']:
        """Runs a combinatorial design of experiments (DoE) with the given characteristics. If possible, loads results
        from the given file which must be stored in the 'results' sub-package. When experiments are running, stores
        their results in the given file every <save_time> seconds."""
        assert len(configuration) > 0, "Empty configuration passed"
        # retrieve the path of the results and load the json dictionary if the file exists
        with importlib.resources.path('experiments.results', f'{file_name}.pkl') as path:
            pass
        if path.exists():
            with open(path, 'rb') as file:
                results = pickle.load(file=file)
        else:
            results = {}
        # iterate through the configuration to create the combinatorial design
        gap = time.time()
        indices, parameters, total = [], [], 1
        for param in configuration.values():
            if isinstance(param, dict):
                indices.append(param.keys())
                parameters.append(param.values())
                total *= len(param)
            elif isinstance(param, list):
                indices.append(param)
                parameters.append(param)
                total *= len(param)
            else:
                parameters.append([param])
        indices = indices[0] if len(indices) == 1 else itertools.product(*indices)
        parameters = itertools.product(*parameters)
        signature = configuration.keys()
        # run and store the experiments
        to_save = False
        experiments = {}
        for index, param in tqdm(zip(indices, parameters), total=total):
            # build the input configuration and use it to create an instance of the experiment
            config = {k: v for k, v in zip(signature, param)}
            experiment = cls(**config)
            experiments[index] = experiment
            # check if the experiment output is already in the dictionary based on its key
            key = experiment.key
            out = results.get(key)
            # if the experiment is not in the dictionary, store its output (it computes the results automatically)
            # otherwise, check that the data is correct and inject the stored result
            if out is None:
                results[key] = experiment.output
                # whenever the gap is larger than the expected time save the results
                # otherwise, flag that results must be saved at the end of the doe
                if time.time() - gap >= save_time:
                    # dump the file before writing to check if it is json-compliant
                    dump = pickle.dumps(results)
                    with open(path, 'wb') as file:
                        file.write(dump)
                    gap = time.time()
                    to_save = False
                else:
                    to_save = True
            else:
                out = out.copy()
                for k, exp in experiment.configuration.items():
                    ref = out.pop(k)
                    assert exp == ref, f"Wrong attribute '{k}' loaded for '{key}', expected {exp}, got {ref}"
                experiment._cache['result'] = out.pop('result')
                assert len(out) == 0, f"Output has additional keys {out.keys()} which are not expected for '{key}'"
        # if necessary, save the results at the end of the doe, then return the experiments
        if to_save:
            # dump the file before writing to check if it is json-compliant
            dump = pickle.dumps(results)
            with open(path, 'wb') as file:
                file.write(dump)
        return experiments

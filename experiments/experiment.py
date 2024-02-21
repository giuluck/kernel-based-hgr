import gc
import itertools
import os
import pickle
import re
import shutil
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterable, Callable

from tqdm import tqdm

from src.hgr import Oracle
from src.serializable import Cacheable, Serializable


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Experiment(Cacheable):
    """Interface for an experiment."""

    class Result(Serializable):
        """Wrapper class for an experiment result."""

        def __init__(self, timestamp: float, execution: float, external: Optional[str], **kwargs):
            """Builds a new result file with the given timestamp, execution time, and other arguments.
            Moreover, it keeps a pointer to an external file containing additional results."""
            self._kwargs: Dict[str, Any] = dict(timestamp=timestamp, execution=execution, **kwargs)
            self._external: Optional[str] = external

        def __getitem__(self, key: str) -> Any:
            # id the key is in the default arguments, return it
            output = self._kwargs.get(key)
            if output is not None:
                return output
            # otherwise raise an exception
            raise KeyError(f"Key '{key}' is either not defined for experiment {self} or in the external files")

        def __call__(self, external: Optional[str] = None):
            """Returns all the results in form of dictionary (include external results if needed)."""
            return {**self._kwargs} if external is None else {**self._kwargs, **self._external_kwargs(folder=external)}

        @property
        def configuration(self) -> Dict[str, Any]:
            return dict(external=self._external, **self._kwargs)

        @property
        def external(self) -> Optional[str]:
            return self._external

        def _external_kwargs(self, folder: str) -> Dict[str, Any]:
            """The external kwargs."""
            if self._external is None:
                return dict()
            with open(os.path.join(folder, 'results', self._external), 'rb') as file:
                return pickle.load(file=file)

    @property
    @abstractmethod
    def name(self) -> str:
        """The experiment name."""
        pass

    @abstractmethod
    def _compute(self, folder: str) -> Result:
        """Computes the results of the experiment and stores them in the given folder."""
        pass

    @property
    def result(self) -> Result:
        """Returns the result of the experiment. If the results were not cached yet, runs the experiment."""
        return self._cache['result']

    @property
    def output(self) -> Dict[str, Any]:
        """The full output of the experiment, i.e., configuration and result."""
        return {**self.configuration, 'result': self.result.configuration}

    @classmethod
    def doe(cls, folder: str, file_name: str, save_time: int, verbose: bool, **configuration: Any) -> dict:
        """Runs a combinatorial design of experiments (DoE) with the given characteristics. If possible, loads results
        from the given file which must be stored in the 'results' sub-package. When experiments are running, stores
        their results in the given file every <save_time> seconds."""
        assert len(configuration) > 0, "Empty configuration passed"
        # retrieve the path of the results and load the pickle dictionary if the file exists
        exp_folder = os.path.join(folder, 'exports')
        os.makedirs(exp_folder, exist_ok=True)
        res_folder = os.path.join(folder, 'results')
        os.makedirs(res_folder, exist_ok=True)
        filepath = os.path.join(res_folder, f'{file_name}.pkl')
        if os.path.exists(filepath):
            with open(filepath, 'rb') as file:
                results = pickle.load(file=file)
        else:
            results = {}
        # iterate through the configuration to create the combinatorial design
        indices, parameters, total = [], [], 1
        for param in configuration.values():
            if isinstance(param, dict):
                indices.append(param.keys())
                parameters.append(param.values())
                total *= len(param)
            elif isinstance(param, list):
                indices.append(list(range(len(param))))
                parameters.append(param)
                total *= len(param)
            else:
                parameters.append([param])
        indices = indices[0] if len(indices) == 1 else itertools.product(*indices)
        parameters = itertools.product(*parameters)
        signature = configuration.keys()
        # run and store the experiments
        experiments = {}
        gap = time.time()
        to_save = False
        iterable = enumerate(zip(indices, parameters))
        if not verbose:
            iterable = tqdm(iterable, total=total, desc='Fetching Experiments')
        for i, (index, param) in iterable:
            # build the input configuration and use it to create an instance of the experiment
            config = {k: v for k, v in zip(signature, param)}
            # -------------------------------------------------------------------------------------
            # QUICK PATCH TO HANDLE ORACLE METRIC WHICH NEEDS TO BE BUILT WITH A DATASET INSTANCE
            metric = config.get('metric')
            if metric is not None and isinstance(metric, Oracle):
                dataset = config.get('dataset')
                assert dataset is not None, "Trying to use Oracle metric without passing a dataset"
                # noinspection PyArgumentList
                config['metric'] = metric.instance(dataset=dataset)
            # -------------------------------------------------------------------------------------
            # noinspection PyArgumentList
            experiment = cls(**config)
            experiments[index] = experiment
            # check if the experiment output is already in the dictionary based on its key
            key = experiment.key
            out = results.get(key)
            # if the experiment is not in the dictionary, store its output (it computes the results automatically)
            # otherwise, check that the data is correct and inject the stored result
            if out is None:
                if verbose:
                    print(flush=True)
                    print(f'Running Experiment {i + 1} of {total}:')
                    for par, val in experiment.configuration.items():
                        print(f'  > {par.upper()}: {val}')
                    print(end='', flush=True)
                experiment._cache['result'] = experiment._compute(folder=folder)
                results[key] = experiment.output
                # whenever the gap is larger than the expected time save the results
                # otherwise, flag that results must be saved at the end of the doe
                if time.time() - gap >= save_time:
                    # dump the file before writing to check if it is pickle-compliant
                    dump = pickle.dumps(results)
                    with open(filepath, 'wb') as file:
                        file.write(dump)
                    gap = time.time()
                    # run garbage collector to free memory before running the next experiments
                    gc.collect()
                    to_save = False
                else:
                    to_save = True
            else:
                out = out.copy()
                for k, exp in experiment.configuration.items():
                    ref = out.pop(k)
                    assert exp == ref, f"Wrong attribute '{k}' loaded for '{key}', expected {exp}, got {ref}"
                experiment._cache['result'] = Experiment.Result(**out.pop('result'))
                assert len(out) == 0, f"Output has additional keys {out.keys()} which are not expected for '{key}'"
        # if necessary, save the results at the end of the doe, then return the experiments
        if to_save:
            # dump the file before writing to check if it is pickle-compliant
            dump = pickle.dumps(results)
            with open(filepath, 'wb') as file:
                file.write(dump)
            # run garbage collector to free memory before running the next experiments
            gc.collect()
        return experiments

    @staticmethod
    def clear_results(folder: str,
                      file: List[str],
                      dataset: Optional[Iterable[str]] = None,
                      metric: Optional[Iterable[Optional[str]]] = None,
                      pattern: Optional[str] = None,
                      custom: Optional[Callable[[dict], bool]] = None,
                      force: bool = False):
        folder = os.path.join(folder, 'results')
        # build sets and pattern
        pattern = None if pattern is None else re.compile(pattern)
        datasets = None if dataset is None else set(dataset)
        metrics = None if metric is None else set(metric)
        # iterate over all the files
        for filename in file:
            # if it does not exist, there is nothing to clear
            filepath = os.path.join(folder, f'{filename}.pkl')
            if not os.path.exists(filepath):
                continue
            # otherwise, retrieve the dictionary of results
            with open(filepath, 'rb') as file:
                results = pickle.load(file=file)
            print(f"Retrieved {len(results)} experiments from '{filename}.pkl'")
            # build a dictionary of results to keep
            # a result must if there is at least a non-null matcher that does not match
            output = {}
            externals = []
            histories = []
            for idx, res in results.items():
                if datasets is not None and res['dataset']['name'] not in datasets:
                    # print(f"LEAVE: '{idx}' from '{filename}.pkl' (unmatch dataset '{res['dataset']['name']}')")
                    output[idx] = res
                elif metrics is not None and res['metric']['name'] not in metrics:
                    # print(f"LEAVE: '{idx}' from '{filename}.pkl' (unmatch metric '{res['metric']['name']}')")
                    output[idx] = res
                elif pattern is not None and not pattern.match(idx):
                    # print(f"LEAVE: '{idx}' from '{filename}.pkl' (unmatch pattern)")
                    output[idx] = res
                elif custom is not None and not custom(res):
                    output[idx] = res
                else:
                    external = res['result']['external']
                    print(f"CLEAR: '{idx}' from '{filename}.pkl'", end='')
                    if external is not None:
                        externals.append(external)
                        print(f" and external file '{external}'", end='')
                    # -------------------------------------------------------------------------------------
                    # QUICK PATCH TO MULTIPLE EXTERNAL FILES IN HISTORY CALLBACK FOR LEARNING EXPERIMENTS
                    if res['experiment'] == 'learning':
                        history = res['result']['history'].subfolder
                        histories.append(history)
                        print(f" (plus history files in {history})", end='')
                    # -------------------------------------------------------------------------------------
                    print()
            if not force:
                result = input(f"\nAre you sure you want to remove {len(results) - len(output)} experiments from "
                               f"'{filename}.pkl', leaving {len(output)} experiments left? (Y/N)\n")
                if result.lower() not in ['y', 'yes']:
                    print(f"\nClearing procedure for file '{filename}.pkl' aborted\n")
                    break
                else:
                    print(f"\nClearing procedure for file '{filename}.pkl' completed\n")
            else:
                print(f"Removed {len(results) - len(output)} experiments from '{filename}.pkl ({len(output)} left)\n")
            # dump the file before writing to check if it is pickle-compliant
            dump = pickle.dumps(output)
            with open(filepath, 'wb') as file:
                file.write(dump)
            for external in externals:
                os.remove(os.path.join(folder, external))
            for history in histories:
                shutil.rmtree(os.path.join(folder, history))

    @staticmethod
    def clear_exports(folder: str):
        folder = os.path.join(folder, 'exports')
        if not os.path.exists(folder):
            return
        for file in os.listdir(folder):
            print(f'CLEAR: export file {file}')
            os.remove(os.path.join(folder, file))

import importlib.resources
import json
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

from src.datasets import Dataset
from src.hgr import HGR
from src.serializable import Serializable


@dataclass(frozen=True)
class Experiment(Serializable):
    """Interface for an experiment."""

    dataset: Dataset = field()
    """The dataset used in the experiment."""

    metric: HGR = field()
    """The HGR metric used in the experiment."""

    _cached_result: Dict[str, Any] = field(init=False, repr=False, default_factory=dict)
    """The experiment result."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The experiment name."""
        pass

    @abstractmethod
    def _run(self) -> Dict[str, Any]:
        """Runs the experiment and returns its results."""
        pass

    @property
    def _result(self) -> Dict[str, Any]:
        """Returns the result of the experiment. If the results were not cached yet, tries to load it from the stored
        files or, if the file does not exist, runs the experiment."""
        if len(self._cached_result) == 0:
            with importlib.resources.path('experiments.results', self.filename) as file:
                if not file.exists():
                    start = time.time()
                    result = self._run()
                    gap = time.time() - start
                    self._cached_result.update({**result, 'time': gap, 'timestamp': start})
                    with open(file, mode='w') as out_file:
                        json.dump(self.config, fp=out_file, indent=2)
                else:
                    with open(file, mode='r') as in_file:
                        config = json.load(fp=in_file)
                    try:
                        metric = config.pop('metric')
                        dataset = config.pop('dataset')
                        experiment = config.pop('experiment')
                    except KeyError:
                        raise AssertionError(f"Error when loading experiment '{self.filename}'")
                    assert metric == self.metric.config, f"Error when loading experiment '{self.filename}'"
                    assert dataset == self.dataset.config, f"Error when loading experiment '{self.filename}'"
                    assert experiment == self.name, f"Error when loading experiment '{self.filename}'"
                    assert 'timestamp' in config, f"Error when loading experiment '{self.filename}'"
                    self._cached_result.update(config)
        return self._cached_result

    @property
    def config(self) -> Dict[str, Any]:
        return {
            'experiment': self.name,
            'dataset': self.dataset.config,
            'metric': self.metric.config,
            **self._result
        }

    @property
    def filename(self) -> str:
        """The experiment name that can be used for file naming."""
        return f'{self.name}_d={self.dataset.fullname}_m={self.metric.fullname}.json'

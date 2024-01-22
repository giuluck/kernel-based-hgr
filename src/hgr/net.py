from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch

from src.hgr.hgr import HGR


@dataclass(frozen=True)
class AdversarialHGR(HGR):
    @property
    def name(self) -> str:
        return 'nn'

    @property
    def config(self) -> Dict[str, Any]:
        return dict()

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError()

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

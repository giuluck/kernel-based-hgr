from dataclasses import dataclass

import torch

from src.hgr.hgr import HGR


@dataclass(frozen=True)
class AdversarialHGR(HGR):
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

from dataclasses import dataclass, field

import torch

from src.hgr.hgr import HGR


@dataclass(frozen=True)
class AdversarialHGR(HGR):
    @property
    def name(self) -> str:
        return 'nn'

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

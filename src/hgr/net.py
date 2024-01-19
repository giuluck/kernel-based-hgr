from dataclasses import dataclass, field

import torch

from src.hgr.hgr import HGR


@dataclass(frozen=True)
class AdversarialHGR(HGR):
    name: str = field(kw_only=True, default='HGR-NN')

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

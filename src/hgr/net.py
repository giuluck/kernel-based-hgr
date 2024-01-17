import torch

from src.hgr.hgr import HGR


class AdversarialHGR(HGR):
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

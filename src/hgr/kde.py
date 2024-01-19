from dataclasses import dataclass, field
from math import sqrt, pi

import torch

from src.hgr.hgr import HGR


class KDE:
    """A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization."""

    SQRT_PI: float = sqrt(2 * pi)

    def __init__(self, train):
        n, d = train.shape
        self.dimension = d
        self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
        self.train = train

    @staticmethod
    def _unsqueeze_multiple_times(inp, axis, times):
        out = inp
        for i in range(times):
            out = out.unsqueeze(axis)
        return out

    def pdf(self, x):
        assert x.shape[-1] == self.dimension, f"Expected dimension {self.dimension}, got {x.shape[-1]}"
        data = x.unsqueeze(-2)
        train = self._unsqueeze_multiple_times(self.train, 0, len(x.shape) - 1)
        # noinspection PyTypeChecker
        gaussian_values = torch.exp(-((data - train).norm(dim=-1) ** 2 / (self.bandwidth ** 2) / 2))
        return gaussian_values.mean(dim=-1) / self.bandwidth / self.SQRT_PI


@dataclass(frozen=True)
class DensityHGR(HGR):
    """Torch-based implementation of the HGR/CHI2 metric obtained from the official repository of "Fairness-Aware
    Learning for Continuous Attributes and Treatments" (https://github.com/criteo-research/continuous-fairness/)."""

    chi2: bool = field(kw_only=True)
    """Whether to return the chi^2 approximation of the HGR or its actual value."""

    @staticmethod
    def joint_2(x: torch.Tensor, y: torch.Tensor, damping: float = 1e-10, eps: float = 1e-9) -> torch.Tensor:
        # add an eps value to avoid nan vectors in case of very degraded solutions
        x = (x - x.mean()) / (x.std(dim=None, correction=0) + eps)
        y = (y - y.mean()) / (y.std(dim=None, correction=0) + eps)
        data = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], -1)
        joint_density = KDE(data)
        n_bins = int(min(50, 5. / joint_density.bandwidth))
        x_centers = torch.linspace(-2.5, 2.5, n_bins)
        y_centers = torch.linspace(-2.5, 2.5, n_bins)
        xx, yy = torch.meshgrid([x_centers, y_centers], indexing='ij')
        grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], -1)
        h2d = joint_density.pdf(grid) + damping
        h2d /= h2d.sum()
        return h2d

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h2d = DensityHGR.joint_2(a, b)
        marginal_a = h2d.sum(dim=1).unsqueeze(1)
        marginal_b = h2d.sum(dim=0).unsqueeze(0)
        q = h2d / (torch.sqrt(marginal_a) * torch.sqrt(marginal_b))
        if self.chi2:
            return (q ** 2).sum(dim=[0, 1]) - 1.0
        else:
            return torch.linalg.svd(q)[1][1]

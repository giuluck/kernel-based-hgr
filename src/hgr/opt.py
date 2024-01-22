from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

import numpy as np
import torch
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import pearsonr

from src.hgr import HGR

EPS: float = 0.0
"""The tolerance used to account for null standard deviation."""


@dataclass(frozen=True)
class KernelBasedHGR(HGR):
    """Computes the Kernel-based HGR by solving a constrained least square problem using a minimization solver."""

    degree_a: int = field(kw_only=True)
    """The kernel degree for the first variable."""

    degree_b: int = field(kw_only=True)
    """The kernel degree for the first variable."""

    @property
    def config(self) -> Dict[str, Any]:
        return dict(name=self.name, degree_a=self.degree_a, degree_b=self.degree_b)

    @property
    def name(self) -> str:
        return 'kb'

    @staticmethod
    def higher_order_coefficients(f: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the kernel-based hgr for higher order degrees."""
        # compute the kernel matrices
        n, dx = f.shape
        _, dy = g.shape
        d = dx + dy
        fg = np.concatenate((f, -g), axis=1)

        # define the function to optimize as the least square problem:
        #   - func:   || F @ alpha - G @ beta ||_2^2 =
        #           =   (F @ alpha - G @ beta) @ (F @ alpha - G @ beta)
        #   - grad:   [ 2 * F.T @ (F @ alpha - G @ beta) | -2 * G.T @ (F @ alpha - G @ beta) ] =
        #           =   2 * [F | -G].T @ (F @ alpha - G @ beta)
        #   - hess:   [  2 * F.T @ F | -2 * F.T @ G ]
        #             [ -2 * G.T @ F |  2 * G.T @ G ] =
        #           =    2 * [F  -G].T @ [F  -G]
        def _fun(inp):
            alp, bet = inp[:dx], inp[dx:]
            diff = f @ alp - g @ bet
            obj_func = diff @ diff
            obj_grad = 2 * fg.T @ diff
            return obj_func, obj_grad

        fun_hess = 2 * fg.T @ fg

        # define the constraint
        #   - func:   var(G @ beta) --> = 1
        #   - grad: [ 0 | 2 * G.T @ G @ beta / n ]
        #   - hess: [ 0 |         0       ]
        #           [ 0 | 2 * G.T @ G / n ]
        cst_hess = np.zeros(shape=(d, d), dtype=float)
        cst_hess[dx:, dx:] = 2 * g.T @ g / n
        constraint = NonlinearConstraint(
            fun=lambda inp: np.var(g @ inp[dx:], ddof=0),
            jac=lambda inp: np.concatenate(([0] * dx, 2 * g.T @ g @ inp[dx:] / n)),
            hess=lambda *_: cst_hess,
            lb=1,
            ub=1
        )

        # set the initial point as [ 1 / std(F @ 1) | 1 / std(G @ 1) ] then solve the problem
        alp0 = np.ones(dx) / f.sum(axis=1).std(ddof=0)
        bet0 = np.ones(dy) / g.sum(axis=1).std(ddof=0)
        x0 = np.concatenate((alp0, bet0))
        s = minimize(_fun, jac=True, hess=lambda *_: fun_hess, x0=x0, constraints=[constraint], method='trust-constr')
        return s.x[:dx], s.x[dx:]

    def correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        # build the kernel matrices
        f = np.stack([a ** d - np.mean(a ** d) for d in np.arange(self.degree_a) + 1], axis=1)
        g = np.stack([b ** d - np.mean(b ** d) for d in np.arange(self.degree_b) + 1], axis=1)
        # handle trivial or simpler cases:
        #  - if both degrees are 1, simply compute the projected vectors as standardized original vectors
        #  - if one degree is 1, standardize that vector and compute the other's coefficients using lstsq
        #  - if no degree is 1, use the optimization routine and compute the projected vectors from the coefficients
        if self.degree_a == 1 and self.degree_b == 1:
            alpha, beta = np.ones(1), np.ones(1)
        elif self.degree_a == 1:
            alpha = np.ones(1) / (a.std(ddof=0) + EPS)
            beta, _, _, _ = np.linalg.lstsq(g, f @ alpha, rcond=None)
        elif self.degree_b == 1:
            beta = np.ones(1) / (b.std(ddof=0) + EPS)
            alpha, _, _, _ = np.linalg.lstsq(f, g @ beta, rcond=None)
        else:
            alpha, beta = self.higher_order_coefficients(f=f, g=g)
        fa = f @ alpha
        gb = g @ beta
        correlation, _ = pearsonr(fa, gb)
        alpha = alpha / (fa.std(ddof=0) + EPS)
        beta = beta / (gb.std(ddof=0) + EPS)
        return dict(correlation=abs(correlation), alpha=list(alpha), beta=list(beta))

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        def standardize(t: torch.Tensor) -> torch.Tensor:
            t_std, t_mean = torch.std_mean(t, correction=0)
            return (t - t_mean) / (t_std + EPS)

        # build the kernel matrices
        f = torch.stack([a ** d - torch.mean(a ** d) for d in np.arange(self.degree_a) + 1], dim=1)
        g = torch.stack([b ** d - torch.mean(b ** d) for d in np.arange(self.degree_b) + 1], dim=1)
        # handle trivial or simpler cases:
        #  - if both degrees are 1, simply compute the projected vectors as standardized original vectors
        #  - if one degree is 1, standardize that vector and compute the other's coefficients using lstsq
        #  - if no degree is 1, use the optimization routine and compute the projected vectors from the coefficients
        if self.degree_a == 1 and self.degree_b == 1:
            fa = standardize(a)
            gb = standardize(b)
        elif self.degree_a == 1:
            # the 'gelsd' driver allows to have both more precise and more reproducible results
            fa = standardize(a)
            beta, _, _, _ = torch.linalg.lstsq(g, fa, driver='gelsd')
            gb = standardize(g @ beta)
        elif self.degree_b == 1:
            # the 'gelsd' driver allows to have both more precise and more reproducible results
            gb = standardize(b)
            alpha, _, _, _ = torch.linalg.lstsq(f, gb, driver='gelsd')
            fa = standardize(f @ alpha)
        else:
            alpha, beta = self.higher_order_coefficients(f=f.numpy(force=True), g=g.numpy(force=True))
            fa = standardize(f @ torch.tensor(alpha))
            gb = standardize(g @ torch.tensor(beta))
        # return the correlation as the absolute value of the vector product (since the vectors are standardized)
        return torch.abs(fa @ gb)

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

import numpy as np
import torch
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import pearsonr

from src.hgr import HGR

EPS: float = 0.0
"""The tolerance used to account for null standard deviation."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class KernelBasedHGR(HGR):
    """Kernel-based HGR computed by solving a constrained least square problem using a minimization solver."""

    degree_a: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The kernel degree for the first variable."""

    degree_b: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The kernel degree for the first variable."""

    @property
    def name(self) -> str:
        return 'kb'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, degree_a=self.degree_a, degree_b=self.degree_b)

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
            std = a.std(ddof=0) + EPS
            alpha = np.ones(1) / std
            beta, _, _, _ = np.linalg.lstsq(g, f[:, 0] / std, rcond=None)
        elif self.degree_b == 1:
            std = b.std(ddof=0) + EPS
            beta = np.ones(1) / std
            alpha, _, _, _ = np.linalg.lstsq(f, g[:, 0] / std, rcond=None)
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


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class SingleKernelHGR(HGR):
    """Kernel-based HGR computed using one kernel only for both variables and then taking the maximal correlation."""

    degree: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True)
    """The kernel degree for the variables."""

    @property
    def name(self) -> str:
        return 'sk'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, degree=self.degree)

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        # build the kernel matrices
        f = np.stack([a ** d - np.mean(a ** d) for d in np.arange(self.degree) + 1], axis=1)
        g = np.stack([b ** d - np.mean(b ** d) for d in np.arange(self.degree) + 1], axis=1)
        # compute correlation for kernel on alpha
        beta_a = 1 / (b.std(ddof=0) + EPS)
        gb_a = g[:, 0] * beta_a
        alpha_a, _, _, _ = np.linalg.lstsq(f, gb_a, rcond=None)
        correlation_a, _ = pearsonr(f @ alpha_a, gb_a)
        # compute correlation for kernel on beta
        alpha_b = 1 / (a.std(ddof=0) + EPS)
        fa_b = f[:, 0] * alpha_b
        beta_b, _, _, _ = np.linalg.lstsq(g, fa_b, rcond=None)
        correlation_b, _ = pearsonr(fa_b, g @ beta_b)
        # choose the best correlation and return
        if correlation_a > correlation_b:
            correlation, alpha, beta = correlation_a, list(alpha_a), [beta_a] + [0] * (self.degree - 1)
        else:
            correlation, alpha, beta = correlation_b, [alpha_b] + [0] * (self.degree - 1), list(beta_b)
        return dict(correlation=correlation, alpha=alpha, beta=beta)

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        def standardize(t: torch.Tensor) -> torch.Tensor:
            t_std, t_mean = torch.std_mean(t, correction=0)
            return (t - t_mean) / (t_std + EPS)

        # build the kernel matrices
        f = torch.stack([a ** d - torch.mean(a ** d) for d in np.arange(self.degree) + 1], dim=1)
        g = torch.stack([b ** d - torch.mean(b ** d) for d in np.arange(self.degree) + 1], dim=1)
        # compute correlation for kernel on alpha ('gelsd' allows to have more precise and reproducible results)
        gb_a = standardize(b)
        alpha_a, _, _, _ = torch.linalg.lstsq(f, gb_a, driver='gelsd')
        fa_a = standardize(f @ alpha_a)
        correlation_a = torch.abs(fa_a @ gb_a)
        # compute correlation for kernel on beta ('gelsd' allows to have more precise and reproducible results)
        fa_b = standardize(a)
        beta_b, _, _, _ = torch.linalg.lstsq(g, fa_b, driver='gelsd')
        gb_b = standardize(g @ beta_b)
        correlation_b = torch.abs(fa_b @ gb_b)
        # return the maximal correlation
        return torch.maximum(correlation_a, correlation_b)

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import pearsonr

from src.hgr.hgr import HGR


@dataclass(frozen=True)
class KernelBasedHGR(HGR):
    """Computes the Kernel-based HGR by solving a constrained least square problem using a minimization solver."""

    degree_a: int = field(kw_only=True)
    """The kernel degree for the first variable."""

    degree_b: int = field(kw_only=True)
    """The kernel degree for the first variable."""

    eps: float = field(kw_only=True, default=1e-9)
    """The tolerance used to account for null standard deviation."""

    smart_init: bool = field(kw_only=True, default=True)
    """Whether to initialize x0 as [pearson(x, y) / std(x), 0, ..., 0, 1 / std(y), 0, ..., 0] or as a random vector."""

    method: str = field(kw_only=True, default='trust-constr')
    """The method to solve the constrained optimization problem, either 'SLSQP' or 'trust-constr'."""

    lasso: float = field(kw_only=True, default=0.0)
    """The amount of lasso penalization."""

    def _higher_order_coefficients(self, f: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        #
        # plus, add the lasso penalizer
        #   - func:     norm_1([alpha, beta])
        #   - grad:   [ sign(alpha) | sign(beta) ]
        #   - hess:   [      0      |      0     ]
        #             [      0      |      0     ]
        if self.lasso == 0:
            def _fun(inp):
                alp, bet = inp[:dx], inp[dx:]
                diff = f @ alp - g @ bet
                obj_func = diff @ diff
                obj_grad = 2 * fg.T @ diff
                return obj_func, obj_grad
        else:
            def _fun(inp):
                alp, bet = inp[:dx], inp[dx:]
                diff = f @ alp - g @ bet
                obj_func = diff @ diff
                obj_grad = 2 * fg.T @ diff
                pen_func = np.abs(inp).sum()
                pen_grad = np.sign(inp)
                return obj_func + self.lasso * pen_func, obj_grad + self.lasso * pen_grad

        fun_hess = 2 * fg.T @ fg

        # define the constraint
        #   - func:   var(G @ beta) --> = 1
        #   - grad: [ 0 | 2 * G.T @ G @ beta / n ]
        #   - hess: [ 0 |         0       ]
        #           [ 0 | 2 * G.T @ G / n ]
        cst_hess = np.zeros(shape=(d, d), dtype=float)
        cst_hess[dx:, dx:] = 2 * g.T @ g / n
        constraint = NonlinearConstraint(
            fun=lambda inp: np.var(g @ inp[dx:]),
            jac=lambda inp: np.concatenate(([0] * dx, 2 * g.T @ g @ inp[dx:] / n)),
            hess=lambda *_: cst_hess,
            lb=1,
            ub=1
        )

        # choose an initial point based on the strategy
        if self.smart_init:
            # Start from the solution of the problem with degree_x = degree_y = 1, i.e.:
            #   - alpha = [ r / std(x), 0, ..., 0 ]
            #   -  beta = [ 1 / std(y), 0, ..., 0 ]
            a = f[:, 0]
            b = g[:, 0]
            x0 = np.zeros(shape=d, dtype=float)
            x0[0] = abs(pearsonr(a, b)[0]) / a.std()
            x0[self.degree_a] = 1. / b.std()
        else:
            x0 = np.random.random(size=d)

        # solve the problem and return the results
        s = minimize(_fun, jac=True, hess=lambda *_: fun_hess, x0=x0, constraints=[constraint], method=self.method)
        alpha, beta = s.x[:self.degree_a], s.x[self.degree_a:]
        return alpha, beta

    def kbhgr(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the HGR in the same way as in self.correlation() returns the alpha and beta coefficients as well."""
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
            alpha = np.ones(1) / (a.std(ddof=1) + self.eps)
            beta, _, _, _ = np.linalg.lstsq(g, f @ a, rcond=None)
        elif self.degree_b == 1:
            beta = np.ones(1) / (b.std(ddof=1) + self.eps)
            alpha, _, _, _ = np.linalg.lstsq(f, g @ beta, rcond=None)
        else:
            alpha, beta = self._higher_order_coefficients(f=f, g=g)
        fa = f @ alpha
        gb = g @ beta
        correlation, _ = pearsonr(fa, gb)
        alpha = alpha / fa.std(ddof=0)
        beta = beta / gb.std(ddof=0)
        return abs(correlation), alpha, beta

    def correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.kbhgr(a=a, b=b)[0]

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        def standardize(t: torch.Tensor) -> torch.Tensor:
            t_std, t_mean = torch.std_mean(t, correction=0)
            return (t - t_mean) / (t_std + self.eps)

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
            alpha, beta = self._higher_order_coefficients(f=f.numpy(force=True), g=g.numpy(force=True))
            fa = standardize(f @ torch.tensor(alpha))
            gb = standardize(g @ torch.tensor(beta))
        # return the correlation as the absolute value of the vector product (since the vectors are standardized)
        return torch.abs(fa @ gb)

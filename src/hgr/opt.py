from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import scipy.linalg
import torch
from scipy.optimize import NonlinearConstraint, minimize
from scipy.stats import pearsonr

from src.hgr import KernelsHGR

DEGREE: int = 7
"""Default degree for kernel-based metrics."""

LASSO: float = 0.0
"""Default lasso amount for kernel-based metrics."""

TOL: float = 1e-2
"""The tolerance used when checking for linear dependencies."""

EPS: float = 0.0
"""The tolerance used to account for null standard deviation."""


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class KernelBasedHGR(KernelsHGR):
    """Kernel-based HGR interface."""

    lasso: float = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=LASSO)
    """The amount of lasso penalization."""

    guess: Optional[np.ndarray] = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=None)
    """The initial guess for the problem."""

    @property
    @abstractmethod
    def degree_a(self) -> int:
        """The kernel degree for the first variable."""
        pass

    @property
    @abstractmethod
    def degree_b(self) -> int:
        """The kernel degree for the first variable."""
        pass

    def _kernels(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
        # center the kernels with respect to the training data
        a_ref = experiment.dataset.excluded(backend='numpy')
        b_ref = experiment.dataset.target(backend='numpy')
        f = np.stack([a ** d - np.mean(a_ref ** d) for d in np.arange(self.degree_a) + 1], axis=1)
        g = np.stack([b ** d - np.mean(b_ref ** d) for d in np.arange(self.degree_b) + 1], axis=1)
        return f @ experiment.result['alpha'], g @ experiment.result['beta']

    @staticmethod
    def kernel(v, degree: int, use_torch: bool):
        """Computes the kernel of the given vector with the given degree and using either numpy or torch as backend."""
        if use_torch:
            return torch.stack([v ** d - torch.mean(v ** d) for d in np.arange(degree) + 1], dim=1)
        else:
            return np.stack([v ** d - np.mean(v ** d) for d in np.arange(degree) + 1], axis=1)

    @staticmethod
    def _get_linearly_independent(f: np.ndarray, g: np.ndarray) -> Tuple[List[int], List[int]]:
        """Returns the list of indices of those columns that are linearly independent to other ones."""
        n, dx = f.shape
        _, dy = g.shape
        d = dx + dy
        # build a new matrix [ 1 | F_1 | G_1 | F_2 | G_2 | ... ]
        #   - this order is chosen so that lower grades are preferred in case of linear dependencies
        #   - the F and G indices are built depending on which kernel has the higher degree
        if dx < dy:
            f_indices = [2 * i + 1 for i in range(dx)]
            g_indices = [2 * i + 2 for i in range(dx)] + [i + 1 for i in range(2 * dx, d)]
        else:
            f_indices = [2 * i + 1 for i in range(dy)] + [i + 1 for i in range(2 * dy, d)]
            g_indices = [2 * i + 2 for i in range(dy)]
        fg_bias = np.ones((len(f), d + 1))
        fg_bias[:, f_indices] = f
        fg_bias[:, g_indices] = g
        # compute the QR factorization and retrieve the R matrix
        #   - get the diagonal of R
        #   - if a null value is found, it means that the respective column is linearly dependent to other columns
        # noinspection PyUnresolvedReferences
        r = scipy.linalg.qr(fg_bias, mode='r')[0]
        r = np.abs(np.diag(r))
        # eventually, retrieve the indices to be set to zero:
        #   - create a range going from 0 to degree - 1
        #   - mask it by selecting all those value in the diagonal that are smaller than the tolerance
        #   - finally exclude the first value in both cases since their linear dependence might be caused by a
        #      deterministic dependency in the data which we don't want to exclude
        f_indices = np.arange(dx)[r[f_indices] <= TOL][1:]
        g_indices = np.arange(dy)[r[g_indices] <= TOL][1:]
        return ([idx for idx in range(dx) if idx not in f_indices],
                [idx for idx in range(dy) if idx not in g_indices])

    def _higher_order_coefficients(self, f: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the kernel-based hgr for higher order degrees."""
        # retrieve the indices of the linearly dependent columns and impose a linear constraint so that the respective
        # weight is null for all but the first one (this processing step allow to avoid degenerate cases when the
        # matrix is not full rank)
        f_indices, g_indices = KernelBasedHGR._get_linearly_independent(f=f, g=g)
        f_slim = f[:, f_indices]
        g_slim = g[:, g_indices]
        n, dx = f_slim.shape
        _, dy = g_slim.shape
        d = dx + dy
        fg = np.concatenate((f_slim, -g_slim), axis=1)

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
        def _fun(inp):
            alp, bet = inp[:dx], inp[dx:]
            diff = f_slim @ alp - g_slim @ bet
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
        cst_hess[dx:, dx:] = 2 * g_slim.T @ g_slim / n
        constraint = NonlinearConstraint(
            fun=lambda inp: np.var(g_slim @ inp[dx:], ddof=0),
            jac=lambda inp: np.concatenate(([0] * dx, 2 * g_slim.T @ g_slim @ inp[dx:] / n)),
            hess=lambda *_: cst_hess,
            lb=1,
            ub=1
        )
        # if no guess is provided, set the initial point as [ 1 / std(F @ 1) | 1 / std(G @ 1) ] then solve the problem
        if self.guess is None:
            # TODO: check variability if np.random is used
            alp0 = np.ones(dx) / f_slim.sum(axis=1).std(ddof=0)
            bet0 = np.ones(dy) / g_slim.sum(axis=1).std(ddof=0)
            x0 = np.concatenate((alp0, bet0))
        else:
            x0 = self.guess
        s = minimize(_fun, jac=True, hess=lambda *_: fun_hess, x0=x0, constraints=[constraint], method='trust-constr')
        # reconstruct alpha and beta by adding zeros wherever the indices were not considered
        alpha = np.zeros(self.degree_a)
        alpha[f_indices] = s.x[:dx]
        beta = np.zeros(self.degree_b)
        beta[g_indices] = s.x[dx:]
        return alpha, beta

    def _compute_numpy(self,
                       a: np.ndarray,
                       b: np.ndarray,
                       degree_a: int,
                       degree_b: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """Computes HGR using numpy as backend and returns the correlation along with alpha and beta."""
        # build the kernel matrices
        f = KernelBasedHGR.kernel(a, degree=degree_a, use_torch=False)
        g = KernelBasedHGR.kernel(b, degree=degree_b, use_torch=False)
        # handle trivial or simpler cases:
        #  - if both degrees are 1, simply compute the projected vectors as standardized original vectors
        #  - if one degree is 1, standardize that vector and compute the other's coefficients using lstsq
        #  - if no degree is 1, use the optimization routine and compute the projected vectors from the coefficients
        if degree_a == 1 and degree_b == 1:
            alpha, beta = np.ones(1), np.ones(1)
        elif degree_a == 1:
            std = a.std(ddof=0) + EPS
            alpha = np.ones(1) / std
            beta, _, _, _ = np.linalg.lstsq(g, f[:, 0] / std, rcond=None)
        elif degree_b == 1:
            std = b.std(ddof=0) + EPS
            beta = np.ones(1) / std
            alpha, _, _, _ = np.linalg.lstsq(f, g[:, 0] / std, rcond=None)
        else:
            alpha, beta = self._higher_order_coefficients(f=f, g=g)
        fa = f @ alpha
        gb = g @ beta
        correlation, _ = pearsonr(fa, gb)
        alpha = alpha / (fa.std(ddof=0) + EPS)
        beta = beta / (gb.std(ddof=0) + EPS)
        return abs(float(correlation)), alpha, beta

    def _compute_torch(self, a: torch.Tensor, b: torch.Tensor, degree_a: int, degree_b: int) -> torch.Tensor:
        """Computes HGR using numpy as backend and returns the correlation (without alpha and beta)."""

        def standardize(t: torch.Tensor) -> torch.Tensor:
            t_std, t_mean = torch.std_mean(t, correction=0)
            return (t - t_mean) / (t_std + EPS)

        # build the kernel matrices
        f = KernelBasedHGR.kernel(a, degree=degree_a, use_torch=True)
        g = KernelBasedHGR.kernel(b, degree=degree_b, use_torch=True)
        # handle trivial or simpler cases:
        #  - if both degrees are 1, simply compute the projected vectors as standardized original vectors
        #  - if one degree is 1, standardize that vector and compute the other's coefficients using lstsq
        #  - if no degree is 1, use the optimization routine and compute the projected vectors from the coefficients
        if degree_a == 1 and degree_b == 1:
            fa = standardize(a)
            gb = standardize(b)
        elif degree_a == 1:
            # the 'gelsd' driver allows to have both more precise and more reproducible results
            fa = standardize(a)
            beta, _, _, _ = torch.linalg.lstsq(g, fa, driver='gelsd')
            gb = standardize(g @ beta)
        elif degree_b == 1:
            # the 'gelsd' driver allows to have both more precise and more reproducible results
            gb = standardize(b)
            alpha, _, _, _ = torch.linalg.lstsq(f, gb, driver='gelsd')
            fa = standardize(f @ alpha)
        else:
            alpha, beta = self._higher_order_coefficients(f=f.numpy(force=True), g=g.numpy(force=True))
            fa = standardize(f @ torch.tensor(alpha, dtype=f.dtype))
            gb = standardize(g @ torch.tensor(beta, dtype=g.dtype))
        # return the correlation as the absolute value of the vector product (since the vectors are standardized)
        return torch.abs(fa @ gb)


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class DoubleKernelHGR(KernelBasedHGR):
    """Kernel-based HGR computed by solving a constrained least square problem using a minimization solver."""

    degree_a: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=DEGREE)
    degree_b: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=DEGREE)

    @property
    def name(self) -> str:
        return 'kb'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, degree_a=self.degree_a, degree_b=self.degree_b, lasso=self.lasso, guess=self.guess)

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        hgr, alpha, beta = self._compute_numpy(a=a, b=b, degree_a=self.degree_a, degree_b=self.degree_b)
        return float(hgr), dict(alpha=alpha, beta=beta)

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._compute_torch(a=a, b=b, degree_a=self.degree_a, degree_b=self.degree_b)


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class SingleKernelHGR(KernelBasedHGR):
    """Kernel-based HGR computed using one kernel only for both variables and then taking the maximal correlation."""

    degree: int = field(init=True, repr=True, compare=False, hash=None, kw_only=True, default=DEGREE)
    """The kernel degree for the variables."""

    @property
    def name(self) -> str:
        return 'sk'

    @property
    def degree_a(self) -> int:
        return self.degree

    @property
    def degree_b(self) -> int:
        return self.degree

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, degree=self.degree, lasso=self.lasso, guess=self.guess)

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        # compute single-kernel correlations along with kernels
        hgr_a, alpha_a, beta_a = self._compute_numpy(a=a, b=b, degree_a=self.degree, degree_b=1)
        hgr_b, alpha_b, beta_b = self._compute_numpy(a=a, b=b, degree_a=1, degree_b=self.degree)
        # choose the best correlation and return
        if hgr_a > hgr_b:
            hgr = hgr_a
            alpha = alpha_a
            beta = np.zeros(self.degree)
            beta[0] = beta_a
        else:
            hgr = hgr_b
            alpha = np.zeros(self.degree)
            alpha[0] = alpha_b
            beta = beta_b
        return float(hgr), dict(alpha=alpha, beta=beta)

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # compute single-kernel correlations
        hgr_a = self._compute_torch(a=a, b=b, degree_a=self.degree, degree_b=1)
        hgr_b = self._compute_torch(a=a, b=b, degree_a=1, degree_b=self.degree)
        # return the maximal correlation
        return torch.maximum(hgr_a, hgr_b)

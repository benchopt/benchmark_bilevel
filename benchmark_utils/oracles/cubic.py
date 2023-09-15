import jax
import numpy as np
import jax.numpy as jnp
from joblib import Memory

from .base import BaseOracle


import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

memory = Memory("__cache__",
                verbose=False)


@memory.cache
def gen_matrices(n_samples, dim, L_inner, L_outer, mu, seed,
                 low_rank_outer=False):
    rng = np.random.RandomState(seed)

    hess_inner = []
    for _ in range(n_samples):
        A = rng.randn(dim, dim)
        A = A.T @ A
        _, U = np.linalg.eigh(A)
        D = np.logspace(np.log10(mu), np.log10(L_inner), dim)
        D = np.diag(D)
        A = U @ D @ U.T
        hess_inner.append(A)

    hess_outer = []
    for _ in range(n_samples):
        if low_rank_outer:
            x = rng.randn(dim)
            hess_outer.append(x[:, None] @ x[None, :])
        else:
            A = rng.randn(dim, dim)
            A = A.T @ A
            _, U = np.linalg.eigh(A)
            D = np.logspace(np.log10(mu), np.log10(L_outer), dim-1)
            D = np.diag(np.r_[0, D])
            A = U @ D @ U.T
            hess_outer.append(A)

    return (
        np.stack(hess_inner), np.stack(hess_outer),
        rng.randn(n_samples, dim, dim),
        rng.randn(n_samples, dim),
        rng.randn(n_samples, dim)
    )


def cubic(inner_var, outer_var, hess_inner, hess_outer, cross,
          linear_inner):
    res = .5 * inner_var @ (hess_inner @ inner_var)
    res += .5 * outer_var @ (hess_outer @ outer_var)
    res += outer_var @ cross @ inner_var
    res += linear_inner @ inner_var
    res += .5 * (np.exp(outer_var) * inner_var ** 2).sum()
    return res


def batched_cubic(inner_var, outer_var, hess_inner, hess_outer, cross,
                  linear_inner):
    return jnp.mean(jax.vmap(cubic,
                             in_axes=(None, None, 0, 0, 0, 0)))


class CubicOracle(BaseOracle):
    """Class defining the oracles for the L^2 regularized logistic loss.

    **NOTE:** This class is taylored for the binary logreg.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data for the model.
    y : ndarray, shape (n_samples,)
        Targets for the logistic regression. Must be binary targets.
    reg : {'exp', ‘lin’, ‘none’}, default='none',
        Parametrization of the regularization parameter
        - 'exp' the parametrization is exponential
        - 'lin' the parametrization is linear
        - 'none' no regularization
    """
    def __init__(self, n_samples, dim, L_inner, L_outer, mu,
                 random_state=None):
        super().__init__()

        self.n_samples = n_samples
        self.L_inner = L_inner
        self.mu = mu
        self.L_outer = L_outer

        (self.hess_inner, self.hess_outer, self.cross_mat, self.linear_inner,
         self.linear_outer) = gen_matrices(
            n_samples, dim, L_inner, L_outer, mu, random_state
        )

        self.hess_inner_full = np.mean(self.hess_inner, axis=0)
        self.hess_outer_full = np.mean(self.hess_outer, axis=0)
        self.cross_mat_full = np.mean(self.cross_mat, axis=0)
        self.linear_inner_full = np.mean(self.linear_inner, axis=0)
        self.linear_outer_full = np.mean(self.linear_outer, axis=0)

        self.variables_shape = np.array(
            [[self.hess_inner.shape[1]], [self.hess_outer.shape[1]]]
        )

    def _get_jax_oracle(self, get_full_batch=False):
        raise NotImplementedError("No Numba implementation for quadratic  "
                                  + "oracle available")

    def _get_numba_oracle(self):
        raise NotImplementedError("No Numba implementation for quadratic  "
                                  + "oracle available")

    def value(self, inner_var, outer_var, idx):

        if isinstance(idx, slice) and idx == slice(0, self.n_samples):
            hess_inner = self.hess_inner_full
            hess_outer = self.hess_outer_full
            cross_mat = self.cross_mat_full
            linear_inner = self.linear_inner_full
        else:
            hess_inner = np.mean(self.hess_inner[idx], axis=0)
            hess_outer = np.mean(self.hess_outer[idx], axis=0)
            cross_mat = np.mean(self.cross_mat[idx], axis=0)
            linear_inner = np.mean(self.linear_inner[idx], axis=0)

        res = cubic(inner_var, outer_var, hess_inner, hess_outer,
                    cross_mat, linear_inner)
        return res

    def grad_inner_var(self, inner_var, outer_var, idx):

        if isinstance(idx, slice) and idx == slice(0, self.n_samples):
            hess_inner = self.hess_inner_full
            cross_mat = self.cross_mat_full
            linear_inner = self.linear_inner_full
        else:
            hess_inner = np.mean(self.hess_inner[idx], axis=0)
            cross_mat = np.mean(self.cross_mat[idx], axis=0)
            linear_inner = np.mean(self.linear_inner[idx], axis=0)

        res = (hess_inner + np.diag(np.exp(outer_var))) @ inner_var
        res += cross_mat.T @ outer_var + linear_inner
        return res

    def grad_outer_var(self, inner_var, outer_var, idx):

        if isinstance(idx, slice) and idx == slice(0, self.n_samples):
            hess_outer = self.hess_outer_full
            cross_mat = self.cross_mat_full
        else:
            hess_outer = np.mean(self.hess_outer[idx], axis=0)
            cross_mat = np.mean(self.cross_mat[idx], axis=0)

        res = hess_outer @ outer_var + cross_mat @ inner_var
        res += .5 * (np.exp(outer_var) * inner_var ** 2)
        return res

    def cross(self, inner_var, outer_var, v, idx):

        if isinstance(idx, slice) and idx == slice(0, self.n_samples):
            cross_mat = self.cross_mat_full
        else:
            cross_mat = np.mean(self.cross_mat[idx], axis=0)

        return cross_mat @ v + np.exp(outer_var) * inner_var * v

    def hvp(self, inner_var, outer_var, v, idx):

        if isinstance(idx, slice) and idx == slice(0, self.n_samples):
            hess_inner = self.hess_inner_full
        else:
            hess_inner = np.mean(self.hess_inner[idx], axis=0)

        return (hess_inner + np.diag(np.exp(outer_var))) @ v

    def prox(self, inner_var, outer_var):
        return inner_var, outer_var

    def inverse_hvp(self, inner_var, outer_var, v, idx, approx='cg'):
        return v

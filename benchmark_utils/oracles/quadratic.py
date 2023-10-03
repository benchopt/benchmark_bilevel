import jax
import numpy as np
import jax.numpy as jnp
from joblib import Memory
from functools import partial

from .base import BaseOracle


import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

memory = Memory("__cache__",
                verbose=False)


@memory.cache
def gen_matrices(n_samples, d_inner, d_outer, L_inner, L_outer, L_cross, mu,
                 seed):
    rng = np.random.RandomState(seed)

    # Generate H^1/2
    U, *_ = np.linalg.svd(rng.randn(d_inner, 1))
    D = np.logspace(np.log10(mu) / 2, np.log10(L_inner)/2, d_inner)
    D = np.diag(D)
    A = U @ D @ U.T

    # Generate x with correlation matrix H (spectrum [mu, 1])
    # and take H_i as empirical correlation x.x^T
    X = rng.randn(n_samples, 1, d_inner) @ A
    hess_inner = X.transpose(0, 2, 1) @ X / X.shape[1]

    # Generate H^1/2
    U, *_ = np.linalg.svd(rng.randn(d_outer, 1))
    D = np.logspace(np.log10(mu)/2, np.log10(L_outer)/2, d_outer-1)
    D = np.diag(np.r_[0, D])
    A = U @ D @ U.T

    # Generate x with correlation matrix H (spectrum [mu, 1])
    # and take H_i as empirical correlation x.x^T
    X = rng.randn(n_samples, 1, d_outer) @ A
    hess_outer = X.transpose(0, 2, 1) @ X / X.shape[1]

    # Generate singular vectors and values for rectangular correlation matrix
    U, _, V = np.linalg.svd(np.random.randn(d_outer, d_inner))
    D = np.zeros((d_outer, d_inner))
    D[:min(d_outer, d_inner), :min(d_outer, d_inner)] = np.diag(
        np.logspace(np.log10(mu), np.log10(L_cross), min(d_outer, d_inner)))

    # Generate x with correlation matrix UDV^T
    X = rng.randn(n_samples, 1, d_inner)
    X1 = U @ D @ X.transpose(0, 2, 1)
    X2 = X @ V.T

    cross_mat = X1 @ X2

    return (
        hess_inner, hess_outer,
        cross_mat,
        rng.randn(n_samples, d_inner),
        rng.randn(n_samples, d_outer)
    )


def quadratic(inner_var, outer_var, hess_inner, hess_outer, cross,
              linear_inner, linear_outer):
    res = .5 * inner_var @ (hess_inner @ inner_var)
    res += .5 * outer_var @ (hess_outer @ outer_var)
    res += outer_var @ cross @ inner_var
    res += linear_inner @ inner_var
    res += linear_outer @ outer_var
    return res


def batched_quadratic(inner_var, outer_var, hess_inner, hess_outer, cross,
                      linear_inner, linear_outer):
    batched_loss = jax.vmap(quadratic, in_axes=(None, None, 0, 0, 0, 0, 0))
    return jnp.mean(
        batched_loss(inner_var, outer_var, hess_inner, hess_outer,
                     cross, linear_inner, linear_outer)
    )


class QuadraticOracle(BaseOracle):
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
    def __init__(self, n_samples, d_inner, d_outer, L_inner, L_outer, L_cross,
                 mu, random_state=None):
        super().__init__()

        self.n_samples = n_samples
        self.L_inner = L_inner
        self.L_cross = L_cross
        self.mu = mu
        self.L_outer = L_outer

        (self.hess_inner, self.hess_outer, self.cross_mat, self.linear_inner,
         self.linear_outer) = gen_matrices(
            n_samples, d_inner, d_outer, L_inner, L_outer, L_cross, mu,
            random_state,
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

        @partial(jax.jit, static_argnames=('batch_size'))
        def jax_oracle(inner_var, outer_var, start=0, batch_size=1):
            hess_inner = jax.lax.dynamic_slice(
                self.hess_inner, (start, 0, 0),
                (batch_size, self.hess_inner.shape[1],
                 self.hess_inner.shape[2])
            )
            hess_outer = jax.lax.dynamic_slice(
                self.hess_outer, (start, 0, 0),
                (batch_size, self.hess_outer.shape[1],
                 self.hess_outer.shape[2])
            )
            cross_mat = jax.lax.dynamic_slice(
                self.cross_mat, (start, 0, 0),
                (batch_size, self.cross_mat.shape[1], self.cross_mat.shape[2])
            )
            linear_inner = jax.lax.dynamic_slice(
                self.linear_inner, (start, 0),
                (batch_size, self.linear_inner.shape[1])
            )
            linear_outer = jax.lax.dynamic_slice(
                self.linear_outer, (start, 0),
                (batch_size, self.linear_outer.shape[1])
            )
            return batched_quadratic(inner_var, outer_var, hess_inner,
                                     hess_outer, cross_mat, linear_inner,
                                     linear_outer)
        if get_full_batch:
            @jax.jit
            def jax_oracle_fb(inner_var, outer_var):
                return quadratic(inner_var, outer_var, self.hess_inner_full,
                                 self.hess_outer_full, self.cross_mat_full,
                                 self.linear_inner_full,
                                 self.linear_outer_full)
            return jax_oracle, jax_oracle_fb
        else:
            return jax_oracle

    def _get_numba_oracle(self):
        raise NotImplementedError("No Numba implementation for quadratic  "
                                  + "oracle available")

    def value(self, inner_var, outer_var, idx):

        if isinstance(idx, slice) and idx == slice(0, self.n_samples):
            hess_inner = self.hess_inner_full
            hess_outer = self.hess_outer_full
            cross_mat = self.cross_mat_full
            linear_inner = self.linear_inner_full
            linear_outer = self.linear_outer_full
        else:
            hess_inner = np.mean(self.hess_inner[idx], axis=0)
            hess_outer = np.mean(self.hess_outer[idx], axis=0)
            cross_mat = np.mean(self.cross_mat[idx], axis=0)
            linear_inner = np.mean(self.linear_inner[idx], axis=0)
            linear_outer = np.mean(self.linear_outer[idx], axis=0)

        res = quadratic(inner_var, outer_var, hess_inner, hess_outer,
                        cross_mat, linear_inner, linear_outer)
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

        res = hess_inner @ inner_var + cross_mat.T @ outer_var + linear_inner
        return res

    def grad_outer_var(self, inner_var, outer_var, idx):

        if isinstance(idx, slice) and idx == slice(0, self.n_samples):
            hess_outer = self.hess_outer_full
            cross_mat = self.cross_mat_full
            linear_outer = self.linear_outer_full
        else:
            hess_outer = np.mean(self.hess_outer[idx], axis=0)
            cross_mat = np.mean(self.cross_mat[idx], axis=0)
            linear_outer = np.mean(self.linear_outer[idx], axis=0)

        res = hess_outer @ outer_var + cross_mat @ inner_var + linear_outer
        return res

    def cross(self, inner_var, outer_var, v, idx):

        if isinstance(idx, slice) and idx == slice(0, self.n_samples):
            cross_mat = self.cross_mat_full
        else:
            cross_mat = np.mean(self.cross_mat[idx], axis=0)

        return cross_mat @ v

    def hvp(self, inner_var, outer_var, v, idx):

        if isinstance(idx, slice) and idx == slice(0, self.n_samples):
            hess_inner = self.hess_inner_full
        else:
            hess_inner = np.mean(self.hess_inner[idx], axis=0)

        return hess_inner @ v

    def prox(self, inner_var, outer_var):
        return inner_var, outer_var

    def inverse_hvp(self, inner_var, outer_var, v, idx, approx='cg'):
        return v

    def inner_var_star(self, outer_var):
        return np.linalg.solve(
                self.hess_inner_full,
                - self.linear_inner_full - self.cross_mat_full.T @ outer_var
            )

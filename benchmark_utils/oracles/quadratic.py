import numpy as np
import jax
import jax.numpy as jnp

from .base import BaseOracle

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)


def quadratic(inner_var, outer_var, hess_inner, hess_outer, cross,
              linear_inner):
    res = .5 * jnp.dot(inner_var, hess_inner @ inner_var)
    res += .5 * jnp.dot(outer_var, hess_outer @ outer_var)
    res += outer_var @ jnp.dot(cross, inner_var)
    res += jnp.dot(linear_inner, inner_var)
    return res


def batched_quadratic(inner_var, outer_var, hess_inner, hess_outer, cross,
                      linear_inner):
    return jnp.mean(jax.vmap(quadratic,
                             in_axes=(None, None, 0, 0, 0, 0)))


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
    def __init__(self, hess_inner, hess_outer, cross, linear_inner,
                 L_inner, mu, L_outer):
        super().__init__()

        self.hess_inner = np.stack(hess_inner)
        self.hess_outer = np.stack(hess_outer)
        self.cross_mat = np.stack(cross)
        self.linear_inner = np.stack(linear_inner)
        self.L_inner = L_inner
        self.mu = mu
        self.L_outer = L_outer

        self.variables_shape = (self.hess_inner.shape[1],
                                self.hess_outer.shape[1])
        self.n_samples = self.hess_inner.shape[0]

    def _get_jax_oracle(self, get_full_batch=False):
        raise NotImplementedError("No Numba implementation for quadratic  "
                                  + "oracle available")

    def _get_numba_oracle(self):
        raise NotImplementedError("No Numba implementation for quadratic  "
                                  + "oracle available")

    def value(self, inner_var, outer_var, idx):
        hess_inner = np.mean(self.hess_inner[idx], axis=0)
        hess_outer = np.mean(self.hess_outer[idx], axis=0)
        cross_mat = np.mean(self.cross_mat[idx], axis=0)
        linear_inner = np.mean(self.linear_inner[idx], axis=0)

        res = quadratic(inner_var, outer_var, hess_inner, hess_outer,
                        cross_mat, linear_inner)
        return res

    def grad_inner_var(self, inner_var, outer_var, idx):
        hess_inner = np.mean(self.hess_inner[idx], axis=0)
        cross_mat = np.mean(self.cross_mat[idx], axis=0)
        linear_inner = np.mean(self.linear_inner[idx], axis=0)
        res = hess_inner @ inner_var + cross_mat.T @ outer_var + linear_inner
        return res

    def grad_outer_var(self, inner_var, outer_var, idx):
        hess_outer = np.mean(self.hess_outer[idx], axis=0)
        cross_mat = np.mean(self.cross_mat[idx], axis=0)
        res = hess_outer @ outer_var + cross_mat @ inner_var
        return res

    def cross(self, inner_var, outer_var, v, idx):
        cross_mat = np.mean(self.cross_mat[idx], axis=0)
        return cross_mat @ v

    def hvp(self, inner_var, outer_var, v, idx):
        hess_inner = np.mean(self.hess_inner[idx], axis=0)
        return hess_inner @ v

    def prox(self, inner_var, outer_var):
        return inner_var, outer_var

    def inverse_hvp(self, inner_var, outer_var, v, idx, approx='cg'):
        return v

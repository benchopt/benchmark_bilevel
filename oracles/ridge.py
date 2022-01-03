import numpy as np
from scipy.sparse.linalg import svds

# from numba import generated_jit
from numba import float64, int64, types    # import the types
from numba.experimental import jitclass

from .base import BaseOracle


# @generated_jit
# def ls_grad_theta(x, theta, y):
#     print(x)
#     if isinstance(x, types.Array) and x.ndim == 1:
#         assert False
#         return lambda x, theta, y: (x @ theta - y) * x
#     if isinstance(x, types.Array) and x.ndim == 2:
#         return lambda x, theta, y:


# @generated_jit
# def ls_hvp(x, v):
#     if isinstance(x, types.Array) and x.ndim == 1:
#         assert False
#         return lambda x, v: (x @ v) * x
#     if isinstance(x, types.Array) and x.ndim == 2:
#         return lambda x, v: x.T @ (x @ v) / x.shape[0]


spec = [
    ('X', float64[:, ::1]),          # an array field
    ('y', float64[::1]),               # a simple scalar field
    ('reg', types.unicode_type),
    ('n_samples', int64),
    ('n_features', int64),
    # ('variables_shape', int64[:, :]),
]


@jitclass(spec)
class RidgeRegressionOracleNumba():
    """Class defining the oracles for the L^2 regularized least squares
    loss."""

    def __init__(self, X, y, reg):

        self.X = X
        self.y = y
        self.reg = reg

        # attributes
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        # self.variables_shape = np.array([
        #     [self.n_features], [self.n_features]
        # ])

    def value(self, theta, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]
        n_samples = x.shape[0]  # if x.ndim == 2 else 1
        res = x @ theta - y
        tmp = 0.5 / n_samples * (res @ res)
        if self.reg == 'exp':
            tmp += .5 * theta.dot(np.exp(lmbda) * theta)
        elif self.reg == 'lin':
            tmp += .5 * theta.dot(lmbda * theta)
        return tmp

    def grad_inner_var(self, theta, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]
        tmp = x.T @ (x @ theta - y) / x.shape[0]
        if self.reg == 'exp':
            tmp += np.exp(lmbda) * theta
        elif self.reg == 'lin':
            tmp += lmbda * theta
        return tmp

    def grad_outer_var(self, theta, lmbda, idx):
        if self.reg == 'exp':
            grad = .5 * np.exp(lmbda) * theta ** 2
        elif self.reg == 'lin':
            grad = .5 * theta ** 2
        else:
            grad = np.zeros_like(lmbda)
        return grad

    def cross(self, theta, lmbda, v, idx):
        if self.reg == 'exp':
            res = np.exp(lmbda) * theta * v
        elif self.reg == 'lin':
            res = theta * v
        else:
            res = np.zeros_like(lmbda)
        return res

    def hvp(self, theta, lmbda, v, idx):
        x = self.X[idx]
        tmp = x.T @ (x @ v) / x.shape[0]
        if self.reg == 'exp':
            tmp += np.exp(lmbda) * v
        elif self.reg == 'lin':
            tmp += lmbda * v
        return tmp

    def inverse_hvp(self, theta, lmbda, v, idx, approx='cg'):
        if approx == 'id':
            return v
        if approx != 'cg':
            raise NotImplementedError
        assert idx.ndim == 1
        x = self.X[idx]
        H = np.dot(x.T, x) / x.shape[0]
        if self.reg == 'exp':
            H += np.diag(np.exp(lmbda))
        elif self.reg == 'lin':
            H += np.diag(lmbda)
        return np.linalg.solve(H, v)

    def inner_var_star(self, lmbda, idx):
        assert idx.ndim == 1
        x = self.X[idx]
        y = self.y[idx]
        n_samples = x.shape[0]
        b = x.T.dot(y) / n_samples
        H = x.T.dot(x) / n_samples
        if self.reg == 'exp':
            H += np.diag(np.exp(lmbda))
        elif self.reg == 'lin':
            H += np.diag(lmbda)
        return np.linalg.solve(H, b)

    def oracles(self, theta, lmbda, v, idx, inverse='id'):
        """Returns the value, the gradient,
        """
        x = self.X[idx]
        y = self.y[idx]
        n_samples = x.shape[0]
        residual = (x @ theta - y)
        val = 0.5 / n_samples * (residual @ residual)
        grad = x.T @ (residual / n_samples)
        hvp = x.T @ (x @ v) / n_samples
        inv_hvp = self.inverse_hvp(theta, lmbda, v, idx, approx=inverse)
        if self.reg:
            reg = np.exp(lmbda) / self.n_features
            val += .5 * (reg @ theta ** 2)
            grad += reg * theta
            hvp += reg * v

        return val, grad, hvp, self.cross(theta, lmbda, inv_hvp, idx)

    def prox(self, theta, lmbda):
        if self.reg == 'exp':
            lmbda[lmbda < -12] = -12
            lmbda[lmbda > 12] = 12
        elif self.reg == 'lin':
            lmbda = np.maximum(lmbda, 0)
        return theta, lmbda


class RidgeRegressionOracle(BaseOracle):
    """Class defining the oracles for the L^2 regularized least squares loss.
    """
    def __init__(self, X, y, reg=False):
        super().__init__()

        self.numba_oracle = RidgeRegressionOracleNumba(X, y, reg)

        # attributes
        self.n_samples, self.n_features = X.shape
        self.variables_shape = np.array([
            [self.n_features], [self.n_features]
        ])

    def value(self, theta, lmbda, idx):
        return self.numba_oracle.value(theta, lmbda, idx)

    def grad_inner_var(self, theta, lmbda, idx):
        return self.numba_oracle.grad_inner_var(theta, lmbda, idx)

    def grad_outer_var(self, theta, lmbda, idx):
        return self.numba_oracle.grad_outer_var(theta, lmbda, idx)

    def cross(self, theta, lmbda, idx):
        return self.numba_oracle.cross(theta, lmbda, idx)

    def hvp(self, theta, lmbda, idx):
        return self.numba_oracle.hvp(theta, lmbda, idx)

    def inverse_hvp(self, theta, lmbda, idx):
        return self.numba_oracle.inverse_hvp(theta, lmbda, idx)

    def inner_var_star(self, lmbda, idx):
        return self.numba_oracle.inner_var_star(lmbda, idx)

    def prox(self, theta, lmbda):
        return self.numba_oracle.prox(theta, lmbda)

    def lipschitz_inner(self, inner_var, outer_var):
        H = np.dot(self.X.T, self.X) / self.X.shape[0]
        if self.reg == 'exp':
            H += np.diag(np.exp(outer_var))
        elif self.reg == 'lin':
            H += np.diag(outer_var)
        return svds(H, k=1, return_singular_vectors=False)

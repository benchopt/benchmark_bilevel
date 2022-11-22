import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import svds

# from numba import generated_jit
from numba import float64, int64, types    # import the types
from numba.experimental import jitclass

from .base import BaseOracle


spec = [
    ('X', float64[:, ::1]),          # an array field
    ('y', float64[::1]),               # a simple scalar field
    ('reg', types.unicode_type),
    ('n_samples', int64),
    ('n_features', int64),
]


@jitclass(spec)
class RidgeRegressionOracleNumba():
    """Class defining the oracles for the L^2 regularized least squares
    loss."""

    def __init__(self, X, y, reg):

        self.X = X
        if not issparse(self.X):
            self.X = np.ascontiguousarray(X)
        self.y = y
        self.reg = reg

        # attributes
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

    def set_order(self, idx):
        self.X[:] = self.X[idx]
        self.y[:] = self.y[idx]

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
        if lmbda.shape[0] == 1:
            grad = grad.sum() * np.ones((1,))
        return grad

    def grad(self, theta, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]
        grad_theta = x.T @ (x @ theta - y) / x.shape[0]
        if self.reg == 'exp':
            alpha = np.exp(lmbda)
            grad_theta += alpha * theta
            grad_lmbda = .5 * alpha * theta ** 2
        elif self.reg == 'lin':
            grad_theta += lmbda * theta
            grad_lmbda = .5 * theta ** 2
        else:
            grad_lmbda = np.zeros_like(lmbda)
        if lmbda.shape[0] == 1:
            grad_lmbda = grad_lmbda.sum() * np.ones((1,))
        return grad_theta, grad_lmbda

    def cross(self, theta, lmbda, v, idx):
        if self.reg == 'exp':
            res = np.exp(lmbda) * theta * v
        elif self.reg == 'lin':
            res = theta * v
        else:
            res = np.zeros_like(lmbda)
        if lmbda.shape[0] == 1:
            res = res.sum() * np.ones((1,))
        return res

    def hvp(self, theta, lmbda, v, idx):
        x = self.X[idx]
        tmp = x.T @ (x @ v) / x.shape[0]
        if self.reg == 'exp':
            tmp += np.exp(lmbda) * v
        elif self.reg == 'lin':
            tmp += lmbda * v
        return tmp

    def inverse_hvp(self, theta, lmbda, v, idx, approx):
        if approx == 'id':
            return v
        if approx != 'cg':
            raise NotImplementedError
        x = self.X[idx]
        assert x.ndim == 2
        H = np.dot(x.T, x) / x.shape[0]
        if self.reg != 'none':
            alpha = np.exp(lmbda) if self.reg == 'exp' else lmbda
            if lmbda.shape[0] == 1:
                H += alpha * np.eye(H.shape[0])
            else:
                H += np.diag(alpha)
        return np.linalg.solve(H, v)

    def inner_var_star(self, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]
        assert x.ndim == 2
        n_samples = x.shape[0]
        b = x.T.dot(y) / n_samples
        H = x.T.dot(x) / n_samples
        if self.reg != 'none':
            alpha = np.exp(lmbda) if self.reg == 'exp' else lmbda
            if lmbda.shape[0] == 1:
                H += alpha * np.eye(H.shape[0])
            else:
                H += np.diag(alpha)
        return np.linalg.solve(H, b)

    def oracles(self, theta, lmbda, v, idx, inverse):
        """Returns the value, the gradient,
        """
        x = self.X[idx]
        y = self.y[idx]
        n_samples = x.shape[0]
        residual = (x @ theta - y)
        val = 0.5 / n_samples * (residual @ residual)
        grad = x.T @ (residual / n_samples)
        hvp = x.T @ (x @ v) / n_samples
        if self.reg != 'none':
            alpha = np.exp(lmbda) if self.reg == 'exp' else lmbda
            val += .5 * (theta @ (alpha * theta))
            grad += alpha * theta
            hvp += alpha * v

        inv_hvp = self.inverse_hvp(theta, lmbda, v, idx, inverse)

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
    def __init__(self, X, y, reg='none'):
        super().__init__()

        # Make sure reg is valid
        assert reg in ['exp', 'lin', 'none'], f"Unknown value for reg: '{reg}'"

        # Store info for other
        self.X = np.ascontiguousarray(X)
        self.y = y.astype(np.float64)
        self.reg = reg

        # Create a numba oracle for the numba functions
        self.numba_oracle = RidgeRegressionOracleNumba(
            self.X, self.y, self.reg
        )

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

    def grad(self, theta, lmbda, idx):
        return self.numba_oracle.grad(theta, lmbda, idx)

    def cross(self, theta, lmbda, v, idx):
        return self.numba_oracle.cross(theta, lmbda, v, idx)

    def hvp(self, theta, lmbda, v, idx):
        return self.numba_oracle.hvp(theta, lmbda, v, idx)

    def inverse_hvp(self, theta, lmbda, v, idx, approx='cg'):
        return self.numba_oracle.inverse_hvp(
            theta, lmbda, v, idx, approx
        )

    def inner_var_star(self, lmbda, idx):
        return self.numba_oracle.inner_var_star(lmbda, idx)

    def prox(self, theta, lmbda):
        return self.numba_oracle.prox(theta, lmbda)

    def oracles(self, theta, lmbda, v, idx, inverse='id'):
        return self.numba_oracle.oracles(theta, lmbda, v, idx, inverse)

    def lipschitz_inner(self, inner_var, outer_var):
        H = np.dot(self.X.T, self.X) / self.X.shape[0]
        if self.reg == 'exp':
            H += np.diag(np.exp(outer_var))
        elif self.reg == 'lin':
            H += np.diag(outer_var)
        return svds(H, k=1, return_singular_vectors=False)

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import linalg as splinalg

from numba import njit
from numba import float64, int64, types    # import the types
from numba.experimental import jitclass

import jax
import jax.numpy as jnp
from functools import partial
from jax.nn import log_sigmoid

from .base import BaseOracle
from .special import expit, logsig, expit_njit, logsig_njit

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)


def grad_theta_log_loss(x, y, theta):
    """Returns the gradient of the logistic loss."""
    n_samples, n_features = x.shape
    tmp = y * (x @ theta)
    tmp2 = expit(-tmp)

    grad = -(x.T @ (y * tmp2)) / n_samples

    return grad


@njit
def grad_theta_log_loss_njit(x, y, theta):
    """Returns the gradient of the logistic loss."""
    n_samples, n_features = x.shape
    tmp = y * (x @ theta)
    tmp2 = expit_njit(-tmp)

    grad = -(x.T @ (y * tmp2)) / n_samples

    return grad


def hvp_log_loss(x, y, theta, v):
    """Returns an hessian-vector product for the logistic loss and a vector v.
    """
    n_samples, n_features = x.shape
    tmp = np.zeros_like(y)
    tmp2 = y * (x @ theta)

    idx1 = tmp2 < 0
    tmp[idx1] = np.exp(tmp2[idx1]) * expit(- tmp2[idx1])**2
    idx2 = tmp2 >= 0
    tmp[idx2] = np.exp(- tmp2[idx2]) * expit(tmp2[idx2])**2

    xv = (x @ v)

    hvp = (x.T @ (xv * tmp)) / n_samples
    return hvp


@njit
def hvp_log_loss_njit(x, y, theta, v):
    """Returns an hessian-vector product for the logistic loss and a vector v.
    """
    n_samples, n_features = x.shape
    tmp = np.zeros_like(y)
    tmp2 = y * (x @ theta)

    idx1 = tmp2 < 0
    tmp[idx1] = np.exp(tmp2[idx1]) * expit_njit(- tmp2[idx1])**2
    idx2 = tmp2 >= 0
    tmp[idx2] = np.exp(- tmp2[idx2]) * expit_njit(tmp2[idx2])**2

    xv = (x @ v)

    hvp = (x.T @ (xv * tmp)) / n_samples
    return hvp


def value_grad_hvp_log_loss(x, y, theta, v):
    """Returns value, gradient, hessian-vector product for the logistic loss.
    """
    n_samples, n_features = x.shape
    tmp = y * safe_sparse_dot(x, theta)
    val = - logsig(tmp).mean()

    tmp2 = expit(-tmp)
    grad = -safe_sparse_dot(x.T, (y * tmp2)) / n_samples

    idx1 = tmp < 0
    tmp2[idx1] = np.exp(tmp[idx1]) * tmp2[idx1]**2
    idx2 = tmp >= 0
    tmp2[idx2] = np.exp(- tmp[idx2]) * expit(tmp[idx2])**2

    hvp = safe_sparse_dot(x.T, (x.dot(v) * tmp2)) / n_samples
    return val, grad, hvp


def _get_hvp_op(x, y, theta, reg, lmbda):
    n_samples, n_features = x.shape
    tmp = np.zeros_like(y)
    tmp2 = y * safe_sparse_dot(x, theta)
    assert tmp2.shape == y.shape

    idx1 = tmp2 < 0
    tmp[idx1] = np.exp(tmp2[idx1]) * expit(-tmp2[idx1])**2
    idx2 = tmp2 >= 0
    tmp[idx2] = np.exp(- tmp2[idx2]) * expit(tmp2[idx2])**2

    # Precompute as much as possible
    if sparse.issparse(x):
        tmp = sparse.dia_matrix((tmp, 0), shape=(n_samples, n_samples))
        dX = safe_sparse_dot(tmp, x)
    else:
        dX = tmp[:, np.newaxis] * x

    if reg == 'exp':
        alpha = np.exp(lmbda)
    elif reg == 'lin':
        alpha = lmbda

    def hvp(v):
        ret = np.empty_like(v)
        ret = x.T.dot(dX.dot(v) / n_samples)
        if reg != 'none':
            ret += alpha * v
        return ret

    Hop = splinalg.LinearOperator(
        shape=(n_features, n_features),
        matvec=lambda z: hvp(z),
        rmatvec=lambda z: hvp(z),
    )

    return Hop


@jax.jit
def jax_loss_sample(inner_var, outer_var, x, y):
    return -log_sigmoid(y*jnp.dot(inner_var, x))


@jax.jit
def jax_loss(theta, lmbda, X, y):
    batched_loss = jax.vmap(jax_loss_sample, in_axes=(None, None, 0, 0))
    return jnp.mean(batched_loss(theta, lmbda, X, y), axis=0)


spec = [
    ('X', float64[:, ::1]),          # an array field
    ('y', float64[::1]),               # a simple scalar field
    ('reg', types.unicode_type),
    ('n_samples', int64),
    ('n_features', int64),
    ("variables_shape", int64[:, ::1])
]


@jitclass(spec)
class LogisticRegressionOracleNumba():
    """Numba class defining the oracles for the L^2 regularized logistic loss.

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
    def __init__(self, X, y, reg='none'):

        self.X = X
        self.y = y
        self.reg = reg

        # attributes
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.variables_shape = np.array([
            [self.n_features], [self.n_features]
        ])

    def set_order(self, idx):
        self.X = self.X[idx]
        self.y = self.y[idx]

    def value(self, theta, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]
        tmp = - logsig_njit(y * (x @ theta)).mean()
        if self.reg == 'exp':
            tmp += .5 * theta.dot(np.exp(lmbda) * theta)
        elif self.reg == 'lin':
            tmp += .5 * theta.dot(lmbda * theta)
        return tmp

    def grad_inner_var(self, theta, lmbda, idx):
        tmp = grad_theta_log_loss_njit(self.X[idx], self.y[idx], theta)
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
        grad_theta = grad_theta_log_loss_njit(self.X[idx], self.y[idx], theta)
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
        tmp = hvp_log_loss_njit(self.X[idx], self.y[idx], theta, v)
        if self.reg == 'exp':
            tmp += np.exp(lmbda) * v
        elif self.reg == 'lin':
            tmp += lmbda * v
        return tmp

    def oracles(self, theta, lmbda, v, idx, inverse='id'):
        """Returns the value, the gradient,
        """
        x = self.X[idx]
        y = self.y[idx]
        n_samples = x.shape[0]
        tmp = y * (x @ theta)
        val = - logsig_njit(tmp).mean()

        tmp2 = expit_njit(-tmp)
        grad = -(x.T @ (y * tmp2)) / n_samples

        idx1 = tmp < 0
        tmp2[idx1] = np.exp(tmp[idx1]) * tmp2[idx1]**2
        idx2 = ~idx1
        tmp2[idx2] = np.exp(- tmp[idx2]) * expit_njit(tmp[idx2])**2

        hvp = (x.T @ ((x @ v) * tmp2)) / n_samples

        if self.reg != 'none':
            alpha = np.exp(lmbda) if self.reg == 'exp' else lmbda
            val += .5 * (theta @ (alpha * theta))
            grad += alpha * theta
            hvp += alpha * v

        if inverse == 'id':
            inv_hvp = v
        elif inverse == 'cg':
            H = x.T @ (tmp.reshape(-1, 1) * x)
            if self.reg != 'none':
                alpha = np.exp(lmbda) if self.reg == 'exp' else lmbda
                if lmbda.shape[0] == 1:
                    H += alpha * np.eye(H.shape[0])
                else:
                    H += np.diag(alpha)
            inv_hvp = np.linalg.solve(H, v)
        else:
            raise NotImplementedError('inverse unknown')

        return val, grad, hvp, self.cross(theta, lmbda, inv_hvp, idx)

    def prox(self, theta, lmbda):
        if self.reg == 'exp':
            lmbda[lmbda < -12] = -12
            lmbda[lmbda > 12] = 12
        elif self.reg == 'lin':
            lmbda = np.maximum(lmbda, 0)
        return theta, lmbda


class LogisticRegressionOracle(BaseOracle):
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
    def __init__(self, X, y, reg='none'):
        super().__init__()

        # Make sure the targets are {1, -1}.
        target_type = type_of_target(y)
        if target_type != 'binary':
            y = y > np.median(y)
        self.encoder = OrdinalEncoder()
        y = self.encoder.fit_transform(y[:, None]).flatten()
        y = 2 * y - 1
        assert set(y) == set([-1, 1])

        # Make sure reg is valid
        assert reg in ['exp', 'lin', 'none'], f"Unknown value for reg: '{reg}'"

        # Store info for other
        self.X = X
        self.y = y.astype(np.float64)
        self.reg = reg

        # attributes
        self.n_samples, self.n_features = X.shape
        self.variables_shape = np.array([
            [self.n_features], [self.n_features]
        ])

        # else:
        #     if not sparse.issparse(self.X):
        #         self.numba_oracle = LogisticRegressionOracleNumba(
        #             np.ascontiguousarray(self.X), self.y, self.reg
        #         )

    def _get_jax_oracle(self, get_full_batch=False):
        if sparse.issparse(self.X):
            raise ValueError("X should not be sparse")

        @partial(jax.jit, static_argnames=('batch_size'))
        def jax_oracle(inner_var, outer_var, start=0, batch_size=1):
            x = jax.lax.dynamic_slice(
                self.X, (start, 0),
                (batch_size, self.X.shape[1])
            )
            y = jax.lax.dynamic_slice(
                self.y, (start, ), (batch_size, ))
            res = jax_loss(inner_var, outer_var, x, y)
            if self.reg == 'exp':
                res += jnp.dot(jnp.exp(outer_var) * inner_var, inner_var)/2
            elif self.reg == 'lin':
                res += jnp.dot(outer_var * inner_var, inner_var)/2
            return res

        if get_full_batch:
            @jax.jit
            def jax_oracle_fb(inner_var, outer_var):
                res = jax_loss(inner_var, outer_var, self.X, self.y)
                if self.reg == 'exp':
                    res += jnp.dot(jnp.exp(outer_var) * inner_var, inner_var)/2
                elif self.reg == 'lin':
                    res += jnp.dot(outer_var * inner_var, inner_var)/2
                return res
            return jax_oracle, jax_oracle_fb
        else:
            return jax_oracle

    def _get_numba_oracle(self):
        if sparse.issparse(self.X):
            raise ValueError("X should not be sparse")
        return LogisticRegressionOracleNumba(
            np.ascontiguousarray(self.X), self.y, self.reg
        )

    def value(self, theta, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]
        tmp = - logsig(y * (x @ theta)).mean()
        if self.reg == 'exp':
            tmp += .5 * theta.dot(np.exp(lmbda) * theta)
        elif self.reg == 'lin':
            tmp += .5 * theta.dot(lmbda * theta)
        return tmp

    def grad_inner_var(self, theta, lmbda, idx):
        tmp = grad_theta_log_loss(self.X[idx], self.y[idx], theta)
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
        grad_theta = grad_theta_log_loss(self.X[idx], self.y[idx], theta)
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
        tmp = hvp_log_loss(self.X[idx], self.y[idx], theta, v)
        if self.reg == 'exp':
            tmp += np.exp(lmbda) * v
        elif self.reg == 'lin':
            tmp += lmbda * v
        return tmp

    def prox(self, theta, lmbda):
        if self.reg == 'exp':
            lmbda[lmbda < -12] = -12
            lmbda[lmbda > 12] = 12
        elif self.reg == 'lin':
            lmbda = np.maximum(lmbda, 0)
        return theta, lmbda

    def inverse_hvp(self, theta, lmbda, v, idx, approx='cg'):
        if approx == 'id':
            return v
        if approx != 'cg':
            raise NotImplementedError
        x_i = self.X[idx]
        y_i = self.y[idx]
        Hop = _get_hvp_op(x_i, y_i, theta, self.reg, lmbda)
        Hv, success = splinalg.cg(
            Hop, v,
            x0=v.copy(),
            tol=1e-8,
            maxiter=5000,
        )
        if success != 0:
            print('CG did not converge to the desired precision')
        return Hv

    def oracles(self, theta, lmbda, v, idx, inverse='id'):
        """Returns the value, the gradient,
        """
        val, grad, hvp = value_grad_hvp_log_loss(
            self.X[idx], self.y[idx], theta, v
        )
        inv_hvp = self.inverse_hvp(theta, lmbda, v, idx, approx=inverse)

        if self.reg != 'none':
            alpha = np.exp(lmbda) if self.reg == 'exp' else lmbda
            val += .5 * (theta @ (alpha * theta))
            grad += alpha * theta
            hvp += alpha * v

        return val, grad, hvp, self.cross(theta, lmbda, inv_hvp, idx)

    def lipschitz_inner(self, theta, lmbda):
        Hop = _get_hvp_op(self.X, self.y, theta, self.reg, lmbda)
        return svds(Hop, k=1, return_singular_vectors=False)

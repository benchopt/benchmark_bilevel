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

from .base import BaseOracle

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)


@njit
def logsig(x):
    """Computes the log-sigmoid function component-wise.

    Implemented as proposed in [1].

    [1] http://fa.bianp.net/blog/2019/evaluate_logistic/"""

    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])

    return out


@njit
def expit(t):
    """Computes the sigmoid function component-wise.

    Implemented as proposed in [1].

    [1] http://fa.bianp.net/blog/2019/evaluate_logistic/"""

    out = np.zeros_like(t)
    idx1 = t >= 0
    out[idx1] = 1 / (1 + np.exp(-t[idx1]))
    idx2 = t < 0
    tmp = np.exp(t[idx2])
    out[idx2] = tmp / (1 + tmp)
    return out


@njit
def grad_theta_log_loss(x, y, theta):
    """Returns the gradient of the logistic loss."""
    n_samples, n_features = x.shape
    tmp = y * (x @ theta)
    tmp2 = expit(-tmp)

    grad = -(x.T @ (y * tmp2)) / n_samples

    return grad


@njit
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


spec = [
    ('X', float64[:, ::1]),          # an array field
    ('y', float64[::1]),               # a simple scalar field
    ('reg', types.unicode_type),
    ('n_samples', int64),
    ('n_features', int64),
]


@jitclass(spec)
class LogisticRegressionOracleNumba():
    """Class defining the oracles for the L^2 regularized logistic loss.

    **NOTE:** This class is taylored for the binary logreg.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data for the model.
    y : ndarray, shape (n_samples,)
        Targets for the logistic regression. Must be binary targets.
    reg : bool
        Whether or not to apply regularisation.
    """
    def __init__(self, X, y, reg=False):

        self.X = X
        self.y = y
        self.reg = reg

        # attributes
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

    def set_order(self, idx):
        self.X = self.X[idx]
        self.y = self.y[idx]

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
        return grad_theta, grad_lmbda

    def cross(self, theta, lmbda, v, idx):
        if self.reg == 'exp':
            if lmbda.shape == (1,):
                res = np.exp(lmbda) * theta.dot(v)
            else:
                res = np.exp(lmbda) * theta * v
        elif self.reg == 'lin':
            if lmbda.shape == (1,):
                res = np.array([theta.dot(v)])
            else:
                res = theta * v
        else:
            res = np.zeros_like(lmbda)
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

    def hessian(self, theta, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]
        tmp = y * (x @ theta)

        tmp2 = np.zeros_like(tmp)
        idx1 = tmp < 0
        tmp2[idx1] = np.exp(tmp[idx1]) * tmp2[idx1]**2
        idx2 = ~idx1
        tmp2[idx2] = np.exp(- tmp[idx2]) * expit(tmp[idx2])**2
        H = x.T @ (tmp2.reshape(-1, 1) * x)

        if self.reg == 'exp':
            if lmbda.shape == (1,):
                H += np.exp(lmbda)*np.eye(x.shape[1])
            else:
                H += np.diag(np.exp(lmbda))
        elif self.reg == 'lin':
            if lmbda.shape == (1,):
                H += lmbda*np.eye(x.shape[1])
            else:
                H += np.diag(lmbda)
        return H

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
        x = self.X[idx]
        y = self.y[idx]
        n_samples = x.shape[0]
        tmp = y * (x @ theta)
        val = - logsig(tmp).mean()

        tmp2 = expit(-tmp)
        grad = -(x.T @ (y * tmp2)) / n_samples

        idx1 = tmp < 0
        tmp2[idx1] = np.exp(tmp[idx1]) * tmp2[idx1]**2
        idx2 = ~idx1
        tmp2[idx2] = np.exp(- tmp[idx2]) * expit(tmp[idx2])**2

        hvp = (x.T @ ((x @ v) * tmp2)) / n_samples

        if self.reg != 'none':
            alpha = np.exp(lmbda) if self.reg == 'exp' else lmbda
            val += .5 * (alpha @ theta ** 2)
            grad += alpha * theta
            hvp += alpha * v

        if inverse == 'id':
            inv_hvp = v
        elif inverse == 'cg':
            H = x.T @ (tmp2.reshape(-1, 1) * x)
            if self.reg == 'exp':
                if lmbda.shape == (1,):
                    H += np.exp(lmbda)*np.eye(x.shape[1])
                else:
                    H += np.diag(np.exp(lmbda))
            elif self.reg == 'lin':
                if lmbda.shape == (1,):
                    H += lmbda*np.eye(x.shape[1])
                else:
                    H += np.diag(lmbda)
            inv_hvp = np.linalg.solve(H, v)
        else:
            raise NotImplementedError('inverse unknown')

        return val, grad, hvp, self.cross(theta, lmbda, inv_hvp, idx)


class LogisticRegressionOracle(BaseOracle):
    """Class defining the oracles for the L^2 regularized logistic loss.

    **NOTE:** This class is taylored for the binary logreg.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data for the model.
    y : ndarray, shape (n_samples,)
        Targets for the logistic regression. Must be binary targets.
    reg : bool
        Whether or not to apply regularisation.
    """
    def __init__(self, X, y, reg=False):
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

        self.numba_oracle = LogisticRegressionOracleNumba(X, y, reg)

        # Store info for other oracles
        self.X = X
        self.y = y
        self.reg = reg

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

    def cross(self, theta, lmbda, v, idx):
        return self.numba_oracle.cross(theta, lmbda, v, idx)

    def hessian(self, theta, lmbda, idx):
        return self.numba_oracle.hessian(theta, lmbda, idx)

    def hvp(self, theta, lmbda, v, idx):
        return self.numba_oracle.hvp(theta, lmbda, v, idx)

    def prox(self, theta, lmbda):
        return self.numba_oracle.prox(theta, lmbda)

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
            val += .5 * (alpha @ theta ** 2)
            grad += alpha * theta
            hvp += alpha * v

        return val, grad, hvp, self.cross(theta, lmbda, inv_hvp, idx)

    def lipschitz_inner(self, theta, lmbda):
        Hop = _get_hvp_op(self.X, self.y, theta, self.reg, lmbda)
        return svds(Hop, k=1, return_singular_vectors=False)

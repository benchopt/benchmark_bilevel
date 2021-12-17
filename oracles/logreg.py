import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import linalg as splinalg


from .base import BaseOracle


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


def log_loss(x, y, theta):
    """Returns the logistic loss."""
    return - logsig(y * safe_sparse_dot(x, theta)).mean()


def grad_theta_log_loss(x, y, theta):
    """Returns the gradient of the logistic loss."""
    n_samples, n_features = x.shape
    tmp = y * safe_sparse_dot(x, theta)
    tmp2 = expit(-tmp)

    grad = -safe_sparse_dot(x.T, (y * tmp2)) / n_samples

    return grad


def hvp_log_loss(x, y, theta, v):
    """Returns an hessian-vector product for the logistic loss and a vector v.
    """
    n_samples, n_features = x.shape
    tmp = np.zeros_like(y)
    tmp2 = y * safe_sparse_dot(x, theta)

    idx1 = tmp2 < 0
    tmp[idx1] = np.exp(tmp2[idx1]) * expit(- tmp2[idx1])**2
    idx2 = tmp2 >= 0
    tmp[idx2] = np.exp(- tmp2[idx2]) * expit(tmp2[idx2])**2

    xv = safe_sparse_dot(x, v)

    hvp = safe_sparse_dot(x.T, (xv * tmp)) / n_samples
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
        assert target_type == 'binary', (
            f"Only work for binary targets, got '{target_type}'"
        )
        self.encoder = OrdinalEncoder()
        y = self.encoder.fit_transform(y[:, None]).flatten()
        y = 2 * y - 1

        self.X = X
        self.y = y
        self.reg = reg

        # attributes
        self.n_samples, n_features = X.shape
        self.variables_shape = ((n_features,), (n_features,))

    def value(self, theta, lmbda, idx):
        tmp = log_loss(self.X[idx], self.y[idx], theta)
        if self.reg:
            tmp += .5 * theta.dot(np.exp(lmbda) * theta)
        return tmp

    def grad_inner_var(self, theta, lmbda, idx):
        tmp = grad_theta_log_loss(self.X[idx], self.y[idx], theta)
        if self.reg:
            tmp += theta * np.exp(lmbda)
        return tmp

    def grad_outer_var(self, theta, lmbda, idx):
        if self.reg:
            res = .5 * np.exp(lmbda) * theta ** 2
        else:
            res = np.zeros_like(lmbda)
        return res

    def cross(self, theta, lmbda, v, idx):
        if self.reg:
            res = np.exp(lmbda) * theta * v
        else:
            res = np.zeros_like(lmbda)
        return res

    def hvp(self, theta, lmbda, v, idx):
        tmp = hvp_log_loss(self.X[idx], self.y[idx], theta, v)
        if self.reg:
            tmp += np.exp(lmbda)*v
        return tmp

    def inverse_hvp(self, theta, lmbda, v, idx, approx='cg'):
        assert approx in ['cg', 'id', 'neumann']
        if approx == 'id':
            return v
        x_i = self.X[idx]
        y_i = self.y[idx]
        Hop = _get_hvp_op(x_i, y_i, theta, self.reg, lmbda)
        Hv, success = splinalg.cg(
            Hop, v,
            x0=v.copy(),
            tol=1e-15,
            maxiter=1000,
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
        if self.reg:
            val += .5 * np.exp(lmbda) * theta ** 2
            grad += theta * np.exp(lmbda)
            hvp += np.exp(lmbda)*v

        inv_hvp = self.inverse_hvp(theta, lmbda, v, idx, approx=inverse)

        return val, grad, hvp, self.cross(theta, lmbda, inv_hvp, idx)

    def lipschitz_inner(self, theta, lmbda):
        Hop = _get_hvp_op(self.X, self.y, theta, self.reg, lmbda)
        return svds(Hop, k=1, return_singular_vectors=False)


def _get_hvp_op(x, y, theta, reg, lmbda):
    n_samples, n_features = x.shape
    tmp = np.zeros_like(y)
    tmp2 = y * safe_sparse_dot(x, theta)
    assert tmp2.shape == y.shape

    idx1 = tmp2 < 0
    tmp[idx1] = np.exp(tmp2[idx1]) * expit(- tmp2[idx1])**2
    idx2 = tmp2 >= 0
    tmp[idx2] = np.exp(- tmp2[idx2]) * expit(tmp2[idx2])**2

    # Precompute as much as possible
    if sparse.issparse(x):
        tmp = sparse.dia_matrix((tmp, 0), shape=(n_samples, n_samples))
        dX = safe_sparse_dot(tmp, x)
    else:
        dX = tmp[:, np.newaxis] * x
    alpha = np.exp(lmbda)

    def hvp(v):
        ret = np.empty_like(v)
        ret = x.T.dot(dX.dot(v))
        if reg:
            ret += alpha * v
        return ret

    Hop = splinalg.LinearOperator(
        shape=(n_features, n_features),
        matvec=lambda z: hvp(z),
        rmatvec=lambda z: hvp(z),
    )

    return Hop

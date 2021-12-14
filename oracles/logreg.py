import numpy as np
from scipy.sparse.linalg import svds

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
    return - logsig(y * (x.dot(theta))).mean()


def grad_theta_log_loss(x, y, theta):
    """Returns the gradient of the logistic loss."""
    tmp = y * (x.dot(theta))
    tmp2 = expit(-tmp)

    grad = -((y * tmp2).reshape(-1, 1) * x).mean(axis=0)

    return grad


def hvp_log_loss(x, y, theta, v):
    """Returns an hessian-vector product for the logistic loss and a vector v.
    """
    tmp = np.zeros_like(y)
    tmp2 = y * (x.dot(theta))

    idx1 = tmp2 < 0
    tmp[idx1] = np.exp(tmp2[idx1]) * expit(- tmp2[idx1])**2
    idx2 = tmp2 >= 0
    tmp[idx2] = np.exp(- tmp2[idx2]) * expit(tmp2[idx2])**2

    hvp = ((x.dot(v) * tmp).reshape(-1, 1) * x).mean(axis=0)
    return hvp


def value_grad_hvp_log_loss(x, y, theta, v):
    """Returns value, gradient, hessian-vector product for the logistic loss.
    """
    tmp = y * (x.dot(theta))
    val = - logsig(tmp).mean()

    tmp2 = expit(-tmp)
    grad = -((y * tmp2).reshape(-1, 1) * x).mean(axis=0)

    idx1 = tmp < 0
    tmp2[idx1] = np.exp(tmp[idx1]) * tmp2[idx1]**2
    idx2 = tmp >= 0
    tmp2[idx2] = np.exp(- tmp[idx2]) * expit(tmp[idx2])**2

    hvp = ((x.dot(v) * tmp2).reshape(-1, 1) * x).mean(axis=0)
    return val, grad, hvp


class LogisticRegressionOracle(BaseOracle):
    """Class defining the oracles for the L^2 regularized logistic loss."""
    def __init__(self, X, y, batch_size=1, reg=False):
        self.n_samples = X.shape[0]
        self.variable_shape = (X.shape[1],)
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.reg = reg

    def value(self, theta, lmbda, id_x):
        tmp = log_loss(self.X[id_x], self.y[id_x], theta)
        if self.reg:
            tmp += .5 * theta.dot(np.exp(lmbda) * theta)
        return tmp

    def grad_inner_var(self, theta, lmbda, id_x):
        tmp = grad_theta_log_loss(self.X[id_x], self.y[id_x], theta)
        if self.reg:
            tmp += theta * np.exp(lmbda)
        return tmp

    def grad_outer_var(self, theta, lmbda, id_x):
        if self.reg:
            res = .5 * np.exp(lmbda) * theta ** 2
        else:
            res = np.zeros_like(lmbda)
        return res

    def cross(self, theta, lmbda, v, id_x):
        if self.reg:
            res = np.exp(lmbda) * theta * v
        else:
            res = np.zeros_like(lmbda)
        return res

    def hvp(self, theta, lmbda, v, id_x):
        tmp = hvp_log_loss(self.X[id_x], self.y[id_x], theta, v)
        if self.reg:
            tmp += np.exp(lmbda)*v
        return tmp

    def inverse_hessian_vector_prod(self, theta, lmbda, v, id_x):
        x_i = self.X[id_x]
        y_i = self.y[id_x]
        tmp = np.zeros_like(y_i)
        tmp2 = y_i * (x_i.dot(theta))

        idx1 = tmp2 < 0
        tmp[idx1] = np.exp(tmp2[idx1]) * expit(- tmp2[idx1])**2
        idx2 = tmp2 >= 0
        tmp[idx2] = np.exp(- tmp2[idx2]) * expit(tmp2[idx2])**2
        H = np.dot(x_i.T, tmp[:, None] * x_i) / x_i.shape[0]
        if self.reg:
            H += np.diag(np.exp(lmbda))
        return np.linalg.solve(H, v)

    def oracles(self, theta, lmbda, v, id_x):
        val, grad, h = value_grad_hvp_log_loss(
            self.X[id_x], self.y[id_x], theta, v)
        if self.reg:
            val += .5 * np.exp(lmbda) * theta ** 2
            grad += theta * np.exp(lmbda)
            h += np.exp(lmbda)*v
        return val, grad, self.cross(theta, lmbda, v, id_x), h

    def lipschitz_inner(self, inner_var, outer_var):
        tmp = np.zeros_like(self.y)
        tmp2 = self.y * (self.X.dot(inner_var))

        idx1 = tmp2 < 0
        tmp[idx1] = np.exp(tmp2[idx1]) * expit(- tmp2[idx1])**2
        idx2 = tmp2 >= 0
        tmp[idx2] = np.exp(- tmp2[idx2]) * expit(tmp2[idx2])**2
        H = np.dot(self.X.T, tmp[:, None] * self.X) / self.X.shape[0]
        if self.reg:
            H += np.diag(np.exp(outer_var))
        return svds(H, k=1, return_singular_vectors=False)

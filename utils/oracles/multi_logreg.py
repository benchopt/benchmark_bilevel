import numpy as np
from sklearn.preprocessing import OneHotEncoder

from numba import njit
from numba import float64, int64, types    # import the types
from numba.experimental import jitclass

from .base import BaseOracle
from ..sparse_matrix import CSRMatrix

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)


@njit
def logsumexp(x):
    m = np.zeros(x.shape[0])
    for k in range(x.shape[0]):
        m[k] = np.max(x[k, :])
    x = x - m.reshape(-1, 1)
    e = np.exp(x)
    sumexp = e.sum(axis=1)
    lse = np.log(sumexp) + m
    return lse


@njit
def softmax(x):
    return np.exp(x - logsumexp(x).reshape(-1, 1))


@njit
def multilogreg_loss(x, y, theta_flat):
    """ Compute the multinomial logistic loss.
    """
    n_samples, n_features = x.shape
    _, n_classes = y.shape
    theta = theta_flat.reshape(n_features, n_classes)

    prod = x.dot(theta)

    individual_losses = np.zeros(n_samples, dtype=np.float64)
    lse = logsumexp(prod)
    for i in range(y.shape[0]):
        individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
    loss = (individual_losses).sum() / n_samples
    return loss


@njit
def multilogreg_grad(x, y, theta_flat):
    """ Compute gradient of the multinomial logistic loss.
    """
    n_samples, n_features = x.shape
    _, n_classes = y.shape
    theta = theta_flat.reshape(n_features, n_classes)

    prod = x.dot(theta)
    y_proba = softmax(prod)

    return x.T.dot(y_proba - y).ravel() / n_samples


@njit
def multilogreg_hvp(x, y, theta_flat, v_flat):
    """ Compute the HVP of the multinomial logistic loss with v."""
    n_samples, n_features = x.shape
    _, n_classes = y.shape

    theta = theta_flat.reshape(n_features, n_classes)
    v = v_flat.reshape(n_features, n_classes)

    prod = x.dot(theta)
    y_proba = softmax(prod)
    xv = x.dot(v)
    hvp = x.T.dot(softmax_hvp(y_proba, xv)).ravel() / n_samples

    return hvp


@njit
def multilogreg_value_grad_hvp(x, y, theta_flat, v_flat):
    n_samples, n_features = x.shape
    _, n_classes = y.shape

    theta = theta_flat.reshape(n_features, n_classes)
    v = v_flat.reshape(n_features, n_classes)

    prod = x.dot(theta)
    y_proba = softmax(prod)
    xv = x.dot(v)

    value = 0
    for i in range(y.shape[0]):
        value += np.log(y_proba[i][y[i] == 1][0]) / n_samples
    grad = x.T.dot(y_proba - y).ravel() / n_samples
    hvp = x.T.dot(softmax_hvp(y_proba, xv)).ravel() / n_samples

    return value, grad, hvp


@njit
def softmax_hvp(z, v):
    """
    Computes the HVP for the softmax at x times v where z = softmax(x)
    """
    prod = z * v
    return prod - z * np.sum(prod, axis=1).reshape(-1, 1)


spec = [
    ('X', CSRMatrix.class_type.instance_type),
    ('y', float64[:, ::1]),
    ('reg', types.boolean),
    ('n_samples', int64),
    ('n_features', int64),
    ('n_classes', int64),
    ('variables_shape', int64)
]


@jitclass(spec)
class MulticlassLogisticRegressionOracleNumba():
    def __init__(self, X, y, reg=True):

        # Store info for other
        self.X = X
        self.y = y
        self.reg = reg

        self.n_samples, self.n_features = X.shape
        self.n_classes = y.shape[1]

    def value(self, theta_flat, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]

        res = multilogreg_loss(x, y, theta_flat)

        if self.reg:
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            res += .5 * (np.exp(lmbda).reshape(-1, 1) * (theta ** 2)).sum()

        return res

    def grad_inner_var(self, theta_flat, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]

        res = multilogreg_grad(x, y, theta_flat)

        if self.reg:
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            res += (np.exp(lmbda).reshape(-1, 1) * theta).ravel()
        return res

    def grad_outer_var(self, theta_flat, lmbda, idx):
        if self.reg:
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            grad = .5*(np.exp(lmbda).reshape(-1, 1) * (theta ** 2)).sum(axis=1)
        else:
            grad = np.zeros(self.n_features)
        return grad

    def grad(self, theta_flat, lmbda, idx):
        grad_inner = self.grad_inner_var(theta_flat, lmbda, idx)
        grad_outer = self.grad_outer_var(theta_flat, lmbda, idx)
        return grad_inner, grad_outer

    def cross(self, theta_flat, lmbda, v_flat, idx):
        res = np.zeros(lmbda.shape)
        if self.reg:
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            v = v_flat.reshape(self.n_features, self.n_classes)
            for i in range(self.n_features):
                res[i] = np.exp(lmbda[i]) * theta[i].dot(v[i])
        return res

    def hvp(self, theta_flat, lmbda, v_flat, idx):
        x = self.X[idx]
        y = self.y[idx]

        res = multilogreg_hvp(x, y, theta_flat, v_flat)
        if self.reg:
            v = v_flat.reshape(self.n_features, self.n_classes)
            res += (np.exp(lmbda).reshape(-1, 1) * v).ravel()

        return res

    def prox(self, theta, lmbda):
        return theta, lmbda

    def oracles(self, theta_flat, lmbda, v_flat, idx):

        x = self.X[idx]
        y = self.y[idx]

        val, grad, hvp = multilogreg_value_grad_hvp(x, y, theta_flat, v_flat)
        cross = np.zeros(lmbda.shape)

        if self.reg:
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            v = v_flat.reshape(self.n_features, self.n_classes)

            val += .5 * (np.exp(lmbda).reshape(-1, 1) * (theta ** 2)).sum()
            grad += (np.exp(lmbda).reshape(-1, 1) * theta).ravel()
            hvp += (np.exp(lmbda).reshape(-1, 1) * v).ravel()

            for i in range(self.n_features):
                cross[i] = np.exp(lmbda[i]) * theta[i].dot(v[i])

        return val, grad, hvp, cross


class MulticlassLogisticRegressionOracle(BaseOracle):
    """Class defining the oracles for the L^2 regularized multinomial logistic
    loss.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data for the model.
    y : vector, shape (n_samples,) or ndarray, shape (n_samples, n_classes)
        Targets for the logistic regression. Must be binary targets.
    reg : bool
        Whether or not to apply l^2 regularisation.
    """

    def __init__(self, X, y, reg=True):
        super().__init__()

        # Make sure the targets are one hot encoded.
        if y.ndim == 1:
            y = OneHotEncoder().fit_transform(y[:, None]).toarray()

        # Store info for other
        self.X = X
        self.y = y.astype(np.float64)
        self.reg = reg

        self.n_samples, self.n_features = X.shape
        self.n_classes = y.shape[1]

        self.variables_shape = np.array([
            [(self.n_features, self.n_classes)],
            [self.n_features]
        ], dtype=object)

        self.numba_oracle = MulticlassLogisticRegressionOracleNumba(
            self.X, self.y, self.reg
        )

    def value(self, theta_flat, lmbda, idx):
        return self.numba_oracle.value(theta_flat, lmbda, idx)

    def grad_inner_var(self, theta_flat, lmbda, idx):
        return self.numba_oracle.grad_inner_var(theta_flat, lmbda, idx)

    def grad_outer_var(self, theta_flat, lmbda, idx):
        return self.numba_oracle.grad_outer_var(theta_flat, lmbda, idx)

    def grad(self, theta_flat, lmbda, idx):
        return self.numba_oracle.grad(theta_flat, lmbda, idx)

    def cross(self, theta, lmbda, v_flat, idx):
        return self.numba_oracle.cross(theta, lmbda, v_flat, idx)

    def hvp(self, theta_flat, lmbda, v_flat, idx):
        return self.numba_oracle.hvp(theta_flat, lmbda, v_flat, idx)

    def oracles(self, theta_flat, lmbda, v_flat, idx):
        return self.numba_oracle.oracles(theta_flat, lmbda, v_flat, idx)

    def accuracy(self, theta_flat, lmbda, x, y):
        if y.ndim == 2:
            y = y.argmax(axis=1)
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        prod = x @ theta
        return np.mean(np.argmax(prod, axis=1) != y)

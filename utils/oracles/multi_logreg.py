import numpy as np
# from scipy import sparse
import scipy.special as sc
# from scipy.sparse.linalg import svds
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.extmath import safe_sparse_dot
# from scipy.sparse import linalg as splinalg

# from numba import njit
# from numba import float64, int64, types    # import the types
# from numba.experimental import jitclass

from .base import BaseOracle

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)


def multilogreg_loss(x, y, theta_flat):
    """ Compute the multinomial logistic loss.
    """
    n_samples, n_features = x.shape
    _, n_classes = y.shape[1]
    theta = theta_flat.reshape(n_features, n_classes)

    prod = safe_sparse_dot(x, theta)

    individual_losses = -prod[y == 1] + sc.logsumexp(prod, axis=1)
    loss = -(individual_losses).sum() / n_samples
    return loss


def multilogreg_grad(x, y, theta_flat):
    """ Compute gradient of the multinomial logistic loss.
    """
    n_samples, n_features = x.shape
    _, n_classes = y.shape
    theta = theta_flat.reshape(n_features, n_classes)

    prod = safe_sparse_dot(x, theta)
    y_proba = sc.softmax(prod, axis=1)
    return safe_sparse_dot(x.T, y_proba - y).ravel() / n_samples


def multilogreg_hvp(x, y, theta_flat, v_flat):
    """ Compute the HVP of the multinomial logistic loss with v."""
    n_samples, n_features = x.shape
    _, n_classes = y.shape

    theta = theta_flat.reshape(n_features, n_classes)
    v = v_flat.reshape(n_features, n_classes)

    prod = safe_sparse_dot(x, theta),
    y_proba = sc.softmax(prod, axis=1)
    xv = safe_sparse_dot(x, v)
    hvp = safe_sparse_dot(x.T, softmax_hvp(y_proba, xv)).ravel() / n_samples

    return hvp


def multilogreg_value_grad_hvp(x, y, theta_flat, v_flat):
    n_samples, n_features = x.shape
    _, n_classes = y.shape

    theta = theta_flat.reshape(n_features, n_classes)
    v = v_flat.reshape(n_features, n_classes)

    prod = safe_sparse_dot(x, theta)
    y_proba = sc.softmax(prod, axis=1)
    xv = safe_sparse_dot(x, v)

    value = np.log(y_proba[y == 1]).sum() / n_samples
    grad = safe_sparse_dot(x.T, y_proba - y).ravel() / n_samples
    hvp = safe_sparse_dot(x.T, softmax_hvp(y_proba, xv)).ravel() / n_samples

    return value, grad, hvp


def softmax_hvp(z, v):
    """
    Computes the HVP for the softmax at x times v where z = softmax(x)
    """
    prod = z * v
    return prod - z * np.sum(prod, axis=1, keepdims=True)


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
        ])

    def value(self, theta_flat, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]

        res = multilogreg_loss(x, y, theta_flat)

        if self.reg:
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            res += .5 * (np.exp(lmbda)[:, None] * (theta ** 2)).sum()

        return res

    def grad_inner_var(self, theta_flat, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]

        res = multilogreg_grad(x, y, theta_flat)

        if self.reg:
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            res += (np.exp(lmbda)[:, None] * theta).ravel()

        return res

    def grad_outer_var(self, theta_flat, lmbda, idx):
        if self.reg:
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            grad = .5 * (np.exp(lmbda) * (theta ** 2)).sum()
        else:
            grad = np.zeros(self.n_features)

        return grad

    def cross(self, theta_flat, lmbda, v_flat, idx):
        res = np.zeros(lmbda.shape)
        if self.reg:
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            v = theta_flat.reshape(self.n_features, self.n_classes)
            for i in range(self.n_features):
                res[i * self.n_features:(i + 1) * self.n_features] = \
                    np.exp(lmbda[i]) * theta[i].dot(v[i])
        return res

    def hvp(self, theta_flat, lmbda, v_flat, idx):
        x = self.X[idx]
        y = self.y[idx]

        res = multilogreg_hvp(x, y, v_flat, idx)
        if self.reg:
            v = v_flat.reshape(self.n_features, self.n_classes)
            res += (np.exp(lmbda)[:, None] * v).ravel()

        return res

    def oracles(self, theta_flat, lmbda, v_flat, idx):

        x = self.X[idx]
        y = self.y[idx]

        val, grad, hvp = multilogreg_value_grad_hvp(x, y, theta_flat, v_flat)
        cross = np.zeros(lmbda.shape)

        if self.reg:
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            v = theta_flat.reshape(self.n_features, self.n_classes)

            val += .5 * (np.exp(lmbda)[:, None] * (theta ** 2)).sum()
            grad += (np.exp(lmbda)[:, None] * theta).ravel()
            hvp += (np.exp(lmbda)[:, None] * v).ravel()

            for i in range(self.n_features):
                cross[i * self.n_features:(i + 1) * self.n_features] = \
                    np.exp(lmbda[i]) * theta[i].dot(v[i])

        return val, grad, hvp, cross

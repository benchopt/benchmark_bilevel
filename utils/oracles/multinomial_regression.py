import numpy as np
from sklearn.preprocessing import OneHotEncoder

import scipy.special as sc

from .base import BaseOracle

import warnings

warnings.filterwarnings("error", category=RuntimeWarning)


def softmax_hvp(z, v):
    """
    Computes the HVP for the softmax at x times v where z = softmax(x)
    """
    prod = z * v
    return prod - z * np.sum(prod, axis=1, keepdims=True)


def multilogreg_oracle(X, Y, theta, idx):
    x = X[idx]
    y = Y[idx]
    n_samples, n_features = x.shape
    Y_proba = sc.softmax(x @ theta, axis=1)
    individual_losses = -np.log(Y_proba[y == 1])
    loss = -(individual_losses).sum() / n_samples
    grad_theta = -x.T @ ((Y_proba - y)) / n_samples
    return loss, grad_theta


class MultinomialLogRegOracle(BaseOracle):
    """Class defining the oracles for datacleaning
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

    def __init__(self, X, y):
        super().__init__()

        # Make sure the targets are one hot encoded.
        if y.ndim == 1:
            y = OneHotEncoder().fit_transform(y[:, None]).toarray()

        # Store info for other
        self.X = np.ascontiguousarray(X)
        self.y = y.astype(np.float64)

        # attributes
        self.n_samples, self.n_features = X.shape
        _, self.n_classes = y.shape
        self.variables_shape = np.array(
            [[self.n_features * self.n_classes], [self.n_samples]]
        )

    def value(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        n_samples, n_features = x.shape
        prod = x @ theta
        individual_losses = -prod[y == 1] + sc.logsumexp(prod, axis=1)
        loss = (individual_losses).sum() / n_samples
        return loss

    def grad_inner_var(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        n_samples, n_features = x.shape
        Y_proba = sc.softmax(x @ theta, axis=1)
        grad_theta = x.T @ ((Y_proba - y)) / n_samples
        return grad_theta.ravel()

    def grad_outer_var(self, theta_flat, lmbda, idx):
        return 0

    def grad(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        n_samples, n_features = x.shape
        Y_proba = sc.softmax(x @ theta, axis=1)
        grad_theta = x.T @ ((Y_proba - y)) / n_samples
        return grad_theta.ravel(), 0.0

    def cross(self, theta_flat, lmbda, v_flat, idx):
        return 0

    def hvp(self, theta_flat, lmbda, v_flat, idx):
        return 0

    def prox(self, theta, lmbda):
        return theta, lmbda

    def inverse_hvp(self, theta_flat, lmbda, v_flat, idx, approx="cg"):
        return 0

    def oracles(self, theta_flat, lmbda, v_flat, idx, inverse="id"):
        """Returns the value, the gradient,"""
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        n_samples, n_features = x.shape
        prod = x @ theta
        Y_proba = sc.softmax(prod, axis=1)
        individual_losses = -prod[y == 1] + sc.logsumexp(prod, axis=1)
        loss = (individual_losses).sum() / n_samples
        grad_theta = x.T @ ((Y_proba - y)) / n_samples
        return loss, grad_theta.ravel(), None, None

    def accuracy(self, theta_flat, lmbda, x, y):
        n_samples, _ = x.shape
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        prod = x @ theta
        return np.sum(np.argmax(prod, axis=1) != y) / n_samples

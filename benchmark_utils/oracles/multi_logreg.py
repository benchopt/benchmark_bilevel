import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import OneHotEncoder

import scipy.special as sc
from scipy.sparse import issparse
from scipy.sparse import linalg as splinalg

from .base import BaseOracle

import warnings

warnings.filterwarnings("error", category=RuntimeWarning)


def my_softmax_and_logsumexp(x):
    m = np.max(x, axis=1)
    x = x - m[:, None]
    e = np.exp(x)
    s = e / e.sum(axis=1, keepdims=True)
    lse = np.log(e.sum(axis=1)) + m
    return s, lse


def softmax_hvp(z, v):
    """
    Computes the HVP for the softmax at x times v where z = softmax(x)
    """
    prod = z * v
    return prod - z * np.sum(prod, axis=1, keepdims=True)


class MultiLogRegOracle(BaseOracle):
    """Class defining the oracles for multiclass logistic regression

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

    def __init__(self, X, y, reg='exp'):
        super().__init__()

        # Make sure the targets are one hot encoded.
        if y.ndim == 1:
            y = OneHotEncoder().fit_transform(y[:, None]).toarray()

        # Store info for other
        self.X = X
        if not issparse(self.X):
            self.X = np.ascontiguousarray(X)
        self.y = y.astype(np.float64)
        self.reg = reg

        # attributes
        self.n_samples, self.n_features = X.shape
        _, self.n_classes = y.shape
        self.variables_shape = np.array(
            [[self.n_features * self.n_classes], [self.n_classes]]
        )

    def value(self, theta_flat, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]
        theta = theta_flat.reshape(self.n_features, self.n_classes)

        prod = safe_sparse_dot(x, theta)
        loss = (-prod[y == 1] + sc.logsumexp(prod, axis=1)).mean()
        regul = 0
        if self.reg == 'exp':
            alpha = np.exp(lmbda)
            regul = 0.5 * alpha @ (theta * theta).sum(axis=0)
        return loss + 0.5 * regul

    def grad_inner_var(self, theta_flat, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]
        theta = theta_flat.reshape(self.n_features, self.n_classes)

        n_samples, n_features = x.shape
        Y_proba = sc.softmax(safe_sparse_dot(x, theta), axis=1)
        grad_theta = safe_sparse_dot(
            x.T, (Y_proba - y)
        ) / n_samples

        if self.reg == 'exp':
            alpha = np.exp(lmbda)
            grad_theta += alpha * theta

        return grad_theta.ravel()

    def grad_outer_var(self, theta_flat, lmbda, idx):

        if self.reg == 'exp':
            alpha = np.exp(lmbda)
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            grad_lmbda = 0.5 * alpha * (theta * theta).sum(axis=0)
        elif self.reg == 'none':
            grad_lmbda = np.zeros_like(lmbda)
        else:
            raise ValueError()

        return grad_lmbda

    def grad(self, theta_flat, lmbda, idx):
        x = self.X[idx]
        y = self.y[idx]
        theta = theta_flat.reshape(self.n_features, self.n_classes)

        n_samples, n_features = x.shape
        Y_proba = sc.softmax(safe_sparse_dot(x, theta), axis=1)
        grad_theta = safe_sparse_dot(
            x.T, (Y_proba - y)
        ) / n_samples

        if self.reg == 'exp':
            alpha = np.exp(lmbda)
            grad_theta += alpha * theta
            grad_lmbda = 0.5 * alpha * (theta * theta).sum(axis=0)
        elif self.reg == 'none':
            grad_lmbda = np.zeros_like(lmbda)
        else:
            raise ValueError()

        return grad_theta.ravel(), grad_lmbda

    def cross(self, theta_flat, lmbda, v_flat, idx):
        if self.reg == "exp":
            theta = theta_flat.reshape(self.n_features, self.n_classes)
            v = v_flat.reshape(self.n_features, self.n_classes)
            cross_v = np.exp(lmbda) * (theta * v).sum(axis=0)
        else:
            cross_v = np.zeros_like(lmbda)
        return cross_v

    def hvp(self, theta_flat, lmbda, v_flat, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        v = v_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        n_samples, _ = x.shape
        Y_proba = sc.softmax(safe_sparse_dot(x, theta), axis=1)
        xv = safe_sparse_dot(x, v)
        hvp = safe_sparse_dot(
            x.T, softmax_hvp(Y_proba, xv)
        ) / n_samples

        if self.reg == 'exp':
            alpha = np.exp(lmbda)
            hvp += alpha * v
        elif self.reg != 'none':
            raise NotImplementedError
        return hvp.ravel()

    def prox(self, theta, lmbda):
        return theta, lmbda

    def inverse_hvp(self, theta_flat, lmbda, v_flat, idx, approx="cg"):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        v = v_flat.reshape(self.n_features, self.n_classes)
        if approx == "id":
            return v
        if approx != "cg":
            raise NotImplementedError
        x = self.X[idx]
        Y_proba = sc.softmax(safe_sparse_dot(x, theta), axis=1)
        n_samples, n_features = x.shape
        n_classes = self.n_classes

        def compute_hvp(v_flat):
            v = v_flat.reshape(n_features, n_classes)
            xv = safe_sparse_dot(x, v)
            hvp = safe_sparse_dot(
                x.T, (softmax_hvp(Y_proba, xv))
            ) / n_samples
            return hvp.ravel() + 2 * lmbda * v_flat

        Hop = splinalg.LinearOperator(
            shape=(n_features * n_classes, n_features * n_classes),
            matvec=lambda z: compute_hvp(z),
            rmatvec=lambda z: compute_hvp(z),
        )
        Hv, success = splinalg.cg(
            Hop,
            v_flat,
            x0=v_flat,
            tol=1e-8,
            maxiter=5000,
        )
        if success != 0:
            print("CG did not converge to the desired precision")
        return Hv

    def oracles(self, theta_flat, lmbda, v_flat, idx, inverse="id"):
        """Returns the value, the gradient,"""
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        v = v_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        n_samples, n_features = x.shape
        prod = safe_sparse_dot(x, theta)
        Y_proba, lse = my_softmax_and_logsumexp(prod)
        individual_losses = -prod[y == 1] + lse
        loss = (individual_losses).mean()
        grad_theta = safe_sparse_dot(x.T, Y_proba - y) / n_samples
        xv = safe_sparse_dot(x, v)
        hvp = safe_sparse_dot(
            x.T, (softmax_hvp(Y_proba, xv))
        ) / n_samples
        cross_v = np.zeros(self.n_classes)

        if self.reg == 'exp':
            alpha = np.exp(lmbda)
            loss += 0.5 * alpha @ (theta * theta).sum(axis=0)
            grad_theta += alpha * theta
            hvp += alpha * v
            cross_v += alpha * (theta * v).sum(axis=0)
        elif self.reg != 'none':
            raise NotImplementedError
        return loss, grad_theta.ravel(), hvp.ravel(), cross_v.ravel()

    def accuracy(self, theta_flat, lmbda, x, y):
        if y.ndim == 2:
            y = y.argmax(axis=1)
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        prod = x @ theta
        return np.mean(np.argmax(prod, axis=1) != y)

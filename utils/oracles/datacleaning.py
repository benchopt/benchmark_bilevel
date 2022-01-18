import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import OneHotEncoder

from scipy.sparse import linalg as splinalg
import scipy.special as sc

from numba import njit
from numba import float64, int64, types  # import the types
from numba.experimental import jitclass
import numba_scipy.special
from .base import BaseOracle

from benchopt.utils import profile
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


def datacleaning_oracle(X, Y, theta, Lbda, v, idx):
    x = X[idx]
    y = Y[idx]
    lbda = Lbda[idx]
    grad_lbda = np.zeros_like(Lbda)
    jvp = np.zeros_like(Lbda)
    n_samples, n_features = x.shape
    prod = x @ theta
    Y_proba, lse = my_softmax_and_logsumexp(prod)
    weights = sc.expit(lbda)
    individual_losses = -prod[y == 1] + lse
    loss = (individual_losses * weights).sum() / n_samples
    grad_theta = x.T @ ((Y_proba - y) * weights[:, None]) / n_samples
    d_weights = weights - weights ** 2
    grad_lbda[idx] = d_weights * individual_losses / n_samples
    xv = x @ v
    hvp = x.T @ (softmax_hvp(Y_proba, xv) * weights[:, None]) / n_samples
    jvp[idx] = d_weights * np.sum((Y_proba - y) * xv, axis=1) / n_samples
    return loss, grad_theta, grad_lbda, hvp, jvp


class DataCleaningOracle(BaseOracle):
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

    def __init__(self, X, y, reg=1e-3):
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
        self.reg = reg

    def value(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        n_samples, _ = x.shape
        lbda = lmbda[idx]
        prod = x @ theta
        weights = sc.expit(lbda)
        individual_losses = -prod[y == 1] + sc.logsumexp(prod, axis=1)
        regul = np.dot(theta_flat, theta_flat)
        return (individual_losses * weights).sum() / n_samples + self.reg * regul

    def grad_inner_var(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        lbda = lmbda[idx]
        n_samples, n_features = x.shape
        Y_proba = sc.softmax(x @ theta, axis=1)
        weights = sc.expit(lbda)
        grad_theta = x.T @ ((Y_proba - y) * weights[:, None]) / n_samples
        return grad_theta.ravel() + 2 * self.reg * theta_flat

    def grad_outer_var(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        lbda = lmbda[idx]
        grad_lbda = np.zeros_like(lmbda)
        n_samples, n_features = x.shape
        prod = x @ theta
        weights = sc.expit(lbda)
        individual_losses = -prod[y == 1] + sc.logsumexp(prod, axis=1)
        d_weights = weights - weights ** 2
        grad_lbda[idx] = d_weights * individual_losses / n_samples
        return grad_lbda

    def grad(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        lbda = lmbda[idx]
        grad_lbda = np.zeros_like(lmbda)
        n_samples, n_features = x.shape
        prod = x @ theta
        Y_proba = sc.softmax(prod, axis=1)
        weights = sc.expit(lbda)
        individual_losses = -prod[y == 1] + sc.logsumexp(prod, axis=1)
        grad_theta = x.T @ ((Y_proba - y) * weights[:, None]) / n_samples
        d_weights = weights - weights ** 2
        grad_lbda[idx] = d_weights * individual_losses / n_samples
        return grad_theta.ravel() + 2 * self.reg * theta_flat, grad_lbda

    def cross(self, theta_flat, lmbda, v_flat, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        v = v_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        lbda = lmbda[idx]
        jvp = np.zeros_like(lmbda)
        n_samples, n_features = x.shape
        Y_proba = sc.softmax(x @ theta, axis=1)
        weights = sc.expit(lbda)
        d_weights = weights - weights ** 2
        xv = x @ v
        jvp[idx] = d_weights * np.sum((Y_proba - y) * xv, axis=1) / n_samples
        return jvp

    def hvp(self, theta_flat, lmbda, v_flat, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        v = v_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        lbda = lmbda[idx]
        n_samples, n_features = x.shape
        Y_proba = sc.softmax(x @ theta, axis=1)
        weights = sc.expit(lbda)
        xv = x @ v
        hvp = x.T @ (softmax_hvp(Y_proba, xv) * weights[:, None]) / n_samples
        return hvp.ravel() + 2 * self.reg * v_flat

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
        y = self.y[idx]
        lbda = lmbda[idx]
        Y_proba = sc.softmax(x @ theta, axis=1)
        weights = sc.expit(lbda)
        n_samples, n_features = x.shape
        n_classes = self.n_classes

        def compute_hvp(v_flat):
            v = v_flat.reshape(n_features, n_classes)
            xv = x @ v
            hvp = x.T @ (softmax_hvp(Y_proba, xv) * weights[:, None]) / n_samples
            return hvp.ravel() + 2 * self.reg * v_flat

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

    @profile
    def oracles(self, theta_flat, lmbda, v_flat, idx, inverse="id"):
        """Returns the value, the gradient,"""
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        v = v_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        lbda = lmbda[idx]
        jvp = np.zeros_like(lmbda)
        n_samples, n_features = x.shape
        prod = x @ theta
        Y_proba, lse = my_softmax_and_logsumexp(prod)
        weights = sc.expit(lbda)
        individual_losses = -prod[y == 1] + lse
        loss = (individual_losses * weights).sum() / n_samples
        grad_theta = x.T @ ((Y_proba - y) * weights[:, None]) / n_samples
        d_weights = weights - weights ** 2
        xv = x @ v
        hvp = x.T @ (softmax_hvp(Y_proba, xv) * weights[:, None]) / n_samples
        jvp[idx] = d_weights * np.sum((Y_proba - y) * xv, axis=1) / n_samples
        return loss, grad_theta.ravel() + 2 * self.reg * theta_flat, hvp.ravel() + 2 * self.reg * v_flat, jvp

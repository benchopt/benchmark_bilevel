import numpy as np
from scipy.sparse.linalg import svds

from .base import BaseOracle


def ls_loss(x, y, theta):
    """Computes the least squares loss."""
    n = x.shape[0]
    residual = x.dot(theta) - y
    return .5 * residual.dot(residual) / n


def ls_grad_theta(x, y, theta):
    """Computes the gradient of the least squares loss with respect to the
    parameter theta."""
    n = x.shape[0]
    residual = x.dot(theta) - y
    return x.T.dot(residual) / n


def ls_hvp(x, y, theta, v):
    return (x.dot(v)[:, None] * x).mean(axis=0)


class RidgeRegressionOracle(BaseOracle):
    """Class defining the oracles for the L^2 regularized least squares
    loss."""

    def __init__(self, X, y, batch_size=1, reg=False):
        self.n_samples = X.shape[0]
        self.variable_shape = (X.shape[1],)
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.reg = reg

    def value(self, theta, lmbda, idx):
        tmp = ls_loss(self.X[idx], self.y[idx], theta)
        if self.reg:
            tmp += .5 * theta.dot(np.exp(lmbda) * theta)
        return tmp

    def grad_inner_var(self, theta, lmbda, idx):
        tmp = ls_grad_theta(self.X[idx], self.y[idx], theta)
        if self.reg:
            tmp += np.exp(lmbda)*theta
        return tmp

    def grad_outer_var(self, theta, lmbda, idx):
        if self.reg:
            grad = .5 * np.exp(lmbda) * theta ** 2
        else:
            grad = np.zeros_like(lmbda)
        return grad

    def cross(self, theta, lmbda, v, idx):
        if self.reg:
            res = np.exp(lmbda) * theta * v
        else:
            res = np.zeros_like(lmbda)
        return res

    def hvp(self, theta, lmbda, v, idx):
        tmp = ls_hvp(self.X[idx], self.y[idx], theta, v)
        if self.reg:
            tmp += np.exp(lmbda) * v
        return tmp

    def inverse_hessian_vector_prod(self, theta, lmbda, v, idx):
        x = self.X[idx]
        H = np.dot(x.T, x) / x.shape[0]
        if self.reg:
            H += np.diag(np.exp(lmbda))
        return np.linalg.solve(H, v)

    def inner_var_star(self, lmbda, idx):
        X, y = self.X[idx], self.y[idx]
        n_samples = X.shape[0]
        H = X.T.dot(X) / n_samples
        if self.reg:
            H += np.diag(np.exp(lmbda))
        return np.linalg.solve(H, X.T.dot(y) / n_samples)

    def lipschitz_inner(self, inner_var, outer_var):
        H = np.dot(self.X.T, self.X) / self.X.shape[0]
        if self.reg:
            H += np.diag(np.exp(outer_var))
        return svds(H, k=1, return_singular_vectors=False)

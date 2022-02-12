import numpy as np
from scipy.sparse import linalg as splinalg
from sklearn.preprocessing import OneHotEncoder

from numba import njit
from numba import float64, int64
from numba.experimental import jitclass

from .base import BaseOracle
from .special import expit
from ..numba_utils import one_hot_fancy_index
from .special import softmax, softmax_hvp, logsumexp, my_softmax_and_logsumexp

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)


@njit
def datacleaning_oracle(X, Y, theta, Lbda, v, idx):
    x = X[idx]
    y = Y[idx]
    lbda = Lbda[idx]
    grad_lbda = np.zeros_like(Lbda)
    jvp = np.zeros_like(Lbda)
    n_samples, n_features = x.shape
    prod = x @ theta
    Y_proba, lse = my_softmax_and_logsumexp(prod)
    weights = expit(lbda)
    lse = logsumexp(prod)
    individual_losses = np.zeros(n_samples, dtype=np.float64)
    for i in range(y.shape[0]):
        individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
    loss = (individual_losses * weights).sum() / n_samples
    grad_theta = x.T @ ((Y_proba - y) * weights.reshape(-1, 1)) / n_samples
    d_weights = weights - weights ** 2
    grad_lbda[idx] = d_weights * individual_losses / n_samples
    xv = x @ v
    hvp = x.T @ (softmax_hvp(Y_proba, xv) * weights.reshape(-1, 1)) / n_samples
    jvp[idx] = d_weights * np.sum((Y_proba - y) * xv, axis=1) / n_samples
    return loss, grad_theta, grad_lbda, hvp, jvp


spec = [
    ("X", float64[:, ::1]),
    ("y", float64[:, ::1]),
    ("reg", float64),
    ("n_samples", int64),
    ("n_features", int64),
    ("n_classes", int64),
    ("variables_shape", int64[:, ::1])
]


@jitclass(spec)
class DataCleaningOracleNumba():
    """Class defining the oracles for datacleaning

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data for the model.
    y : ndarray, shape (n_samples, n_classes)
        Targets for the logistic regression. Can be binary targets or one-hot
        encoded target.
    reg : float
        Amount of regularization.
    """

    def __init__(self, X, y, reg=2e-1):

        self.X = X
        self.y = y
        self.reg = reg

        # attributes
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        # attributes
        self.n_samples, self.n_features = X.shape
        _, self.n_classes = self.y.shape
        self.variables_shape = np.array(
            [[self.n_features * self.n_classes], [self.n_samples]]
        )

    def value(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        n_samples, _ = x.shape
        lbda = lmbda[idx]
        prod = x @ theta
        weights = expit(lbda)
        lse = logsumexp(prod)
        individual_losses = one_hot_fancy_index(prod, y) + lse
        # individual_losses = np.zeros(n_samples, dtype=np.float64)
        # for i in range(y.shape[0]):
        #     individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
        regul = np.dot(theta_flat, theta_flat)
        val = (individual_losses * weights).sum()
        return val / n_samples + self.reg * regul

    def grad_inner_var(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        lbda = lmbda[idx]
        n_samples, _ = x.shape
        Y_proba = softmax(x @ theta)
        weights = expit(lbda)
        grad_theta = x.T @ ((Y_proba - y) * weights.reshape(-1, 1)) / n_samples
        return grad_theta.ravel() + 2 * self.reg * theta_flat

    def grad_outer_var(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        lbda = lmbda[idx]
        grad_lbda = np.zeros_like(lmbda)
        n_samples, _ = x.shape
        prod = x @ theta
        weights = expit(lbda)
        lse = logsumexp(prod)
        individual_losses = one_hot_fancy_index(prod, y) + lse
        # individual_losses = np.zeros(n_samples, dtype=np.float64)
        # for i in range(y.shape[0]):
        #     individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
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
        Y_proba = softmax(prod)
        weights = expit(lbda)
        lse = logsumexp(prod)
        individual_losses = one_hot_fancy_index(prod, y) + lse
        # individual_losses = np.zeros(n_samples, dtype=np.float64)
        # for i in range(y.shape[0]):
        #     individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
        grad_theta = x.T @ ((Y_proba - y) * weights.reshape(-1, 1)) / n_samples
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
        n_samples, _ = x.shape
        Y_proba = softmax(x @ theta)
        weights = expit(lbda)
        d_weights = weights - weights ** 2
        xv = x @ v
        jvp[idx] = d_weights * np.sum((Y_proba - y) * xv, axis=1) / n_samples
        return jvp

    def hvp(self, theta_flat, lmbda, v_flat, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        v = v_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        lbda = lmbda[idx]
        n_samples, _ = x.shape
        Y_proba = softmax(x @ theta)
        weights = expit(lbda)
        xv = x @ v
        hvp = x.T @ (softmax_hvp(Y_proba, xv) * weights.reshape(-1, 1))
        hvp /= n_samples
        return hvp.ravel() + 2 * self.reg * v_flat

    def prox(self, theta, lmbda):
        return theta, lmbda

    def inverse_hvp(self, theta_flat, lmbda, v_flat, idx, approx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        v = v_flat.reshape(self.n_features, self.n_classes)
        if approx == "id":
            return v
        if approx != "cg":
            raise NotImplementedError
        x = self.X[idx]
        lbda = lmbda[idx]
        Y_proba = softmax(x @ theta)
        weights = expit(lbda)
        n_samples, n_features = x.shape
        n_classes = self.n_classes

        def compute_hvp(v_flat):
            v = v_flat.reshape(n_features, n_classes)
            xv = x @ v
            hvp = x.T @ (softmax_hvp(Y_proba, xv) * weights.reshape(-1, 1))
            hvp /= n_samples
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

    def oracles(self, theta_flat, lmbda, v_flat, idx, inverse):
        """Returns the value, the gradient,"""
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        v = v_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        lbda = lmbda[idx]
        jvp = np.zeros_like(lmbda)
        n_samples, _ = x.shape
        prod = x @ theta
        Y_proba, lse = my_softmax_and_logsumexp(prod)
        weights = expit(lbda)
        individual_losses = one_hot_fancy_index(prod, y) + lse
        # individual_losses = np.zeros(n_samples, dtype=np.float64)
        # for i in range(y.shape[0]):
        #     individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
        loss = (individual_losses * weights).sum() / n_samples
        grad_theta = x.T @ ((Y_proba - y) * weights.reshape(-1, 1)) / n_samples
        d_weights = weights - weights ** 2
        xv = x @ v
        hvp = x.T @ (softmax_hvp(Y_proba, xv) * weights.reshape(-1, 1))
        hvp /= n_samples
        jvp[idx] = d_weights * np.sum((Y_proba - y) * xv, axis=1) / n_samples
        return (
            loss,
            grad_theta.ravel() + 2 * self.reg * theta_flat,
            hvp.ravel() + 2 * self.reg * v_flat,
            jvp,
        )


class DataCleaningOracle(BaseOracle):
    """Class defining the oracles for the data hyper-cleaning task.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data for the model.
    y : ndarray, shape (n_samples,)
        Targets for the logistic regression. Must be binary targets.
    reg : float
        Regularization parameter.
    """
    def __init__(self, X, y, reg=2e-1):
        super().__init__()

        if y.ndim == 1:
            y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

        # Make sure reg is valid
        assert isinstance(reg, float)

        # Store info for other
        self.X = np.ascontiguousarray(X)
        self.y = y.astype(np.float64)
        self.reg = reg

        # Create a numba oracle for the numba functions
        self.numba_oracle = DataCleaningOracleNumba(
            self.X, self.y, self.reg
        )

        # attributes
        self.n_samples, self.n_features = X.shape
        _, self.n_classes = self.y.shape
        self.variables_shape = np.array(
            [[self.n_features * self.n_classes], [self.n_samples]]
        )

    def value(self, theta, lmbda, idx):
        return self.numba_oracle.value(theta, lmbda, idx)

    def grad_inner_var(self, theta, lmbda, idx):
        return self.numba_oracle.grad_inner_var(theta, lmbda, idx)

    def grad_outer_var(self, theta, lmbda, idx):
        return self.numba_oracle.grad_outer_var(theta, lmbda, idx)

    def grad(self, theta, lmbda, idx):
        return self.numba_oracle.grad(theta, lmbda, idx)

    def cross(self, theta, lmbda, v, idx):
        return self.numba_oracle.cross(theta, lmbda, v, idx)

    def hvp(self, theta, lmbda, v, idx):
        return self.numba_oracle.hvp(theta, lmbda, v, idx)

    def prox(self, theta, lmbda):
        return self.numba_oracle.prox(theta, lmbda)

    def inverse_hvp(self, theta, lmbda, v, idx, approx='cg'):
        return self.numba_oracle.inverse_hvp(self, theta, lmbda, v, idx,
                                             approx='cg')

    def oracles(self, theta_flat, lmbda, v_flat, idx, inverse):
        return self.numba_oracle.oracles(self, theta_flat, lmbda, v_flat,
                                         idx, inverse)

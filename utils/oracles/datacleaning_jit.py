import numpy as np

from scipy.sparse import linalg as splinalg

from numba import njit
from numba.experimental import jitclass
from numba import float64, int64  # import the types

import warnings

warnings.filterwarnings("error", category=RuntimeWarning)


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
    return np.exp(x - logsumexp(x))


@njit
def my_softmax_and_logsumexp(x):
    lse = logsumexp(x)
    s = np.exp(x - lse)
    return s, lse


@njit
def softmax_hvp(z, v):
    """
    Computes the HVP for the softmax at x times v where z = softmax(x)
    """
    prod = z * v
    return prod - z * np.sum(prod, axis=1).reshape(-1, 1)


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
    individual_losses = -prod[y == 1] + lse
    loss = (individual_losses * weights).sum() / n_samples
    grad_theta = x.T @ ((Y_proba - y) * weights[:, None]) / n_samples
    d_weights = weights - weights ** 2
    grad_lbda[idx] = d_weights * individual_losses / n_samples
    xv = x @ v
    hvp = x.T @ (softmax_hvp(Y_proba, xv) * weights[:, None]) / n_samples
    jvp[idx] = d_weights * np.sum((Y_proba - y) * xv, axis=1) / n_samples
    return loss, grad_theta, grad_lbda, hvp, jvp


@njit
def one_hot_encoder_numba(y):
    m = np.unique(y).shape[0]
    y_new = np.zeros((y.shape[0], m), dtype=np.float64)
    for i in range(y.shape[0]):
        y_new[i, y[i]] = 1.
    return y_new


spec = [
    ("X", float64[:, ::1]),  # an array field
    ("y", float64[:, ::1]),  # a simple scalar field
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
        Targets for the logistic regression. Must be binary targets.
    reg : bool
        Whether or not to apply regularisation.
    """

    def __init__(self, X, y, reg=2e-1):

        # Make sure the targets are one hot encoded.
        if y.ndim == 1:
            self.y = one_hot_encoder_numba(y)
        else:
            self.y = y.astype(np.float64)

        # Store info for other
        self.X = X

        # attributes
        self.n_samples, self.n_features = X.shape
        _, self.n_classes = self.y.shape
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
        weights = expit(lbda)
        lse = logsumexp(prod)
        individual_losses = np.zeros(n_samples, dtype=np.float64)
        for i in range(y.shape[0]):
            individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
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
        grad_theta = x.T @ ((Y_proba - y) * weights[:, None]) / n_samples
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
        individual_losses = np.zeros(n_samples, dtype=np.float64)
        for i in range(y.shape[0]):
            individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
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
        individual_losses = np.zeros(n_samples, dtype=np.float64)
        for i in range(y.shape[0]):
            individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
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
        hvp = x.T @ (softmax_hvp(Y_proba, xv) * weights[:, None]) / n_samples
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
            hvp = x.T @ (softmax_hvp(Y_proba, xv) * weights[:, None])
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
        individual_losses = np.zeros(n_samples, dtype=np.float64)
        for i in range(y.shape[0]):
            individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
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

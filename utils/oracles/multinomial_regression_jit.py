import numpy as np

from numba import njit
from numba.experimental import jitclass
from numba import float64, int64  # import the types

from .special import one_hot_encoder_numba
from .special import softmax, logsumexp


import warnings

warnings.filterwarnings("error", category=RuntimeWarning)


@njit
def multilogreg_oracle(X, Y, theta, idx):
    x = X[idx]
    y = Y[idx]
    n_samples, n_features = x.shape
    Y_proba = softmax(x @ theta)
    individual_losses = -np.log(Y_proba[y == 1])
    individual_losses = np.zeros(n_samples, dtype=np.float64)
    for i in range(y.shape[0]):
        individual_losses[i] = - np.log(Y_proba[i][y[i] == 1])
    loss = -(individual_losses).sum() / n_samples
    grad_theta = -x.T @ ((Y_proba - y)) / n_samples
    return loss, grad_theta


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
class MultinomialLogRegOracleNumba():
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
            self.y = one_hot_encoder_numba(y)
        else:
            self.y = y.astype(np.float64)

        # Store info for other
        self.X = X

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
        n_samples, _ = x.shape
        prod = x @ theta
        lse = logsumexp(prod)
        individual_losses = np.zeros(n_samples, dtype=np.float64)
        for i in range(y.shape[0]):
            individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
        loss = (individual_losses).sum() / n_samples
        return loss

    def grad_inner_var(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        n_samples, _ = x.shape
        Y_proba = softmax(x @ theta)
        grad_theta = x.T @ ((Y_proba - y)) / n_samples
        return grad_theta.ravel()

    def grad_outer_var(self, theta_flat, lmbda, idx):
        return 0.

    def grad(self, theta_flat, lmbda, idx):
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        n_samples, _ = x.shape
        Y_proba = softmax(x @ theta)
        grad_theta = x.T @ ((Y_proba - y)) / n_samples
        return grad_theta.ravel(), 0.0

    def cross(self, theta_flat, lmbda, v_flat, idx):
        return 0.

    def hvp(self, theta_flat, lmbda, v_flat, idx):
        return 0.

    def prox(self, theta, lmbda):
        return theta, lmbda

    def inverse_hvp(self, theta_flat, lmbda, v_flat, idx, approx="cg"):
        return 0.

    def oracles(self, theta_flat, lmbda, v_flat, idx, inverse="id"):
        """Returns the value, the gradient,"""
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        x = self.X[idx]
        y = self.y[idx]
        n_samples, _ = x.shape
        prod = x @ theta
        Y_proba = softmax(prod)
        lse = logsumexp(prod)
        individual_losses = np.zeros(n_samples, dtype=np.float64)
        for i in range(y.shape[0]):
            individual_losses[i] = - prod[i][y[i] == 1][0] + lse[i]
        loss = (individual_losses).sum() / n_samples
        grad_theta = x.T @ ((Y_proba - y)) / n_samples
        return loss, grad_theta.ravel(), 0., 0.

    def accuracy(self, theta_flat, lmbda, x, y):
        n_samples, _ = x.shape
        theta = theta_flat.reshape(self.n_features, self.n_classes)
        prod = x @ theta
        bin = np.zeros(prod.shape[0])
        for i in range(prod.shape[0]):
            bin[i] = np.argmax(prod[i]) != y[i]
        return np.sum(np.argmax(prod, axis=1) != y) / n_samples

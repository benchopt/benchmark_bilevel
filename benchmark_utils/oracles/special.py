import numpy as np
from numba import njit

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.numba_utils import np_max


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


logsig_njit = njit(logsig)


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


expit_njit = njit(expit)


@njit
def logsumexp(x):
    """Computes the logsumexp function."""
    m = np_max(x, axis=1)
    x = x - m.reshape(-1, 1)
    e = np.exp(x)
    sumexp = e.sum(axis=1)
    lse = np.log(sumexp) + m
    return lse


@njit
def softmax(x):
    """Computes the softmax function."""
    return np.exp(x - logsumexp(x).reshape(-1, 1))


@njit
def my_softmax_and_logsumexp(x):
    lse = logsumexp(x)
    s = np.exp(x - lse.reshape(-1, 1))
    return s, lse


@njit
def softmax_hvp(z, v):
    """
    Computes the HVP for the softmax at x times v where z = softmax(x)
    """
    prod = z * v
    return prod - z * np.sum(prod, axis=1).reshape(-1, 1)

import numpy as np
from numba import njit


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
def one_hot_encoder_numba(y):
    m = np.unique(y).shape[0]
    y_new = np.zeros((y.shape[0], m), dtype=np.float64)
    for i in range(y.shape[0]):
        y_new[i, y[i]] = 1.
    return y_new

import numpy as np

from .base import BaseOracle


class QuadraticOracle(BaseOracle):
    """Class defining the oracles for the L^2 regularized least squares loss.
    """
    def __init__(self, A_z, A_x, b, c):
        super().__init__()

        # Store info for other
        self.A_z = A_z
        self.A_x = A_x
        self.b = b
        self.c = c

    def value(self, z, x, idx):
        res = .5 * (z.dot(self.A_z @ z) + x.dot(self.A_x @ x))
        return res + self.b * (z.dot(x)) + self.c

    def grad_inner_var(self, z, x, idx):
        return self.A_z.dot(z) + self.b * x

    def grad_outer_var(self, z, x, idx):
        return self.A_x.dot(x) + self.b * z

    def cross(self, z, x, v, idx):
        return self.b

    def hvp(self, z, x, v, idx):
        return self.A_z.dot(v)

    def inverse_hvp(self, z, x, v, idx, approx=None):
        return np.linalg.solve(self.A_z, v)

    def inner_var_star(self, x, idx):
        return np.linalg.solve(self.A_z, -self.b * x)

    def lipschitz_inner(self):
        return np.linalg.norm(self.A_z, ord=2)

    def __getattr__(self, name):
        # construct get_* and get_batch_* for all methods in this list:
        method = getattr(self, name.replace('get_', ''))

        return _get_full_batch_method(method).__get__(self, BaseOracle)

        return super().__getattribute__(name)


def _get_full_batch_method(method):
    def get_full_batch(self, *args, **kwargs):
        return method(*args, idx=None, **kwargs)
    return get_full_batch

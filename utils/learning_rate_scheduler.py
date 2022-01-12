import numpy as np

from numba import int64, float64
from numba.experimental import jitclass


spec = [
    ('i_step', int64),
    ('constants', float64[:]),
    ('exponents', float64[:])
]


@jitclass(spec)
class LearningRateScheduler():
    """Scheduler for learning rates, either constant or decreasing.

    This class holds a state counting the number of steps performed.
    The learning rate stays coherent independent on how the iterations are
    splitted.

    Usage
    -----
    >>> lr_scheduler = LearningRateScheduler([1e-1, 1e-3], [2/3, 0])
    >>> for _ in range(max_iter):
    >>>     lr1, lr2 = lr_scheduler.get_lr()
    >>>     inner_var -= lr1 * inner_grad
    >>>     outer_var -= lr2 * outer_grad

    Parameters
    ----------
    constants : ndarray, shape (n_learning_rates)
        Constant in front of each learning rate.
    exponents : ndarray, shape (n_learning_rates)
        Exponent for each learning rate. If 0, this corresponds to constant
        learning rates.
    """
    def __init__(self, constants, exponents):

        self.constants = constants
        self.exponents = exponents

        # Internal state information
        self.i_step = 1

    def get_lr(self):
        lr = self.constants
        mask = self.exponents != 0
        lr[mask] /= self.i_step ** self.exponents[mask]
        self.i_step += 1
        return lr

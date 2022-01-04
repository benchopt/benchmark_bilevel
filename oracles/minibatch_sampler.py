import numpy as np

from numba import int64
from numba.experimental import jitclass


spec = [
    ('n_samples', int64),
    ('sample_order', int64[:]),
    ('batch_size', int64),
    ('i_batch', int64),
    ('n_batches', int64),
]


@jitclass(spec)
class MinibatchSampler():
    """Minibatch sampler helper, relying on shuffling and slices.

    Generating minibatches on the fly can be quite slow and does not allow for
    good vectorization. This helper generate a data order and take contiguous
    slices in the data as mini-batches to allow fast operations.

    **Note:** we don't store the oracle to avoid complicated type casting
    depending on the oracle as we cannot have heritance in jitclass (yet?).

    Usage
    -----
    >>> samples = MinibatchSampler(oracle, batch_size=1)
    >>> for _ in range(max_iter):
    >>>     selector = sampler.get_batch(oracle)
    >>>     grad = oracle.inner_gradient(inner_var, outer_var, selector)

    Parameters
    ----------
    oracle : Oracle jitclass
        An oracle implemented in numba, with attribute `n_samples` and method
        `set_order`.
    """
    def __init__(self, oracle, batch_size=1):
        # underlying oracle information
        self.n_samples = oracle.n_samples
        self.sample_order = np.arange(self.n_samples)

        # Batch size
        self.batch_size = batch_size

        # Internal batch information
        self.i_batch = 0
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size

    def get_batch(self, oracle):
        self.i_batch += 1
        if self.i_batch == self.n_batches:
            self.i_batch = 0
            self.shuffle(oracle)

        selector = slice(self.i_batch * self.batch_size,
                         (self.i_batch + 1) * self.batch_size)
        idx = self.sample_order[selector]
        assert len(idx) > 0
        return selector, idx

    def shuffle(self, oracle):
        idx = np.arange(self.n_samples)
        np.random.shuffle(idx)
        oracle.set_order(idx)
        self.sample_order = self.sample_order[idx]

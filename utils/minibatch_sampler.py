import numpy as np

from numba import types, typed
from numba import int64, float64
from numba.experimental import jitclass


spec = [
    ('n_samples', int64),
    ('sample_order', int64[:]),
    ('batch_size', int64),
    ('i_batch', int64),
    ('n_batches', int64),
    ('memories', types.ListType(float64[:, :]))
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
    def __init__(self, oracle, batch_size=1,
                 memories=typed.List.empty_list(float64[:, :])):
        # underlying oracle information
        self.n_samples = oracle.n_samples
        self.sample_order = np.arange(self.n_samples)

        # Batch size
        self.batch_size = batch_size

        # Internal batch information
        self.i_batch = 0
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size

        # Initialize memories container
        self.memories = memories

    def register_memory(self, memory):
        self.memories.append(memory)

    def get_batch(self, oracle):
        self.i_batch += 1
        if self.i_batch == self.n_batches:
            self.i_batch = 0
            self.shuffle(oracle)

        selector = slice(self.i_batch * self.batch_size,
                         (self.i_batch + 1) * self.batch_size)
        return selector

    def shuffle(self, oracle):
        idx = np.arange(self.n_samples)
        np.random.shuffle(idx)
        oracle.set_order(idx)
        self.sample_order = self.sample_order[idx]
        idx_memory = np.concatenate((idx, np.array([self.n_samples])))
        for memory in self.memories:
            memory[:] = memory[idx_memory]

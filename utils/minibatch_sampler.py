import numpy as np

from numba import types, typed
from numba.experimental import jitclass
from numba import int64, float64, boolean


spec = [
    ('n_samples', int64),
    ('batch_size', int64),
    ('i_batch', int64),
    ('n_batches', int64),
    ('memories', types.ListType(float64[:, :])),
    ('keep_batches', boolean)
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
    def __init__(self, oracle, batch_size=1, keep_batches=False,
                 memories=typed.List.empty_list(float64[:, :])):
        # underlying oracle information
        self.n_samples = oracle.n_samples

        # Batch size
        self.batch_size = batch_size
        self.keep_batches = keep_batches

        # Internal batch information
        self.i_batch = 0
        self.n_batches = self.n_samples // batch_size

        # Initialize memories container
        self.memories = memories.copy()

    def register_memory(self, memory):
        self.memories.append(memory)

    def get_batch(self, oracle):
        self.i_batch += 1
        if self.i_batch == self.n_batches:
            self.i_batch = 0
            self.shuffle(oracle)

        selector = slice(self.i_batch * self.batch_size,
                         (self.i_batch + 1) * self.batch_size)

        return selector, self.i_batch

    def shuffle(self, oracle):
        if self.keep_batches:
            idx_memory = np.arange(self.n_batches)
            np.random.shuffle(idx_memory)
            idx = (
                idx_memory.reshape(-1, 1) * self.batch_size
                + np.arange(self.batch_size)
            ).flatten()
            idx = np.concatenate(
                (idx, -np.arange(self.n_samples % self.n_batches)[::-1])
            )
            idx_memory = np.concatenate((idx_memory, np.array([-1])))

        else:
            idx = np.arange(self.n_samples)
            np.random.shuffle(idx)
            idx_memory = np.concatenate((idx, np.array([-1])))
        oracle.set_order(idx)
        for memory in self.memories:
            memory[:] = memory[idx_memory]

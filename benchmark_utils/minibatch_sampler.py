import numpy as np
from numba import int64

import jax.lax
import jax.random
from jax import jit
import jax.numpy as jnp
from functools import partial


spec = [  # specifications for numba class
    ('n_samples', int64),
    ('batch_size', int64),
    ('i_batch', int64),
    ('n_batches', int64),
    ('batch_order', int64[:]),
]


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
    >>>     selector = sampler.get_batch()
    >>>     grad = oracle.inner_gradient(inner_var, outer_var, selector)

    Parameters
    ----------
    oracle : Oracle jitclass
        An oracle implemented in numba, with attribute `n_samples` and method
        `set_order`.
    """
    def __init__(self, n_samples, batch_size=1):

        # Batch size
        self.n_samples = n_samples
        self.batch_size = batch_size

        # Internal batch information
        self.i_batch = 0
        self.n_batches = (n_samples + batch_size - 1) // batch_size
        self.batch_order = np.arange(self.n_batches)

    def get_batch(self):
        idx = self.batch_order[self.i_batch]
        selector = slice(idx * self.batch_size,
                         (idx + 1) * self.batch_size)
        self.i_batch += 1
        if self.i_batch == self.n_batches:
            np.random.shuffle(self.batch_order)
            self.i_batch = 0

        weight = self.batch_size / self.n_samples
        if idx == self.n_batches - 1 and self.n_samples % self.batch_size != 0:
            weight = (self.n_samples % self.batch_size) / self.n_samples

        return selector, (idx, weight)


@jit
def keep_ibatch(i, key):
    return i + 1, key


@jit
def reset_ibatch(i, key):
    return 0, jax.random.split(key, 1)[0]


@partial(jit, static_argnames=('n_batches', 'batch_size'))
def _sampler(n_batches=10, batch_size=1, **state):
    """Jax version of the minibatch sampler."""
    idx = jax.random.permutation(state['key'], n_batches - 1)[state['i_batch']]
    start = 1 * idx
    state['i_batch'], state['key'] = jax.lax.cond(
        state['i_batch'] == n_batches, reset_ibatch, keep_ibatch,
        state['i_batch'], state['key']
    )

    return start, state


def init_sampler(n_samples=10, batch_size=1, random_state=1):
    """Initialize the minibatch sampler."""
    n_batches = (n_samples + batch_size - 1) // batch_size
    state = dict(
        i_batch=jnp.array(0),
        key=jax.random.PRNGKey(random_state),
    )

    return jit(lambda **state: _sampler(n_batches, batch_size, **state)), state

import jax.numpy as jnp
from jax.experimental import sparse

from scipy.sparse import issparse


def convert_array_framework(x, framework=None):
    if framework == "jax":
        if issparse(x):
            # Passing by a dense matrix is mandatory to be able to use
            # n_batch>1. n_batch = 0 is incompatible with vmapping for the
            # moment.
            x = sparse.BCOO.fromdense(x.todense(), n_batch=1)
        else:
            x = jnp.array(x)
    return x

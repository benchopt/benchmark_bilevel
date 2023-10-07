import jax.numpy as jnp
from jax.experimental import sparse

from scipy.sparse import issparse


def convert_array_framework(x, framework=None):
    if framework == "jax":
        if issparse(x):
            # Passing by a dense matrix is mandatory to be able to use
            # n_batch>1. n_batch = 0 is incompatible with vmapping for the
            # moment.
            nb = 1000
            temp = sparse.BCOO.fromdense(x[:nb].todense(), n_batch=1)
            for i in range((x.shape[0] + nb - 1) // nb - 1):
                temp = sparse.bcoo_concatenate(
                    [
                        temp,
                        sparse.BCOO.fromdense(x[(i+1)*nb:(i+2)*nb].todense(),
                                              n_batch=1)
                    ],
                    dimension=0
                )
            x = temp
        else:
            x = jnp.array(x)
    return x

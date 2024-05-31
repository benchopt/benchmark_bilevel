import jax
import jax.numpy as jnp


@jax.jit
def keep_ibatch(state):
    return state['i_batch'] + 1, state['batch_order'], state['key'],


def reset_ibatch(state):
    key1, key = jax.random.split(state['key'])
    return 0, jax.random.permutation(key1, state['batch_order']), key,


def _sampler(n_batches, batch_size, weights, state):
    """Jax version of the minibatch sampler."""
    idx = state['batch_order'][state['i_batch']]
    start = batch_size * idx
    state['i_batch'], state['batch_order'], state['key'] = jax.lax.cond(
        state['i_batch'] == n_batches, reset_ibatch, keep_ibatch, state
    )
    weight = jax.lax.select(idx == n_batches - 1, weights[1], weights[0])

    return start, idx, weight, state


def init_sampler(n_samples=10, batch_size=1,
                 key=jax.random.PRNGKey(1)):
    """Initialize the minibatch sampler."""
    n_batches = (n_samples + batch_size - 1) // batch_size

    # compute weights for the last incomplete batch
    weights = (batch_size / n_samples, (n_samples % batch_size) / n_samples)
    if n_samples % batch_size == 0:
        weights = (batch_size / n_samples,) * 2

    # init the state of the sampler
    state = dict(
        batch_order=jnp.arange(n_batches),
        i_batch=jnp.array(0),
        key=key,
    )

    return (
        jax.jit(lambda state: _sampler(n_batches, batch_size, weights, state)),
        state
    )

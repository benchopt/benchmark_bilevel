import pytest
import jax
import jax.numpy as jnp

from benchopt.utils.safe_import import set_benchmark_module
set_benchmark_module('.')

from functools import partial  # noqa: E402
from solvers.saba import init_memory  # noqa: E402
from solvers.saba import variance_reduction  # noqa: E402
from benchmark_utils.minibatch_sampler import init_sampler  # noqa: E402


def loss_sample(inner_var, outer_var, x, y):
    return -jax.nn.log_sigmoid(y*jnp.dot(inner_var, x))


def loss(inner_var, outer_var, X, y):
    batched_loss = jax.vmap(loss_sample, in_axes=(None, None, 0, 0))
    return jnp.mean(batched_loss(inner_var, outer_var, X, y), axis=0)


def _get_function(n_samples, n_features):
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (n_samples, n_features))
    y = 2 * (jax.random.uniform(key, (n_samples,)) > 0.5) - 1

    @partial(jax.jit, static_argnames=('batch_size'))
    def f(inner_var, outer_var, start=0, batch_size=1):
        x = jax.lax.dynamic_slice(
            X, (start, 0), (batch_size, X.shape[1])
        )
        y_ = jax.lax.dynamic_slice(
            y, (start, ), (batch_size, )
        )
        res = loss(inner_var, outer_var, x, y_)
        return res

    return f


@pytest.mark.parametrize('batch_size', [1, 32, 64])
def test_init_memory(batch_size):
    n_samples = 1024
    n_features = 10

    f = partial(_get_function(n_samples, n_features), batch_size=batch_size)
    f_fb = partial(f, batch_size=n_samples)

    key = jax.random.PRNGKey(0)

    theta = jax.random.normal(key, (n_features,))
    lmbda = jax.random.normal(key, (n_features,))
    v = jax.random.normal(key, (n_features,))

    sampler_inner, state_sampler_inner = init_sampler(
        n_samples, batch_size=batch_size
    )
    sampler_outer, state_sampler_outer = init_sampler(
        n_samples, batch_size=batch_size
    )

    # Init memory
    memory = init_memory(
        f, f, theta, lmbda, v,
        n_inner_samples=n_samples, n_outer_samples=n_samples,
        batch_size_inner=batch_size, batch_size_outer=batch_size, mode='full',
        state_inner_sampler=state_sampler_inner,
        state_outer_sampler=state_sampler_outer,
        inner_size=n_features, outer_size=n_features
    )

    # check that the average gradients correspond to the true gradient.
    grad_inner, vjp_train = jax.vjp(
        lambda z, x: jax.grad(f_fb, argnums=0)(z, x),
        theta,  lmbda
    )
    hvp, cross_v = vjp_train(v)

    assert jnp.allclose(memory['inner_grad'][-2], grad_inner)
    assert jnp.allclose(memory['hvp'][-2], hvp)
    assert jnp.allclose(memory['cross_v'][-2], cross_v)
    assert jnp.allclose(memory['grad_in_outer'][-2], grad_inner)

    # check that the individual gradients for each sample are correct.
    for _ in range(len(state_sampler_inner['batch_order'])):
        (start_inner, id_inner,
         _, state_sampler_inner) = sampler_inner(state_sampler_inner)
        (start_outer, id_outer,
         _, state_sampler_outer) = sampler_outer(state_sampler_outer)
        grad_inner, vjp_train = jax.vjp(
            lambda z, x: jax.grad(f, argnums=0)(z, x, start=start_inner,
                                                batch_size=batch_size),
            theta, lmbda
        )
        hvp, cross_v = vjp_train(v)

        assert jnp.allclose(memory['inner_grad'][id_inner], grad_inner)
        assert jnp.allclose(memory['hvp'][id_inner], hvp)
        assert jnp.allclose(memory['cross_v'][id_inner], cross_v)
        assert jnp.allclose(memory['grad_in_outer'][id_outer],
                            jax.grad(f, argnums=0)(theta, lmbda,
                                                   start=start_outer,
                                                   batch_size=batch_size))


@pytest.mark.parametrize('batch_size', [1, 32, 64])
@pytest.mark.parametrize('n_samples', [1024])
def test_vr(n_samples, batch_size):
    n_features = 10

    f = partial(_get_function(n_samples, n_features), batch_size=batch_size)
    f_fb = partial(f, batch_size=n_samples)

    key = jax.random.PRNGKey(0)

    theta = jax.random.normal(key, (n_features,))
    lmbda = jax.random.normal(key, (n_features,))

    sampler, state_sampler = init_sampler(
        n_samples, batch_size=batch_size
    )
    n_batches = len(state_sampler['batch_order'])

    # check that variance reduction correctly stores the gradient
    memory = jnp.zeros((n_batches + 2, n_features))

    def check_mem(f, theta, lmbda, memory, sampler, state_sampler):
        for _ in range(n_batches):
            start, id, weights, state_sampler = sampler(state_sampler)
            grad_inner = jax.grad(f, argnums=0)(theta, lmbda,
                                                start=start,
                                                batch_size=batch_size)
            memory = variance_reduction(memory, grad_inner, id, weights)
        return memory

    memory = check_mem(f, theta, lmbda, memory, sampler, state_sampler)

    # check that after one epoch without moving, gradient average is correct
    grad_inner_avg = jax.grad(f_fb, argnums=0)(theta, lmbda)
    assert jnp.allclose(memory[-2], grad_inner_avg)

    # check that no part of the memory were altered by the variance
    for i in range(n_batches):
        start, id, weights, state_sampler = sampler(state_sampler)
        grad_inner = jax.grad(f, argnums=0)(theta, lmbda,
                                            start=start,
                                            batch_size=batch_size)
        assert jnp.allclose(memory[id], grad_inner)

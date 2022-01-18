import pytest
import numpy as np

from benchopt.utils.safe_import import set_benchmark

set_benchmark(".")

from objective import oracles  # noqa: E402
from solvers.saba import init_memory  # noqa: E402
from solvers.saba import MinibatchSampler  # noqa: E402
from solvers.saba import variance_reduction  # noqa: E402


def _make_oracle(n_samples, n_features):

    X = np.random.randn(n_samples, n_features)
    y = 2 * (np.random.rand(n_samples) > 0.5) - 1

    oracle = oracles.LogisticRegressionOracle(X, y, reg="exp")
    return oracle


@pytest.mark.parametrize("batch_size", [1, 32, 64])
def test_init_memory(batch_size):
    n_samples = 1024
    n_features = 10

    oracle = _make_oracle(n_samples, n_features)

    theta = np.random.randn(n_features)
    lmbda = np.random.randn(n_features)
    v = np.random.randn(n_features)

    sampler = MinibatchSampler(n_samples, batch_size=batch_size)

    # Init memory
    (memory_inner_grad, memory_hvp, memory_cross_v, memory_grad_in_outer) = init_memory(
        oracle.numba_oracle, oracle.numba_oracle, theta, lmbda, v, sampler, sampler
    )

    # check that the average gradients correspond to the true gradient.
    _, grad_inner, hvp, cross_v = oracle.get_oracles(theta, lmbda, v, inverse="id")

    assert np.allclose(memory_inner_grad[-1], grad_inner)
    assert np.allclose(memory_hvp[-1], hvp)
    assert np.allclose(memory_cross_v[-1], cross_v)
    assert np.allclose(memory_grad_in_outer[-1], grad_inner)

    # check that the individual gradients for each sample are correct.
    for i in range(sampler.n_batches):
        _, grad_inner, hvp, cross_v = oracle.oracles(
            theta, lmbda, v, slice(i * batch_size, (i + 1) * batch_size), inverse="id"
        )

        assert np.allclose(memory_inner_grad[i], grad_inner)
        assert np.allclose(memory_hvp[i], hvp)
        assert np.allclose(memory_cross_v[i], cross_v)
        assert np.allclose(memory_grad_in_outer[i], grad_inner)


@pytest.mark.parametrize("batch_size", [1, 32, 64])
def test_vr(batch_size):
    n_samples = 1024
    n_features = 10

    oracle = _make_oracle(n_samples, n_features)

    theta = np.random.randn(n_features)
    lmbda = np.random.randn(n_features)

    # check that variance reduction correctly stores the gradient
    n_batches = n_samples // batch_size
    memory = np.zeros((n_batches + 1, n_features))
    for i in range(n_batches):
        grad_inner = oracle.grad_inner_var(
            theta,
            lmbda,
            slice(i * batch_size, (i + 1) * batch_size),
        )
        variance_reduction(grad_inner, memory, i)
        assert np.allclose(memory[i], grad_inner)

    # check that after one epoch without moving, gradient average is correct
    grad_inner_avg = oracle.get_grad_inner_var(theta, lmbda)
    assert np.allclose(memory[-1], grad_inner_avg)

    # check that no part of the memory were altered by the variance
    for i in range(n_batches):
        grad_inner = oracle.grad_inner_var(
            theta,
            lmbda,
            slice(i * batch_size, (i + 1) * batch_size),
        )
        assert np.allclose(memory[i], grad_inner)

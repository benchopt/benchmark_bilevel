import pytest

from benchopt.utils.safe_import import set_benchmark_module
set_benchmark_module('.')

from benchmark_utils.minibatch_sampler import init_sampler  # noqa: E402


@pytest.mark.parametrize('n_samples', [100])
@pytest.mark.parametrize('batch_size', [23])
def test_lr_scheduler(n_samples, batch_size):
    sampler, state_sampler = init_sampler(n_samples, batch_size)
    for k in range(10):
        selector, _, _, state_sampler = sampler(state_sampler)

import pytest
import jax.numpy as jnp

from benchopt.utils.safe_import import set_benchmark_module
set_benchmark_module('.')

import jax  # noqa: E402
from benchmark_utils.learning_rate_scheduler import update_lr  # noqa: E402
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize('constants', [jnp.array([1., 1.]),
                                       jnp.array([2., 2.])])
@pytest.mark.parametrize('exponents', [jnp.array([0., 0.]),
                                       jnp.array([1/2, 1/2]),
                                       jnp.array([1., 1.])])
def test_lr_scheduler(constants, exponents):
    state_lr = dict(
        constants=constants,
        exponents=exponents,
        i_step=1,
    )

    for k in range(1, 6):
        lr = constants / ((k+1) ** exponents)
        lr_, state_lr = update_lr(state_lr)
        assert jnp.allclose(lr - lr_, 0)

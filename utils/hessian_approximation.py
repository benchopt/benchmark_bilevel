import numpy as np
from numba import njit


def hia(inner_oracle, inner_var, outer_var, v, inner_sampler, n_step, step_size):
    """Hessian Inverse Approximation subroutine from [Ghadimi2018].

    This implement Algorithm.3
    """
    p = np.random.randint(n_step)
    for i in range(p):
        inner_slice, _ = inner_sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * hvp
    return n_step * step_size * v


def shia(inner_oracle, inner_var, outer_var, v, inner_sampler, n_step, step_size):
    """Hessian Inverse Approximation subroutine from [Ji2021].

    This implement Algorithm.3
    """
    s = v
    for i in range(n_step):
        inner_slice, _ = inner_sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * hvp
        s += v
    return step_size * s


def sgd_v(
    inner_oracle, inner_var, outer_var, v, grad_out, inner_sampler, n_step, step_size
):
    r"""SGD for the inverse Hessian approximation.

    This function solves the following problem

    .. math::

        \min_v v^\top H v - \nabla_{out}^\top v
    """
    for _ in range(n_step):
        inner_slice, _ = inner_sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * (hvp - grad_out)

    return v

import numpy as np


def hia(inner_oracle, inner_var, outer_var, v, inner_sampler,
        n_step, step_size):
    """Hessian Inverse Approximation subroutine from [Ghadimi2018].

    This implement Algorithm.3
    """
    p = np.random.randint(n_step)
    for i in range(p):
        inner_slice, _ = inner_sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * hvp
    return n_step * step_size * v


def shia(
    inner_oracle, inner_var, outer_var, v, inner_sampler, n_step, step_size
):
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


def sgd_v(inner_oracle, inner_var, outer_var, v, grad_out,
          inner_sampler, n_step, step_size):
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


def joint_shia(
    inner_oracle, inner_var, outer_var, v, inner_var_old, outer_var_old, v_old,
    inner_sampler, n_step, step_size, seed=None
):
    """Hessian Inverse Approximation subroutine from [Ji2021].

    This implement Algorithm.3
    """
    s = v
    s_old = v_old
    for i in range(n_step):
        inner_slice, _ = inner_sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * hvp
        s += v
        hvp_old = inner_oracle.hvp(
            inner_var_old, outer_var_old, v_old, inner_slice
        )
        v_old -= step_size * hvp_old
        s_old += v_old
    return step_size * v, step_size * v_old


def joint_hia(inner_oracle, inner_var, outer_var, v,
              inner_var_old, outer_var_old, v_old,
              inner_sampler, n_step, step_size):
    """Hessian Inverse Approximation subroutine from [Ghadimi2018].

    This is a modification that jointly compute the HIA with the same samples
    for the current estimates and the one from the previous iteration, in
    order to compute the momentum term.
    """
    p = np.random.randint(n_step)
    for i in range(p):
        inner_slice, _ = inner_sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * hvp
        hvp_old = inner_oracle.hvp(
            inner_var_old, outer_var_old, v_old, inner_slice
        )
        v_old -= step_size * hvp_old
    return n_step * step_size * v, n_step * step_size * v_old

from numba import njit


def sgd_inner(
    inner_oracle, inner_var, outer_var, step_size, inner_sampler, n_inner_step
):
    for _ in range(n_inner_step):
        inner_slice, _ = inner_sampler.get_batch()
        grad_inner = inner_oracle.grad_inner_var(inner_var, outer_var, inner_slice)
        inner_var -= step_size * grad_inner

    return inner_var

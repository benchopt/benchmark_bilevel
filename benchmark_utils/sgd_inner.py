import jax


def sgd_inner_jax(inner_var, outer_var, state_sampler, step_size,
                  sampler=None, n_steps=1, grad_inner=None):
    """
    Jax implementation of stochastic gradient descent on the inner problem.

    Parameters
    ----------
    inner_var : array
        Initial value of the inner variable.
    outer_var : array
        Value of the outer variable.
    state_sampler : dict
        State of the sampler.
    step_size : float
        Step size of the gradient descent.
    sampler : callable
        Sampler for the inner problem.
    n_steps : int
        Number of steps of the gradient descent.
    grad_inner : callable
        Gradient of the inner oracle with respect to the inner variable.
    """
    def iter(i, args):
        state_sampler, inner_var = args
        start_idx, *_, state_sampler = sampler(state_sampler)
        inner_var -= step_size * grad_inner(inner_var, outer_var, start_idx)
        return state_sampler, inner_var
    state_sampler, inner_var = jax.lax.fori_loop(0, n_steps, iter,
                                                 (state_sampler, inner_var))

    return inner_var, state_sampler

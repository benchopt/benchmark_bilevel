import jax
from functools import partial
from benchmark_utils.tree_utils import update_sgd_fn


@partial(jax.jit, static_argnames=('grad_inner', 'n_steps'))
def gd_inner_jax(inner_var, outer_var, step_size, grad_inner=None,
                 n_steps=1):
    """
    Jax implementation of gradient descent.

    Parameters
    ----------
    grad_inner : callable
        Gradient of the inner oracle with respect to the inner variable.
    inner_var : pytree
        Initial value of the inner variable.
    outer_var : pytree
        Value of the outer variable.
    step_size : float
        Step size of the gradient descent.
    n_steps : int
        Number of steps of the gradient descent.

    Returns
    -------
    inner_var : array
        Value of the inner variable after n_steps of gradient descent.
    """
    def iter(_, inner_var):
        inner_var = update_sgd_fn(inner_var, grad_inner(inner_var, outer_var),
                                  step_size)
        return inner_var
    inner_var = jax.lax.fori_loop(0, n_steps, iter, inner_var)
    return inner_var

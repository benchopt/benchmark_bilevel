import jax
from functools import partial


@partial(jax.jit, static_argnames=('grad_inner', 'n_steps'))
def gd_inner_jax(inner_var, outer_var, step_size, grad_inner=None,
                 n_steps=1):
    """
    Jax implementation of gradient descent.

    Parameters
    ----------
    grad_inner : callable
        Gradient of the inner oracle with respect to the inner variable.
    inner_var : array
        Initial value of the inner variable.
    outer_var : array
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
    def iter(i, inner_var):
        inner_var -= step_size * grad_inner(inner_var, outer_var)
        return inner_var
    inner_var = jax.lax.fori_loop(0, n_steps, iter, inner_var)
    return inner_var

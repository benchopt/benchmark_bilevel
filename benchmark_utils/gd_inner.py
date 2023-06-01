import jax
from functools import partial


def gd_inner(inner_oracle, inner_var, outer_var, step_size,
             n_steps=1):

    for i in range(n_steps):
        grad_inner = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice(None)
        )
        inner_var -= step_size * grad_inner

    return inner_var


@partial(jax.jit, static_argnums=(0, ), static_argnames=('n_steps'))
def gd_inner_jax(grad_inner, inner_var, outer_var, step_size,
                 n_steps=1):
    def iter(i, inner_var):
        inner_var -= step_size * grad_inner(inner_var, outer_var)
        return inner_var
    inner_var = jax.lax.fori_loop(0, n_steps, iter, inner_var)
    return inner_var

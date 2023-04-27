import numpy as np

import jax
from functools import partial


def hia(inner_oracle, inner_var, outer_var, v, step_size, n_steps=1,
        sampler=None):
    """Hessian Inverse Approximation subroutine from [Ghadimi2018].

    This implement Algorithm.3
    """
    p = np.random.randint(n_steps)
    for i in range(p):
        inner_slice, _ = sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * hvp
    return n_steps * step_size * v


@partial(jax.jit, static_argnums=(0, ), static_argnames=('sampler', 'n_steps'))
def hia_jax(
    grad_inner, inner_var, outer_var, v, state_sampler, step_size,
    sampler=None, n_steps=1, key=None
):
    """Hessian Inverse Approximation subroutine from [Ghadimi2018]
    (jax version).

    This implement Algorithm.3
    """
    p = jax.random.randint(key, shape=(1,), minval=0, maxval=n_steps)

    def hvp(v, start):
        _, hvp_fun = jax.vjp(
            lambda z: grad_inner(z, outer_var, start), inner_var
        )
        return hvp_fun(v)[0]

    def iter(i, args):
        start, args[0] = sampler(**args[0])
        args[1] -= step_size * hvp(args[1], start)
        return args
    res = jax.lax.fori_loop(0, p[0], iter, [state_sampler, v])
    return n_steps * step_size * res[1], jax.random.split(key, 1)[0], res[0]


def shia(
    inner_oracle, inner_var, outer_var, v, step_size, sampler=None, n_steps=1
):
    """Hessian Inverse Approximation subroutine from [Ji2021].

    This implement Algorithm.3
    """
    s = v.copy()
    for _ in range(n_steps):
        inner_slice, _ = sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * hvp
        s += v
    return step_size * s


@partial(jax.jit, static_argnums=(0, ), static_argnames=('sampler', 'n_steps'))
def shia_jax(
    grad_inner, inner_var, outer_var, v, state_sampler, step_size,
    sampler=None, n_steps=1
):
    """Hessian Inverse Approximation subroutine from [Ji2021] (jax version).

    This implement Algorithm.3
    """
    s = v.copy()

    def hvp(v, start):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var, start), inner_var
            )
        return hvp_fun(v)[0]

    def iter(i, args):
        start, args[0] = sampler(**args[0])
        args[1] -= step_size * hvp(args[1], start)
        args[2] += args[1]
        return args
    res = jax.lax.fori_loop(0, n_steps, iter, [state_sampler, v, s])
    return step_size * res[2], res[0]


def shia_fb(
    inner_oracle, inner_var, outer_var, v, n_step, step_size
):
    """Hessian Inverse Approximation subroutine from [Ji2021].

    This implement Algorithm.3
    """
    s = v.copy()
    for i in range(n_step):
        inner_slice = slice(0, inner_oracle.n_samples)
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * hvp
        s += v
    return step_size * s


def sgd_v(inner_oracle, inner_var, outer_var, v, grad_out,
          step_size, sampler=None, n_steps=1):
    r"""SGD for the inverse Hessian approximation.

    This function solves the following problem

    .. math::

        \min_v v^\top H v - \nabla_{out}^\top v
    """
    for _ in range(n_steps):
        inner_slice, _ = sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * (hvp - grad_out)
    return v


@partial(jax.jit, static_argnums=(0, ), static_argnames=('sampler', 'n_steps'))
def sgd_v_jax(grad_inner, inner_var, outer_var, v, grad_out, state_sampler,
              step_size, sampler=None, n_steps=1):
    r"""SGD for the inverse Hessian approximation.

    This function solves the following problem

    .. math::

        \min_v v^\top H v - \nabla_{out}^\top v
    """
    def hvp(v, start):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var, start), inner_var
            )
        return hvp_fun(v)[0]

    def iter(i, args):
        start, args[0] = sampler(**args[0])
        args[1] -= step_size * (hvp(args[1], start) - grad_out)
        return args
    res = jax.lax.fori_loop(0, n_steps, iter, [state_sampler, v])
    return res[1], res[0]


def joint_shia(
    inner_oracle, inner_var, outer_var, v, inner_var_old, outer_var_old, v_old,
    step_size, sampler=None, n_steps=1
):
    """Hessian Inverse Approximation subroutine from [Ji2021].

    This implement Algorithm.3
    """
    s = v.copy()
    s_old = v_old.copy()
    for _ in range(n_steps):
        inner_slice, _ = sampler.get_batch()
        hvp = inner_oracle.hvp(inner_var, outer_var, v, inner_slice)
        v -= step_size * hvp
        s += v
        hvp_old = inner_oracle.hvp(
            inner_var_old, outer_var_old, v_old, inner_slice
        )
        v_old -= step_size * hvp_old
        s_old += v_old
    return step_size * s, step_size * s_old


@partial(jax.jit, static_argnums=(0, ), static_argnames=('sampler', 'n_steps'))
def joint_shia_jax(
    grad_inner, inner_var, outer_var, v, inner_var_old, outer_var_old, v_old,
    state_sampler, step_size, sampler=None, n_steps=1
):
    """Hessian Inverse Approximation subroutine from [Ji2021] (jax version).

    This implement Algorithm.3
    """
    s = v.copy()
    s_old = v_old.copy()

    def hvp(v, start):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var, start), inner_var
            )
        return hvp_fun(v)[0]

    def hvp_old(v, start):
        _, hvp_fun = jax.vjp(
            lambda z: grad_inner(z, outer_var_old, start), inner_var_old
        )
        return hvp_fun(v)[0]

    def iter(i, args):
        start, args[0] = sampler(**args[0])
        args[1] -= step_size * hvp(args[1], start)
        args[2] += args[1]
        args[3] -= step_size * hvp_old(args[3], start)
        args[4] += args[3]
        return args
    res = jax.lax.fori_loop(
        0, n_steps, iter, [state_sampler, v, s, v_old, s_old]
    )

    return step_size * res[2], step_size * res[4], res[0]


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


@partial(jax.jit, static_argnums=(0, ), static_argnames=('sampler', 'n_steps'))
def joint_hia_jax(
    grad_inner, inner_var, outer_var, v, inner_var_old, outer_var_old, v_old,
    state_sampler, step_size, sampler=None, n_steps=1, key=None
):
    """Hessian Inverse Approximation subroutine from [Ji2021] (jax version).

    This implement Algorithm.3
    """
    s = v.copy()
    s_old = v_old.copy()
    p = jax.random.randint(key, shape=(1,), minval=0, maxval=n_steps)

    def hvp(v, start):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var, start), inner_var
            )
        return hvp_fun(v)[0]

    def hvp_old(v, start):
        _, hvp_fun = jax.vjp(
            lambda z: grad_inner(z, outer_var_old, start), inner_var_old
        )
        return hvp_fun(v)[0]

    def iter(i, args):
        start, args[0] = sampler(**args[0])
        args[1] -= step_size * hvp(args[1], start)
        args[2] -= step_size * hvp_old(args[2], start)
        return args
    res = jax.lax.fori_loop(
        0, p[0], iter, [state_sampler, v, s, v_old, s_old]
    )

    return n_steps * step_size * res[1], n_steps * step_size * res[2], \
        jax.random.split(key, 1)[0], res[0]

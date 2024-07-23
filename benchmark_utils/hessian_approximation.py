import jax
from benchmark_utils.tree_utils import update_sgd_fn
from benchmark_utils.tree_utils import tree_scalar_mult, tree_add


def hia_jax(
    inner_var, outer_var, v, state_sampler, step_size,
    sampler=None, n_steps=1, key=jax.random.PRNGKey(1), grad_inner=None
):
    """Hessian Inverse Approximation subroutine from [Ghadimi2018] with
    stochastic Neumann iterations. Aims at approximating the solution
    of the linear system

    .. math:: \nabla^2 f_inner(inner_var, outer_var) x = v


    This implement Algorithm.3 in [Ghadimi2018].

    Parameters
    ----------
    inner_var : array
        Inner variable.

    outer_var : array
        Outer variable.

    v : array
        Right hand side of the linear system.

    state_sampler : dict
        State of the sampler.

    step_size : float
        Step size.

    sampler : callable
        Sampler for the inner problem.

    n_steps : int
        Number of iterations.

    key : jax PRNGKey
        Key for randomness.

    grad_inner : callable
        Gradient of the inner oracle with respect to the inner variable.
    """
    p = jax.random.randint(key, shape=(1,), minval=0, maxval=n_steps)

    def hvp(v, start_idx):
        _, hvp_fun = jax.vjp(
            lambda z: grad_inner(z, outer_var, start_idx), inner_var
        )
        return hvp_fun(v)[0]

    def iter(_, args):
        state_sampler, v = args
        start_idx, *_, state_sampler = sampler(state_sampler)
        v = update_sgd_fn(v, hvp(v, start_idx), step_size)
        return state_sampler, v
    state_sampler, v = jax.lax.fori_loop(0, p[0], iter, (state_sampler, v))
    v = tree_scalar_mult(n_steps * step_size, v)
    return v, jax.random.split(key, 1)[0], state_sampler


def shia_jax(
    inner_var, outer_var, v, state_sampler, step_size,
    sampler=None, n_steps=1, grad_inner=None
):
    """Hessian Inverse Approximation subroutine from [Ji2021] with
    stochastic Neumann iterations. Aims at approximating the solution
    of the linear system

    .. math:: \nabla^2 f_inner(inner_var, outer_var) x = v


    This implement Algorithm.3 in [Ji2021].

    Parameters
    ----------
    inner_var : pytree
        Inner variable.

    outer_var : pytree
        Outer variable.

    v : pytree
        Right hand side of the linear system.

    state_sampler : dict
        State of the sampler.

    step_size : float
        Step size.

    sampler : callable
        Sampler for the inner problem.

    n_steps : int
        Number of iterations.

    key : jax PRNGKey
        Key for randomness.

    grad_inner : callable
        Gradient of the inner oracle with respect to the inner variable.
    """
    s = v.copy()

    def hvp(v, start_idx):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var, start_idx), inner_var
            )
        return hvp_fun(v)[0]

    def iter(_, args):
        state_sampler, v, s = args
        start_idx, *_, state_sampler = sampler(state_sampler)
        v = update_sgd_fn(v, hvp(v, start_idx), step_size)
        s = update_sgd_fn(s, v, -1)  # s += v
        return state_sampler, v, s
    state_sampler, _, s = jax.lax.fori_loop(0, n_steps, iter,
                                            (state_sampler, v, s))
    return step_size * s, state_sampler


def shia_fb_jax(inner_var, outer_var, v, step_size, n_steps=1,
                grad_inner=None):
    """Hessian Inverse Approximation subroutine from [Ji2021] with
    stochastic Neumann iterations with full batch oracles. Aims at
    approximating the solution of the linear system

    .. math:: \nabla^2 f_inner(inner_var, outer_var) x = v


    This implement Algorithm.3 in [Ji2021].

    Parameters
    ----------
    inner_var : array
        Inner variable.

    outer_var : array
        Outer variable.

    v : array
        Right hand side of the linear system.

    step_size : float
        Step size.

    n_steps : int
        Number of iterations.

    grad_inner : callable
        Gradient of the inner oracle with respect to the inner variable.
    """
    s = v.copy()

    def hvp(v):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var), inner_var
            )
        return hvp_fun(v)[0]

    def iter(_, args):
        v, s = args
        v = update_sgd_fn(v, hvp(v), step_size)
        s = update_sgd_fn(s, v, -1)  # s += v
        return v, s
    _, s = jax.lax.fori_loop(0, n_steps, iter, (v, s))
    return tree_scalar_mult(step_size, s)


def sgd_v_jax(inner_var, outer_var, v, grad_out, state_sampler,
              step_size, sampler=None, n_steps=1, grad_inner=None):
    r"""SGD for the inverse Hessian approximation.

    This function solves the following problem

    .. math::

        \min_v v^\top H v - \nabla_{out}^\top v

    where :math:`H` is the Hessian of the inner oracle with respect to
    the inner variable.

    Parameters
    ----------
    inner_var : array
        Inner variable.

    outer_var : array
        Outer variable.

    v : array
        Initial guess for the solution.

    grad_out : array
        Gradient of the outer function with respect to the inner variable.

    state_sampler : dict
        State of the sampler.

    step_size : float
        Step size.

    sampler : callable
        Sampler for the inner problem.

    n_steps : int
        Number of iterations.

    grad_inner : callable
        Gradient of the inner function with respect to the inner variable.
    """
    def hvp(v, start_idx):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var, start_idx), inner_var
            )
        return hvp_fun(v)[0]

    def iter(_, args):
        state_sampler, v = args
        start_idx, *_, state_sampler = sampler(state_sampler)
        v = update_sgd_fn(v,
                          tree_add(hvp(v, start_idx),
                                   tree_scalar_mult(-1, grad_out)),
                          step_size)
        return state_sampler, v
    state_sampler, v = jax.lax.fori_loop(0, n_steps, iter, (state_sampler, v))
    return v, state_sampler


def joint_shia_jax(
    inner_var, outer_var, v, inner_var_old, outer_var_old, v_old,
    state_sampler, step_size, sampler=None, n_steps=1, grad_inner=None
):
    """Hessian Inverse Approximation subroutine from [Ji2021] with
        stochastic Neumann iterations with full batch oracles. Aims at
        approximating the solution of the linear systems

        .. math:: \nabla^2 f_inner(inner_var, outer_var) x = v

        and

        .. math:: \nabla^2 f_inner(inner_var_old, outer_var_old) x = v_old


        This implement Algorithm.3 in [Ji2021].

        Parameters
        ----------
        inner_var : array
            Inner variable for the first linear system.

        outer_var : array
            Outer variable for the first linear system.

        v : array
            Right hand side of the first linear system.

        inner_var_old : array
            Inner variable for the second linear system.

        outer_var_old : array
            Outer variable for the second linear system.

        v_old : array
            Right hand side of the second linear system.

        step_size : float
            Step size.

        n_steps : int
            Number of iterations.

        grad_inner : callable
            Gradient of the inner oracle with respect to the inner variable.
        """
    s = v.copy()
    s_old = v_old.copy()

    def hvp(v, start_idx):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var, start_idx), inner_var
            )
        return hvp_fun(v)[0]

    def hvp_old(v, start_idx):
        _, hvp_fun = jax.vjp(
            lambda z: grad_inner(z, outer_var_old, start_idx), inner_var_old
        )
        return hvp_fun(v)[0]

    def iter(_, args):
        state_sampler, v, s, v_old, s_old = args
        start_idx, *_, state_sampler = sampler(state_sampler)
        v = update_sgd_fn(v, hvp(v, start_idx), step_size)
        s = update_sgd_fn(s, v, -1)  # s += v
        v_old = update_sgd_fn(v_old, hvp_old(v_old, start_idx), step_size)
        s_old = update_sgd_fn(s_old, v_old, -1)  # s_old += v_old
        return state_sampler, v, s, v_old, s_old

    state_sampler, _, s, _, s_old = jax.lax.fori_loop(
        0, n_steps, iter, (state_sampler, v, s, v_old, s_old)
    )
    return (
        tree_scalar_mult(step_size, s), tree_scalar_mult(step_size, s_old),
        state_sampler
    )


def joint_hia_jax(
    inner_var, outer_var, v, inner_var_old, outer_var_old, v_old,
    state_sampler, step_size, sampler=None, n_steps=1,
    key=jax.random.PRNGKey(1), grad_inner=None
):
    """Hessian Inverse Approximation subroutine from [Ghadimi2018] with
    stochastic Neumann iterations. Aims at approximating the solution
    of the linear systems

    .. math:: \nabla^2 f_inner(inner_var, outer_var) x = v

    and

    .. math:: \nabla^2 f_inner(inner_var_old, outer_var_old) x = v_old


    This implement Algorithm.3 in [Ghadimi2018].

    Parameters
    ----------
    inner_var : array
        Inner variable for the first linear system.

    outer_var : array
        Outer variable for the first linear system.

    v : array
        Right hand side of the first linear system.

    inner_var_old : array
        Inner variable for the second linear system.

    outer_var_old : array
        Outer variable for the second linear system.

    v_old : array
        Right hand side of the second linear system.

    state_sampler : dict
        State of the sampler.

    step_size : float
        Step size.

    sampler : callable
        Sampler for the inner problem.

    n_steps : int
        Number of iterations.

    key : jax PRNGKey
        Key for randomness.

    grad_inner : callable
        Gradient of the inner oracle with respect to the inner variable.
    """
    p = jax.random.randint(key, shape=(1,), minval=0, maxval=n_steps)

    def hvp(v, start_idx):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var, start_idx), inner_var
            )
        return hvp_fun(v)[0]

    def hvp_old(v, start_idx):
        _, hvp_fun = jax.vjp(
            lambda z: grad_inner(z, outer_var_old, start_idx), inner_var_old
        )
        return hvp_fun(v)[0]

    def iter(_, args):
        state_sampler, v, v_old = args
        start_idx, *_, state_sampler = sampler(state_sampler)
        v = update_sgd_fn(v, hvp(v, start_idx), step_size)
        v_old = update_sgd_fn(v_old, hvp_old(v_old, start_idx), start_idx)
        return state_sampler, v, v_old

    state_sampler, v, v_old = jax.lax.fori_loop(
        0, p[0], iter, (state_sampler, v, v_old)
    )

    return (
        tree_scalar_mult(n_steps * step_size, v),
        tree_scalar_mult(n_steps * step_size, v_old),
        jax.random.split(key, 1)[0], state_sampler
    )

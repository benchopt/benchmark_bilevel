import jax


def hia_jax(
    inner_var, outer_var, v, state_sampler, step_size,
    sampler=None, n_steps=1, key=jax.random.PRNGKey(1), grad_inner=None
):
    """Hessian Inverse Approximation subroutine from [Ghadimi2018] with
    stochastic Neumann iterations (jax version).

    This implement Algorithm.3
    """
    p = jax.random.randint(key, shape=(1,), minval=0, maxval=n_steps)

    def hvp(v, start_idx):
        _, hvp_fun = jax.vjp(
            lambda z: grad_inner(z, outer_var, start_idx), inner_var
        )
        return hvp_fun(v)[0]

    def iter(i, args):
        state_sampler, v = args
        start_idx, *_, state_sampler = sampler(state_sampler)
        v -= step_size * hvp(v, start_idx)
        return state_sampler, v
    state_sampler, v = jax.lax.fori_loop(0, p[0], iter, (state_sampler, v))
    return n_steps * step_size * v, jax.random.split(key, 1)[0], state_sampler


def shia_jax(
    inner_var, outer_var, v, state_sampler, step_size,
    sampler=None, n_steps=1, grad_inner=None
):
    """Hessian Inverse Approximation subroutine from [Ji2021] with
    stochastic Neumann iterations (jax version).

    This implement Algorithm.3
    """
    s = v.copy()

    def hvp(v, start_idx):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var, start_idx), inner_var
            )
        return hvp_fun(v)[0]

    def iter(i, args):
        state_sampler, v, s = args
        start_idx, *_, state_sampler = sampler(state_sampler)
        v -= step_size * hvp(v, start_idx)
        s += v
        return state_sampler, v, s
    state_sampler, _, s = jax.lax.fori_loop(0, n_steps, iter,
                                            (state_sampler, v, s))
    return step_size * s, state_sampler


def shia_fb_jax(inner_var, outer_var, v, step_size, n_steps=1,
                grad_inner=None):
    """Hessian Inverse Approximation subroutine from [Ji2021] with
    stochastic Neumann iterations (jax version).

    This implement Algorithm.3
    """
    s = v.copy()

    def hvp(v):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var), inner_var
            )
        return hvp_fun(v)[0]

    def iter(i, args):
        v, s = args
        v -= step_size * hvp(v)
        s += v
        return v, s
    _, s = jax.lax.fori_loop(0, n_steps, iter, (v, s))
    return step_size * s


def sgd_v_jax(inner_var, outer_var, v, grad_out, state_sampler,
              step_size, sampler=None, n_steps=1, grad_inner=None):
    r"""SGD for the inverse Hessian approximation.

    This function solves the following problem

    .. math::

        \min_v v^\top H v - \nabla_{out}^\top v
    """
    def hvp(v, start_idx):
        _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, outer_var, start_idx), inner_var
            )
        return hvp_fun(v)[0]

    def iter(i, args):
        state_sampler, v = args
        start_idx, *_, state_sampler = sampler(state_sampler)
        v -= step_size * (hvp(v, start_idx) - grad_out)
        return state_sampler, v
    state_sampler, v = jax.lax.fori_loop(0, n_steps, iter, (state_sampler, v))
    return v, state_sampler


def joint_shia_jax(
    inner_var, outer_var, v, inner_var_old, outer_var_old, v_old,
    state_sampler, step_size, sampler=None, n_steps=1, grad_inner=None
):
    """Hessian Inverse Approximation subroutine from [Ji2021] with
    stochastic Neumann iterations (jax version).

    This implement Algorithm.3
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

    def iter(i, args):
        state_sampler, v, s, v_old, s_old = args
        start_idx, *_, state_sampler = sampler(state_sampler)
        v -= step_size * hvp(v, start_idx)
        s += v
        v_old -= step_size * hvp_old(v_old, start_idx)
        s_old += v_old
        return state_sampler, v, s, v_old, s_old
    state_sampler, _, s, _, s_old = jax.lax.fori_loop(
        0, n_steps, iter, (state_sampler, v, s, v_old, s_old)
    )

    return step_size * s, step_size * s_old, state_sampler


def joint_hia_jax(
    inner_var, outer_var, v, inner_var_old, outer_var_old, v_old,
    state_sampler, step_size, sampler=None, n_steps=1,
    key=jax.random.PRNGKey(1), grad_inner=None
):
    """Hessian Inverse Approximation subroutine from [Ji2021] with
    stochastic Neumann iterations (jax version).

    This implement Algorithm.3
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

    def iter(i, args):
        state_sampler, v, v_old = args
        start_idx, *_, state_sampler = sampler(state_sampler)
        v -= step_size * hvp(v, start_idx)
        v_old -= step_size * hvp_old(v_old, start_idx)
        return state_sampler, v, v_old
    state_sampler, v, v_old = jax.lax.fori_loop(
        0, p[0], iter, (state_sampler, v, v_old)
    )

    return n_steps * step_size * v, n_steps * step_size * v_old, \
        jax.random.split(key, 1)[0], state_sampler

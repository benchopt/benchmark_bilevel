import jax
from functools import partial


def sgd_inner(inner_oracle, inner_var, outer_var, step_size, sampler=None,
              n_steps=1):
    """
    Perform stochastic gradient descent on the inner problem.
    """

    for i in range(n_steps):
        inner_slice, _ = sampler.get_batch()
        grad_inner = inner_oracle.grad_inner_var(
            inner_var, outer_var, inner_slice
        )
        inner_var -= step_size * grad_inner

    return inner_var


@partial(jax.jit, static_argnames=('sampler', 'n_steps', 'grad_inner'))
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


def sgd_inner_vrbo(joint_shia, inner_oracle, outer_oracle, inner_var,
                   outer_var, inner_lr, inner_sampler, outer_sampler,
                   n_steps, memory_inner, memory_outer, n_shia_steps,
                   hia_lr):
    for i in range(n_steps):
        # Step.4.k.1 - Update direction for z
        slice_inner, _ = inner_sampler.get_batch()
        grad_inner_var = inner_oracle.grad_inner_var(
            inner_var, outer_var, slice_inner
        )
        grad_inner_var_old = inner_oracle.grad_inner_var(
            memory_inner[0], outer_var, slice_inner
        )
        memory_inner[1] += grad_inner_var - grad_inner_var_old

        # Step.4.k.2 - Update direction for x
        slice_outer, _ = outer_sampler.get_batch()
        grad_outer, impl_grad = outer_oracle.grad(
            inner_var, outer_var, slice_outer
        )
        grad_outer_old, impl_grad_old = outer_oracle.grad(
            memory_inner[0], outer_var, slice_outer
        )
        ihvp, ihvp_old = joint_shia(
            inner_oracle, inner_var, outer_var, grad_outer,
            memory_inner[0], outer_var, grad_outer_old,
            hia_lr, sampler=inner_sampler, n_steps=n_shia_steps
        )
        impl_grad -= inner_oracle.cross(
            inner_var, outer_var, ihvp, slice_inner
        )
        impl_grad_old -= inner_oracle.cross(
            memory_inner[0], outer_var, ihvp_old, slice_inner
        )
        memory_outer[1] += impl_grad - impl_grad_old

        # Step.4.k.3 - update the inner variable and memory
        memory_inner[0] = inner_var
        inner_var -= inner_lr * memory_inner[1]

    return inner_var, outer_var, memory_inner, memory_outer


@partial(jax.jit, static_argnames=('sampler', 'n_steps', 'joint_shia',
                                   'inner_sampler', 'outer_sampler',
                                   'n_shia_steps', 'grad_inner_fun',
                                   'grad_outer_fun'))
def sgd_inner_vrbo_jax(inner_var,
                       outer_var, inner_var_old, d_inner, d_outer,
                       state_inner_sampler, state_outer_sampler, step_size,
                       shia_lr, n_shia_steps=1, inner_sampler=None,
                       outer_sampler=None, joint_shia=None, n_steps=1,
                       grad_inner_fun=None, grad_outer_fun=None):
    """
    Jax implementation of the inner routine of VRBO.

    Parameters
    ----------
    inner_var : array
        Initial value of the inner variable.
    outer_var : array
        Value of the outer variable.
    inner_var_old : array
        Value of the inner variable at the previous iteration.
    d_inner : array
        Direction for the inner variable.
    d_outer : array
        Direction for the outer variable.
    state_inner_sampler : dict
        State of the sampler for the inner function.
    state_outer_sampler : dict
        State of the sampler for the outer function.
    step_size : float
        Step size of the inner problem.
    shia_lr : float
        Learning rate for inverse Hessian approximation.
    n_shia_steps : int
        Number of steps of the inverse Hessian approximation.
    inner_sampler : callable
        Sampler for the inner problem.
    outer_sampler : callable
        Sampler for the outer problem.
    joint_shia : callable
        Implementation of the inverse Hessian approximation performed jointly
        on the current and previous inner variables.
    n_steps : int
        Number of steps of the inner problem.
    grad_inner_fun : callable
        Gradient of the inner oracle with respect to the inner variable.
    grad_outer_fun : callable
        Gradient of the outer oracle with respect to the inner variable and
        the outer variable.

    Returns
    -------
    inner_var : array
        Value of the inner variable after n_steps of gradient descent.
    inner_var_old : array
        Value of the inner variable at the previous iteration.
    d_inner : array
        Direction for the inner variable.
    d_outer : array
        Direction for the outer variable.
    state_inner_sampler : dict
        State of the sampler for the inner function.
    state_outer_sampler : dict
        State of the sampler for the outer function.
    """
    def iter(i, args):
        # Update inner direction
        start_inner, *_, args['state_inner_sampler'] = inner_sampler(
            args['state_inner_sampler']
        )
        grad_inner, cross_v = jax.vjp(
            lambda x: grad_inner_fun(args['inner_var'], x, start_inner),
            args['outer_var']
        )
        grad_inner_old, cross_v_old = jax.vjp(
            lambda x: grad_inner_fun(args['inner_var_old'], x, start_inner),
            args['outer_var']
        )
        args['d_inner'] += grad_inner - grad_inner_old

        # Update outer direction
        start_outer, *_, args['state_outer_sampler'] = outer_sampler(
            args['state_outer_sampler']
        )
        grad_outer, impl_grad = grad_outer_fun(
            args['inner_var'], args['outer_var'], start_outer
        )
        grad_outer_old, impl_grad_old = grad_outer_fun(
            args['inner_var_old'], args['outer_var'], start_outer
        )
        ihvp, ihvp_old, args['state_inner_sampler'] = joint_shia(
            args['inner_var'], args['outer_var'], grad_outer,
            args['inner_var_old'], args['outer_var'], grad_outer_old,
            args['state_inner_sampler'], shia_lr, sampler=inner_sampler,
            n_steps=n_shia_steps, grad_inner=grad_inner_fun
        )

        impl_grad -= cross_v(ihvp)[0]
        impl_grad_old -= cross_v_old(ihvp_old)[0]

        args['d_outer'] += impl_grad - impl_grad_old

        # Update inner variable and memory
        args['inner_var_old'] = args['inner_var'].copy()
        args['inner_var'] -= step_size * args['d_inner']

        return args
    res = jax.lax.fori_loop(0, n_steps, iter, dict(
        inner_var=inner_var, outer_var=outer_var, inner_var_old=inner_var_old,
        d_inner=d_inner, d_outer=d_outer,
        state_inner_sampler=state_inner_sampler,
        state_outer_sampler=state_outer_sampler
    ))

    (inner_var, inner_var_old, d_inner, d_outer, state_inner_sampler,
     state_outer_sampler) = (res['inner_var'], res['inner_var_old'],
                             res['d_inner'], res['d_outer'],
                             res['state_inner_sampler'],
                             res['state_outer_sampler'])

    return inner_var, inner_var_old, d_inner, d_outer, state_inner_sampler, \
        state_outer_sampler

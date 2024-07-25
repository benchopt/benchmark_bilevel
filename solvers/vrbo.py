from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import jax
    import jax.numpy as jnp
    from functools import partial

    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.hessian_approximation import shia_fb_jax
    from benchmark_utils.tree_utils import update_sgd_fn, tree_diff
    from benchmark_utils.hessian_approximation import joint_shia_jax
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler


class Solver(StochasticJaxSolver):
    """Variance Reduction Bilevel Optimizer (VRBO).

    J. Yang, K. Ji, Y. Liang. "Provabily Faster Algorithms for Bilevel
    Optimization". NeurIPS 2021"""
    name = 'VRBO'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_shia_steps': [10],
        'batch_size': [64],
        'period_frac': [128],
        'n_inner_steps': [10],
        **StochasticJaxSolver.parameters
    }

    def init(self):

        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()

        period = self.n_inner_samples + self.n_outer_samples
        period *= self.period_frac
        period /= self.batch_size
        self.period = int(period)

        step_sizes = jnp.array(  # (inner_ss, hia_lr, outer_ss)
            [
                self.step_size,
                self.step_size,
                self.step_size / self.outer_ratio,
            ]
        )
        exponents = jnp.zeros(3)
        state_lr = init_lr_scheduler(step_sizes, exponents)
        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var,
            inner_var_old=self.inner_var.copy(),
            d_inner=jax.tree_util.tree_map(jnp.zeros_like, self.inner_var),
            d_outer=jax.tree_util.tree_map(jnp.zeros_like, self.outer_var),
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
            i_min=0
        )

    def get_step(self, inner_sampler, outer_sampler):

        # Gradients
        grad_inner = jax.grad(self.f_inner, argnums=0)
        grad_outer = jax.grad(self.f_outer, argnums=(0, 1))

        # Full batch gradients
        f_inner_fb = partial(
            self.f_inner, start=0, batch_size=self.n_inner_samples
        )
        f_outer_fb = partial(
            self.f_outer, start=0, batch_size=self.n_outer_samples
        )
        grad_inner_fb = jax.grad(f_inner_fb, argnums=0)
        grad_outer_fb = jax.grad(f_outer_fb, argnums=(0, 1))

        shia_fb = partial(
            shia_fb_jax, grad_inner=grad_inner_fb, n_steps=self.n_shia_steps
        )

        joint_shia = partial(
            joint_shia_jax, grad_inner=grad_inner, n_steps=self.n_shia_steps,
            sampler=inner_sampler
        )

        inner_loop = partial(
            inner_loop_vrbo, n_steps=self.n_inner_steps,
            n_shia_steps=self.n_shia_steps,
            inner_sampler=inner_sampler, outer_sampler=outer_sampler,
            joint_shia=joint_shia, grad_inner_fun=grad_inner,
            grad_outer_fun=grad_outer

        )

        def fb_directions(inner_var, outer_var, hia_lr, d_inner, d_outer):
            grad_inner, cross_v = jax.vjp(
                lambda x: grad_inner_fb(inner_var, x), outer_var
            )
            grad_outer_in, grad_outer_out = grad_outer_fb(
                inner_var, outer_var
            )
            v = shia_fb(inner_var, outer_var, grad_outer_in, hia_lr)
            d_inner = grad_inner
            d_outer = tree_diff(
                grad_outer_out,
                cross_v(v)[0]
            )
            return d_inner, d_outer

        def identity_directions(inner_var, outer_var, hia_lr, d_inner,
                                d_outer):
            return d_inner, d_outer

        def vrbo_one_iter(carry, i):
            (inner_lr, hia_lr, outer_lr), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Step.1 - (Re)initialize directions for z and x
            carry['d_inner'], carry['d_outer'] = jax.lax.cond(
                i % self.period == 0, fb_directions, identity_directions,
                carry['inner_var'], carry['outer_var'], hia_lr,
                carry['d_inner'],  carry['d_outer']
            )
            # Step.2 - Update outer variable
            carry['outer_var'] = update_sgd_fn(carry['outer_var'],
                                               carry['d_outer'], outer_lr)

            carry['inner_var'], carry['inner_var_old'], carry['d_inner'], \
                carry['d_outer'], carry['state_inner_sampler'], \
                carry['state_outer_sampler'] = inner_loop(
                    carry['inner_var'], carry['outer_var'],
                    carry['inner_var_old'], carry['d_inner'], carry['d_outer'],
                    carry['state_inner_sampler'], carry['state_outer_sampler'],
                    inner_lr, hia_lr
                )

            return carry, None
        return vrbo_one_iter

    # Needs to be redifined to handle i_min
    def run(self, callback):
        carry = self.init()
        i_min = 0

        # Start algorithm
        while callback():
            carry = self.one_epoch(carry, self.eval_freq, i_min)
            self.inner_var = carry["inner_var"]
            self.outer_var = carry["outer_var"]
            i_min += self.eval_freq

    def get_one_epoch_jitted(self, inner_sampler, outer_sampler):
        step = self.get_step(inner_sampler, outer_sampler)

        def one_epoch(carry, eval_freq, i_min):
            carry, _ = jax.lax.scan(
                step, init=carry,
                length=eval_freq,
                xs=jnp.arange(eval_freq)+i_min
            )
            return carry

        return jax.jit(
            one_epoch, static_argnums=(1,)
        )


def inner_loop_vrbo(inner_var, outer_var, inner_var_old, d_inner, d_outer,
                    state_inner_sampler, state_outer_sampler, step_size,
                    shia_lr, n_shia_steps=1, inner_sampler=None,
                    outer_sampler=None, joint_shia=None, n_steps=1,
                    grad_inner_fun=None, grad_outer_fun=None):
    """
    Jax implementation of the inner routine of VRBO.

    Parameters
    ----------
    inner_var : pytree
        Initial value of the inner variable.
    outer_var : pytree
        Value of the outer variable.
    inner_var_old : pytree
        Value of the inner variable at the previous iteration.
    d_inner : pytree
        Direction for the inner variable.
    d_outer : pytree
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
        args['d_inner'] = update_sgd_fn(
            args['d_inner'], tree_diff(grad_inner, grad_inner_old), -1
        )  # d_inner = d_inner + grad_inner - grad_inner_old

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

        # impl_grad = impl_grad - cross_v(ihvp)[0]
        impl_grad = update_sgd_fn(impl_grad, cross_v(ihvp)[0], 1)
        # impl_grad_old = impl_grad_old - cross_v_old(ihvp_old)[0]
        impl_grad_old = update_sgd_fn(impl_grad_old, cross_v_old(ihvp_old)[0],
                                      1)
        args['d_outer'] = update_sgd_fn(
            args['d_outer'], tree_diff(impl_grad, impl_grad_old), -1
        )

        # Update inner variable and memory
        args['inner_var_old'] = args['inner_var'].copy()
        args['inner_var'] = update_sgd_fn(args['inner_var'],
                                          args['d_inner'],
                                          step_size)

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

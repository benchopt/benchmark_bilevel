from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import jax
    import jax.numpy as jnp
    from functools import partial

    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler


class Solver(StochasticJaxSolver):
    """Fully First-order Stochastic Approximation (F2SA).

    J. Kwon, D. Kwon, S. Wright and R. Noewak, "A Fully First-Order Method for
    Stochastic Bilevel Optimization", ICML 2023."""
    name = 'F2SA'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.01],
        'outer_ratio': [1.],
        'batch_size': [64],
        'lmbda0': [1.],
        'delta_lmbda': [.1],
        'n_inner_steps': [10],
        **StochasticJaxSolver.parameters
    }

    def init(self):
        # Init variables
        self.inner_var = self.inner_var0.copy()
        inner_approx_star = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()
        lmbda = self.lmbda0

        step_sizes = jnp.array(
                [self.step_size,
                 self.step_size,
                 self.step_size / self.outer_ratio,
                 self.delta_lmbda]
            )
        exponents = jnp.array(
                [5/7, 4/7, 4/7, 1/7]
            )
        state_lr = init_lr_scheduler(step_sizes, exponents)
        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var,
            inner_approx_star=inner_approx_star, lmbda=lmbda,
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )

    def get_step(self, inner_sampler, outer_sampler):

        grad_inner = jax.grad(self.f_inner, argnums=1)
        grad_outer_outer_var = jax.grad(self.f_outer, argnums=1)

        inner_loop = partial(
                inner_f2sa_jax,
                inner_sampler=inner_sampler,
                outer_sampler=outer_sampler,
                grad_inner=jax.grad(self.f_inner, argnums=0),
                grad_outer=jax.grad(self.f_outer, argnums=0),
                n_steps=self.n_inner_steps
            )

        def f2sa_one_iter(carry, _):

            step_sizes, carry['state_lr'] = update_lr(
                carry['state_lr']
            )
            lr_inner, lr_approx_star, lr_outer, d_lmbda = step_sizes

            # Run the inner procedure
            carry['inner_var'], carry['inner_approx_star'], \
                carry['state_inner_sampler'], carry['state_outer_sampler'] = \
                inner_loop(
                    carry['inner_var'], carry['inner_approx_star'],
                    carry['outer_var'], carry['lmbda'],
                    carry['state_inner_sampler'], carry['state_outer_sampler'],
                    inner_sampler=inner_sampler, outer_sampler=outer_sampler,
                    lr_inner=lr_inner, lr_approx_star=lr_approx_star
                )

            # Compute oracles and the update direction of the outer variable
            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            start_inner1, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            start_inner2, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            d_outer_var = grad_outer_outer_var(
                carry['inner_var'], carry['outer_var'], start_outer
            )
            grad_inner_outer = grad_inner(
                carry['inner_var'], carry['outer_var'], start_inner1
            )
            grad_inner_star = grad_inner(
                carry['inner_approx_star'], carry['outer_var'], start_inner2
            )
            d_outer_var += carry['lmbda'] * (grad_inner_outer
                                             - grad_inner_star)

            # Update inner variable with SGD.
            carry['outer_var'] -= lr_outer * d_outer_var
            carry['lmbda'] += d_lmbda

            return carry, _

        return f2sa_one_iter


def inner_f2sa_jax(inner_var, inner_approx_star,  outer_var, lmbda,
                   state_inner_sampler, state_outer_sampler,
                   inner_sampler=None, outer_sampler=None,
                   lr_inner=.1, lr_approx_star=.1, n_steps=10, grad_inner=None,
                   grad_outer=None):
    """
    Jax implementation of the inner loop of F2SA algorithm.

    Parameters
    ----------
    inner_var : array, shape (d_inner,)
        Initial inner variable.

    inner_approx_star : array, shape (d_inner,)
        Initial inner variable.

    outer_var : array, shape (d_outer,)
        Outer variable.

    lmbda : float
        Lagrange multiplier.

    state_inner_sampler : dict
        State of the inner sampler.

    state_outer_sampler : dict
        State of the outer sampler.

    inner_sampler : callable
        Inner sampler.

    outer_sampler : callable
        Outer sampler.

    lr_inner : float
        Learning rate for the inner variable.

    lr_approx_star : float
        Learning rate for the lagrangian inner variable.

    n_steps : int
        Number of steps of the loop.

    grad_inner : callable
        Gradient of the inner oracle with respect to the inner variable.

    grad_outer : callable
        Gradient of the outer oracle with respect to the inner variable.

    Returns
    -------
    inner_var : array, shape (d_inner,)
        Updated inner variable.

    inner_approx_star : array, shape (d_inner,)
        Updated inner variable to approximate g^*.

    state_inner_sampler : dict
        Updated state of the inner sampler.

    state_outer_sampler : dict
        Updated state of the outer sampler.
    """
    def iter(i, args):
        (inner_var, inner_approx_star, state_inner_sampler,
         state_outer_sampler) = args
        # Get the batches and oracles
        start_idx_inner, *_, state_inner_sampler = inner_sampler(
            state_inner_sampler
        )
        start_idx_lagrangian, *_, state_inner_sampler = inner_sampler(
            state_inner_sampler
        )
        start_idx_outer, *_, state_outer_sampler = outer_sampler(
            state_outer_sampler
        )

        d_inner_var = lmbda * grad_inner(
            inner_var, outer_var, start_idx_inner
        )
        d_inner_var += grad_outer(inner_var, outer_var, start_idx_outer)
        d_inner_approx_star = grad_inner(
            inner_approx_star, outer_var, start_idx_lagrangian
        )

        # # Update the variables
        inner_var -= lr_inner * d_inner_var
        inner_approx_star -= lr_approx_star * d_inner_approx_star
        return (inner_var, inner_approx_star, state_inner_sampler,
                state_outer_sampler)
    (inner_var, inner_approx_star, state_inner_sampler,
     state_outer_sampler) = jax.lax.fori_loop(
        0, n_steps, iter, (inner_var, inner_approx_star,
                           state_inner_sampler, state_outer_sampler)
    )
    return (inner_var, inner_approx_star, state_inner_sampler,
            state_outer_sampler)

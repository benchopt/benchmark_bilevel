from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import jax
    import jax.numpy as jnp
    from functools import partial

    from benchmark_utils.sgd_inner import sgd_inner_jax
    from benchmark_utils.hessian_approximation import shia_jax
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.tree_utils import update_sgd_fn, tree_diff
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler


class Solver(StochasticJaxSolver):
    """Stochastic Bilevel Optimizer (stocBIO).

    K. Ji, J. Yang and Y. Liang. "Bilevel Optimization: Convergence Analysis
    and Enhanced Design". ICML 2021."""
    name = 'StocBiO'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_inner_steps': [10],
        'batch_size': [64],
        'n_shia_steps': [10],
        **StochasticJaxSolver.parameters
    }

    def init(self):
        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()

        step_sizes = jnp.array(
            [self.step_size, self.step_size,
             self.step_size / self.outer_ratio]
        )
        exponents = jnp.zeros(3)
        state_lr = init_lr_scheduler(step_sizes, exponents)

        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var,
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )

    def get_step(self, inner_sampler, outer_sampler):
        grad_inner = jax.grad(self.f_inner, argnums=0)
        grad_outer = jax.grad(self.f_outer, argnums=(0, 1))

        sgd_inner = partial(
            sgd_inner_jax, n_steps=self.n_inner_steps, sampler=inner_sampler,
            grad_inner=grad_inner
        )

        shia = partial(
            shia_jax, n_steps=self.n_shia_steps, grad_inner=grad_inner,
            sampler=inner_sampler
        )

        def stocbio_one_iter(carry, _):
            (inner_lr, hia_lr, outer_lr), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Update the inner variable
            carry['inner_var'], carry['state_inner_sampler'] = sgd_inner(
                carry['inner_var'], carry['outer_var'],
                carry['state_inner_sampler'], step_size=inner_lr)

            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            grad_in, grad_out = grad_outer(carry['inner_var'],
                                           carry['outer_var'],
                                           start_outer)

            # Compute the approximate iHVP with Neumann iterations
            implicit_grad, carry['state_inner_sampler'] = shia(
                carry['inner_var'], carry['outer_var'], grad_in,
                carry['state_inner_sampler'], hia_lr
            )
            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            _, vjp_fun = jax.vjp(
                lambda x: grad_inner(carry['inner_var'], x, start_inner),
                carry['outer_var']
            )
            implicit_grad = vjp_fun(implicit_grad)[0]
            grad_outer_var = tree_diff(grad_out, implicit_grad)

            # Update the outer variable
            carry['outer_var'] = update_sgd_fn(
                carry['outer_var'], grad_outer_var, outer_lr)

            return carry, _
        return stocbio_one_iter

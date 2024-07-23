from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import jax
    import jax.numpy as jnp
    from functools import partial

    from benchmark_utils.tree_utils import update_sgd_fn
    from benchmark_utils.hessian_approximation import hia_jax
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.tree_utils import tree_add, tree_scalar_mult
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler


class Solver(StochasticJaxSolver):
    """Two-Timescale Stochastic Approximation (TTSA).

    M. Hong, H.-T. Wai and Z. Yang. "A Two-Timescale Framework for Bilevel
    Optimization: Complexity Analysis and Application to Actor-Critic". SIAM
    Journal of Optimization. 2023"""
    name = 'TTSA'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_hia_steps': [10],
        'batch_size': [64],
        **StochasticJaxSolver.parameters
    }

    def init(self):
        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()

        # Init lr scheduler
        step_sizes = jnp.array(
            [self.step_size, self.step_size,
             self.step_size / self.outer_ratio]
        )
        exponents = jnp.array([.4, 0., .6])
        state_lr = init_lr_scheduler(step_sizes, exponents)
        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var,
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
            key=jax.random.key(self.random_state)
        )

    def get_step(self, inner_sampler, outer_sampler):
        grad_inner = jax.grad(self.f_inner, argnums=0)
        grad_outer = jax.grad(self.f_outer, argnums=(0, 1))

        hia = partial(
            hia_jax, n_steps=self.n_hia_steps, sampler=inner_sampler,
            grad_inner=grad_inner
        )

        def ttsa_one_iter(carry, _):

            (inner_lr, hia_lr, outer_lr), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Step.1 - Update direction for z with momentum
            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            grad_inner_var = grad_inner(
                carry['inner_var'], carry['outer_var'],
                start_inner
            )

            # Step.2 - Update the inner variable
            carry['inner_var'] = update_sgd_fn(carry['inner_var'],
                                               grad_inner_var,
                                               inner_lr)

            # Step.3 - Compute implicit grad approximation with HIA
            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            grad_in, grad_out = grad_outer(
                carry['inner_var'], carry['outer_var'], start_outer
            )

            v, carry['key'], carry['state_inner_sampler'] = hia(
                carry['inner_var'], carry['outer_var'], grad_in,
                carry['state_inner_sampler'],
                hia_lr, key=carry['key']
            )

            _, vjp_fun = jax.vjp(
                lambda x: grad_inner(carry['inner_var'], x, start_inner),
                carry['outer_var']
            )
            implicit_grad = vjp_fun(v)[0]
            grad_outer_var = tree_add(grad_out,
                                      tree_scalar_mult(-1, implicit_grad))

            # Step.4 - update the outer variables
            carry['outer_var'] = update_sgd_fn(
                carry['outer_var'], grad_outer_var, outer_lr
            )
            return carry, _
        return ttsa_one_iter

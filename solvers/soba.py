from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp


class Solver(StochasticJaxSolver):
    """Stochastic Bilevel Algorithm (SOBA).

    M. Dagréou, P. Ablin, S. Vaiter and T. Moreau, "A framework for bilevel
    optimization that enables stochastic and global variance reduction
    algorithms", NeurIPS 2022."""
    name = 'SOBA'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'batch_size': [64],
        **StochasticJaxSolver.parameters
    }

    def init(self):
        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()
        v = jax.tree_util.tree_map(jnp.zeros_like, self.inner_var)

        # Init lr scheduler
        step_sizes = jnp.array(
            [self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = jnp.array(
            [.5, .5]
        )
        state_lr = init_lr_scheduler(step_sizes, exponents)
        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var, v=v,
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )

    def get_step(self, inner_sampler, outer_sampler):

        grad_inner = jax.grad(self.f_inner, argnums=0)
        grad_outer = jax.grad(self.f_outer, argnums=(0, 1))

        def soba_one_iter(carry, _):

            (inner_step_size, outer_step_size), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Step.1 - get all gradients and compute the implicit gradient.
            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            grad_inner_var, vjp_train = jax.vjp(
                lambda z, x: grad_inner(z, x, start_inner), carry['inner_var'],
                carry['outer_var']
            )
            hvp, cross_v = vjp_train(carry['v'])

            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            grad_in_outer, grad_out_outer = grad_outer(
                carry['inner_var'], carry['outer_var'], start_outer
            )

            # Step.2 - update inner variable with SGD.
            def update_sgd_fn(var, grad, step_size):
                return jax.tree_util.tree_map(lambda x, y: x - step_size * y,
                                              var, grad)

            carry['inner_var'] = update_sgd_fn(
                carry['inner_var'], grad_inner_var, inner_step_size
            )
            carry['v'] = update_sgd_fn(
                carry['v'],
                jax.tree_util.tree_map(jnp.add, hvp, grad_in_outer),
                inner_step_size
            )
            carry['outer_var'] = update_sgd_fn(
                carry['outer_var'],
                jax.tree_util.tree_map(jnp.add, cross_v, grad_out_outer),
                outer_step_size
            )
            return carry, _

        return soba_one_iter

from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    from benchmark_utils.tree_utils import tree_scalar_mult
    from benchmark_utils.tree_utils import update_sgd_fn, tree_add
    from benchmark_utils.tree_utils import init_memory_of_trees, select_memory

    import jax
    import jax.numpy as jnp


class Solver(StochasticJaxSolver):
    """Fully Single Loop Algorithm (FSLA).

    J. Li, B. Gu and H. Huang. "A Fully Single Loop Algorithm for Bilevel
    Optimization without Hessian Inverse". AAAI 2022"""
    name = 'FSLA'

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
            [self.step_size, self.step_size,
             self.step_size / self.outer_ratio]
        )
        exponents = jnp.ones(len(step_sizes)) * 0.5
        state_lr = init_lr_scheduler(step_sizes, exponents)

        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var, v=v,
            memory_outer=init_memory_of_trees(2, self.outer_var),
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )

    def get_step(self, inner_sampler, outer_sampler):
        grad_inner = jax.grad(self.f_inner, argnums=0)
        grad_outer = jax.grad(self.f_outer, argnums=(0, 1))

        def fsla_one_iter(carry, _):

            (inner_lr, eta, outer_lr), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Step.1 - SGD step on the inner problem
            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            grad_inner_var = grad_inner(carry['inner_var'],
                                        carry['outer_var'],
                                        start_inner)
            inner_var_old = carry['inner_var'].copy()
            carry['inner_var'] = update_sgd_fn(carry['inner_var'],
                                               grad_inner_var,
                                               inner_lr)

            # Step.2 - SGD step on the auxillary variable v
            start_inner2, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            _, hvp_fun = jax.vjp(
                lambda z: grad_inner(z, carry['outer_var'], start_inner2),
                carry['inner_var']
            )

            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            grad_outer_in, _ = grad_outer(carry['inner_var'],
                                          carry['outer_var'],
                                          start_outer)
            v_old = carry['v'].copy()
            carry['v'] = update_sgd_fn(
                carry['v'],
                tree_add(hvp_fun(carry['v'])[0],
                         tree_scalar_mult(-1, grad_outer_in)),
                inner_lr
            )

            # Step.3 - compute the implicit gradient estimates, for the old
            # and new variables
            start_outer2, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            _, impl_grad = grad_outer(
                carry['inner_var'], carry['outer_var'], start_outer2
            )
            _, impl_grad_old = grad_outer(
                inner_var_old, carry['memory_outer'][0], start_outer2
            )
            start_inner3, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            _, cross_v_fun = jax.vjp(
                lambda x: grad_inner(carry['inner_var'], x, start_inner3),
                carry['outer_var']
            )
            _, cross_v_fun_old = jax.vjp(
                lambda x: grad_inner(inner_var_old, x, start_inner3),
                carry['memory_outer'][0]
            )
            impl_grad = update_sgd_fn(impl_grad,
                                      cross_v_fun(carry['v'])[0],
                                      1)
            impl_grad_old = update_sgd_fn(impl_grad_old,
                                          cross_v_fun_old(v_old)[0],
                                          1)

            # Step.4 - update direction with momentum
            carry['memory_outer'] = carry['memory_outer'].at[1].set(
                tree_add(
                    impl_grad,
                    tree_scalar_mult(
                        (1-eta),
                        tree_add(carry['memory_outer'][1],
                                 tree_scalar_mult(-1, impl_grad_old))
                    )
                )
            )

            # Step.5 - update the outer variable
            carry['memory_outer'] = carry['memory_outer'].at[0].set(
                carry['outer_var']
            )
            carry['outer_var'] = update_sgd_fn(
                carry['outer_var'],
                select_memory(carry['memory_outer'], 1),
                outer_lr
            )
            return carry, _
        return fsla_one_iter

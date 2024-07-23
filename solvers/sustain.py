from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver


from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import jax
    import jax.numpy as jnp
    from functools import partial

    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.tree_utils import update_sgd_fn, tree_add
    from benchmark_utils.hessian_approximation import joint_hia_jax
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler
    from benchmark_utils.tree_utils import tree_scalar_mult, select_memory
    from benchmark_utils.tree_utils import update_memory, init_memory_of_trees


class Solver(StochasticJaxSolver):
    """SUSTAIN.

    P. Khanduri, S. Zeng, M. Hong, H.-T. Wai, Z. Wang and Z. Yang. "A
    Near-Optimal Algorithm for Stochastic Bilevel Optimization via
    Double-Momentum". NeurIPS 2021"""
    name = 'SUSTAIN'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [.1],
        'outer_ratio': [1.],
        'n_hia_steps': [10],
        'batch_size': [64],
        'eta': [.5],
        **StochasticJaxSolver.parameters
    }

    def init(self):
        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()

        memory_inner = init_memory_of_trees(2, self.inner_var)
        memory_outer = init_memory_of_trees(2, self.outer_var)
        step_sizes = jnp.array(  # (inner_ss, hia_lr, eta, outer_ss)
            [
                self.step_size,
                self.step_size,
                self.eta,
                self.step_size / self.outer_ratio,
            ]
        )
        exponents = jnp.array([1/3, 0, 2/3, 1/3])
        state_lr = init_lr_scheduler(step_sizes, exponents)
        keys = jax.random.split(jax.random.PRNGKey(self.random_state), 3)

        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var,
            memory_inner=memory_inner, memory_outer=memory_outer,
            state_lr=state_lr,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
            key=keys[-1]
        )

    def get_step(self, inner_sampler, outer_sampler):
        grad_inner_fun = jax.grad(self.f_inner, argnums=0)
        grad_outer_fun = jax.grad(self.f_outer, argnums=(0, 1))

        joint_hia = partial(
            joint_hia_jax, grad_inner=grad_inner_fun, n_steps=self.n_hia_steps,
            sampler=inner_sampler
        )

        def sustain_one_iter(carry, _):

            (inner_lr, hia_lr, eta, outer_lr), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Step.1 - Update direction for z with momentum
            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            grad_inner_var, vjp_fun = jax.vjp(
                lambda x: grad_inner_fun(carry['inner_var'], x, start_inner),
                carry['outer_var']
            )
            grad_inner_var_old, vjp_fun_old = jax.vjp(
                lambda x: grad_inner_fun(
                    select_memory(carry['memory_inner'], 0), x,
                    start_inner),
                select_memory(carry['memory_outer'], 0)
            )

            carry['memory_inner'] = carry['memory_inner'].at[1].set(
                grad_inner_var
                + (1-eta) * (select_memory(carry['memory_inner'], 1)
                             - grad_inner_var_old)
            )

            # Step.2 - Compute implicit grad approximation with HIA
            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )
            grad_outer, impl_grad = grad_outer_fun(
                carry['inner_var'], carry['outer_var'], start_outer
            )
            grad_outer_old, impl_grad_old = grad_outer_fun(
                select_memory(carry['memory_inner'], 0),
                select_memory(carry['memory_outer'], 0),
                start_outer
            )

            ihvp, ihvp_old, carry['key'], carry['state_inner_sampler'] = (
                joint_hia(
                    carry['inner_var'], carry['outer_var'], grad_outer,
                    select_memory(carry['memory_inner'][0]),
                    select_memory(carry['memory_outer'], 0),
                    grad_outer_old, carry['state_inner_sampler'], hia_lr,
                    sampler=inner_sampler, key=carry['key'],
                    grad_inner=grad_inner_fun
                )
            )
            impl_grad -= vjp_fun(ihvp)[0]
            impl_grad_old -= vjp_fun_old(ihvp_old)[0]

            # Step.3 - Update direction for x with momentum
            carry['memory_outer'] = update_memory(
                carry['memory_outer'], 1,
                tree_add(
                    impl_grad,
                    tree_scalar_mult(1-eta, tree_add(
                        select_memory(carry['memory_outer'], 1),
                        tree_scalar_mult(-1, impl_grad_old)
                    ))
                )
            )

            # Step.4 - Save the current variables
            carry['memory_inner'] = update_memory(carry['memory_inner'], 0,
                                                  carry['inner_var'])
            carry['memory_outer'] = update_memory(carry['memory_outer'], 0,
                                                  carry['outer_var'])

            # Step.5 - update the variables with the directions
            carry['inner_var'] = update_sgd_fn(
                carry['inner_var'],
                select_memory(carry['memory_inner'], 1),
                inner_lr
            )
            carry['outer_var'] = update_sgd_fn(
                carry['outer_var'],
                select_memory(carry['memory_outer'], 1),
                outer_lr
            )
            return carry, _
        return sustain_one_iter
